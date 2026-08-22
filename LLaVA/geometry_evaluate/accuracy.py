#!/usr/bin/env python3
from __future__ import annotations

"""
Geometry accuracy evaluator.

Metrics
1. Loose Accuracy

2. Constrained Accuracy

3. Semantic Accuracy

Outputs
1. accuracy_results.md
2. Per-model/per-dataset JSONL prediction logs

Shared infrastructure:
    helper.py

Shared geometry definitions:
    ontology.py
"""

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from tqdm import tqdm

from helper import (
    EvaluationModelPaths,
    filter_task_rows,
    generate_response,
    is_cuda_oom,
    load_eval_model,
    load_json_value,
    read_parquet_files,
    stable_sample_key,
    stratified_limit,
    unload_model,
    validate_concepts,
    validate_required_columns,
    warn_expected_count,
)

from ontology import (
    CONCEPT_ALIASES,
    SEMANTIC_UNDIRECTED_SEGMENT_FIELDS,
    SEMANTIC_UNORDERED_NESTED_LIST_FIELDS,
    SEMANTIC_UNORDERED_STRING_LIST_FIELDS,
    STAGE1_CONCEPTS,
    STAGE2_CONCEPTS,
)


# =============================================================================
# Evaluation constants
# =============================================================================

MODEL_ORDER = (
    "base",
    "stage1",
    "stage2",
    "stage2_only",
)

# Stage2-only must also be evaluated on Stage1.
# This is necessary to compare primitive retention/emergence between:
#
#   Base -> Stage2-only
#   Base -> Stage1 -> Stage2
#
STAGE1_MODEL_ORDER = MODEL_ORDER
STAGE2_MODEL_ORDER = MODEL_ORDER

DEFAULT_STAGE1_PATTERN = "stage1_geometry_evaluate_evaluate_*.parquet"
DEFAULT_STAGE2_PATTERN = "stage2_geometry_evaluate_evaluate_*.parquet"


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate geometry concept and semantic accuracy."
    )

    # Data
    parser.add_argument("--stage1-data-dir", type=Path, required=True)
    parser.add_argument("--stage2-data-dir", type=Path, required=True)

    parser.add_argument(
        "--stage1-pattern",
        default=DEFAULT_STAGE1_PATTERN,
    )
    parser.add_argument(
        "--stage2-pattern",
        default=DEFAULT_STAGE2_PATTERN,
    )

    # Models
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--stage1-dir", type=Path, required=True)
    parser.add_argument("--stage2-dir", type=Path, required=True)
    parser.add_argument("--stage2-only-dir", type=Path, required=True)

    # Output
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-file",
        default="accuracy_results.md",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-sample JSONL logs. "
            "Defaults to <output-dir>/accuracy_predictions."
        ),
    )

    # Runtime
    parser.add_argument("--conv-mode", default="llava_v1")

    parser.add_argument(
        "--constrained-seed",
        type=int,
        default=20260822,
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=20260822,
    )

    parser.add_argument(
        "--max-new-tokens-loose",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--max-new-tokens-constrained",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max-new-tokens-semantic",
        type=int,
        default=192,
    )

    parser.add_argument(
        "--max-sample-errors",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help=(
            "0 evaluates the full set. Positive values use "
            "deterministic concept-balanced sampling."
        ),
    )

    parser.add_argument(
        "--expected-stage1-rows",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--expected-stage2-local-rows",
        type=int,
        default=2700,
    )

    return parser.parse_args()


# =============================================================================
# Dataset discovery / validation
# =============================================================================

def discover_parquets(
    directory: Path,
    pattern: str,
    dataset_name: str,
) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(
            f"{dataset_name} directory was not found: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"{dataset_name} path is not a directory: {directory}"
        )

    paths = sorted(
        path
        for path in directory.glob(pattern)
        if path.is_file()
    )

    if not paths:
        raise FileNotFoundError(
            f"No {dataset_name} parquet matched "
            f"{directory / pattern}"
        )

    print(
        f"{dataset_name}: discovered "
        f"{len(paths):,} shard(s)."
    )
    for path in paths:
        print(f"  - {path}")

    return paths


def sort_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = list(enumerate(rows))

    indexed.sort(
        key=lambda item: (
            str(item[1].get("concept") or ""),
            stable_sample_key(item[1], item[0]),
        )
    )

    return [row for _, row in indexed]


def validate_semantic_targets(
    rows: Sequence[dict[str, Any]],
    dataset_name: str,
) -> None:
    """
    Validate every semantic target before loading a model.

    Parsed targets are cached into `_semantic_target`.
    """

    for index, row in enumerate(rows):
        concept = str(row["concept"])

        try:
            target = load_json_value(
                row["semantic_target_json"]
            )
        except Exception as exc:
            raise ValueError(
                f"{dataset_name} row {index}: "
                "semantic_target_json could not be decoded."
            ) from exc

        if not isinstance(target, dict):
            raise ValueError(
                f"{dataset_name} row {index}: "
                "semantic_target_json must decode to an object."
            )

        target_concept = str(target.get("concept"))

        if target_concept != concept:
            raise ValueError(
                f"{dataset_name} row {index}: "
                f"concept mismatch: row={concept!r}, "
                f"target={target_concept!r}."
            )

        entities = target.get("entities")

        if not isinstance(entities, dict):
            raise ValueError(
                f"{dataset_name} row {index}: "
                "semantic_target_json['entities'] must be an object."
            )

        row["_semantic_target"] = target


def prepare_stage1_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    paths = discover_parquets(
        args.stage1_data_dir,
        args.stage1_pattern,
        "Stage1",
    )

    rows = read_parquet_files(paths, "Stage1")

    validate_required_columns(
        rows,
        {"concept", "image", "semantic_target_json"},
        "Stage1",
    )

    validate_concepts(
        rows,
        STAGE1_CONCEPTS,
        "Stage1",
    )

    validate_semantic_targets(rows, "Stage1")

    warn_expected_count(
        len(rows),
        args.expected_stage1_rows,
        "Stage1 evaluation set",
    )

    rows = sort_rows(rows)

    rows = stratified_limit(
        rows,
        STAGE1_CONCEPTS,
        args.max_samples,
        seed=args.sample_seed,
    )

    print(f"Stage1 accuracy rows: {len(rows):,}")
    return rows


def prepare_stage2_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    paths = discover_parquets(
        args.stage2_data_dir,
        args.stage2_pattern,
        "Stage2",
    )

    rows = read_parquet_files(paths, "Stage2")

    validate_required_columns(
        rows,
        {
            "concept",
            "image",
            "semantic_target_json",
            "task_kind",
        },
        "Stage2",
    )

    rows = filter_task_rows(rows, "local")

    if not rows:
        raise RuntimeError(
            "Stage2 contains no task_kind='local' rows."
        )

    validate_concepts(
        rows,
        STAGE2_CONCEPTS,
        "Stage2 local",
    )

    validate_semantic_targets(rows, "Stage2 local")

    warn_expected_count(
        len(rows),
        args.expected_stage2_local_rows,
        "Stage2 local evaluation set",
    )

    rows = sort_rows(rows)

    rows = stratified_limit(
        rows,
        STAGE2_CONCEPTS,
        args.max_samples,
        seed=args.sample_seed,
    )

    print(f"Stage2 accuracy rows: {len(rows):,}")
    return rows


# =============================================================================
# Loose concept scoring
# =============================================================================

def normalize_text(text: str) -> str:
    text = (
        str(text)
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def aliases_for_concept(concept: str) -> list[str]:
    aliases = set(CONCEPT_ALIASES.get(concept, ()))

    # Canonical label is always permitted in Loose scoring.
    aliases.add(concept.replace("_", " "))

    normalized = {
        normalize_text(alias)
        for alias in aliases
        if alias
    }

    return sorted(
        normalized,
        key=lambda alias: (-len(alias), alias),
    )


def predict_loose_concept(
    text: str,
    concepts: Iterable[str],
) -> str | None:
    """
    Longest matched alias wins.

    If multiple concepts tie at the same longest matched alias length,
    return None rather than guessing.
    """

    normalized = normalize_text(text)
    matches: list[tuple[int, str]] = []

    for concept in concepts:
        for alias in aliases_for_concept(concept):
            pattern = (
                rf"(?<![a-z0-9])"
                rf"{re.escape(alias)}"
                rf"(?:s)?"
                rf"(?![a-z0-9])"
            )

            if re.search(pattern, normalized):
                matches.append((len(alias), concept))
                break

    if not matches:
        return None

    best_length = max(length for length, _ in matches)

    best = sorted(
        {
            concept
            for length, concept in matches
            if length == best_length
        }
    )

    return best[0] if len(best) == 1 else None


# =============================================================================
# Prompts / constrained classification
# =============================================================================

def original_loose_prompt(
    row: dict[str, Any],
    dataset_name: str,
) -> str:
    for key in ("prompt", "question", "instruction"):
        value = row.get(key)

        if value is not None and str(value).strip():
            return str(value)

    if dataset_name == "stage1":
        return "What basic geometric object is shown in this image?"

    return "What geometric relation or property is shown in this image?"


def shuffled_candidates(
    concepts: tuple[str, ...],
    sample_id: str,
    seed: int,
) -> list[str]:
    """
    Candidate ordering is deterministic per sample and therefore identical
    across all evaluated checkpoints.
    """

    digest = hashlib.sha256(
        f"{seed}:{sample_id}".encode("utf-8")
    ).digest()

    sample_seed = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    rng = random.Random(sample_seed)

    candidates = list(concepts)
    rng.shuffle(candidates)

    return candidates


def build_constrained_prompt(
    dataset_name: str,
    concepts: tuple[str, ...],
    sample_id: str,
    seed: int,
) -> tuple[str, list[str]]:
    candidates = shuffled_candidates(
        concepts,
        sample_id,
        seed,
    )

    target = (
        "basic geometric object"
        if dataset_name == "stage1"
        else "geometric relation or property"
    )

    options = "\n".join(
        f"- {label}"
        for label in candidates
    )

    prompt = (
        f"Classify the {target} shown in the image.\n"
        "Choose exactly ONE label from the candidate list below.\n"
        "Return the canonical label only. Do not explain.\n\n"
        f"Candidates:\n{options}"
    )

    return prompt, candidates


def parse_constrained_prediction(
    text: str,
    concepts: tuple[str, ...],
) -> str | None:
    """
    Recover exactly one canonical concept.

    Parsed Rate therefore measures unique concept recoverability, not strict
    response-format compliance.
    """

    normalized = normalize_text(text)
    found: list[str] = []

    for concept in concepts:
        label = normalize_text(concept)

        pattern = (
            rf"(?<![a-z0-9])"
            rf"{re.escape(label)}"
            rf"(?![a-z0-9])"
        )

        if re.search(pattern, normalized):
            found.append(concept)

    return found[0] if len(found) == 1 else None


# =============================================================================
# Semantic entity scoring
# =============================================================================

def placeholder_schema(value: Any) -> Any:
    """
    Preserve field names and JSON structure while removing gold identities.
    """

    if isinstance(value, dict):
        return {
            str(key): placeholder_schema(subvalue)
            for key, subvalue in value.items()
        }

    if isinstance(value, list):
        return [placeholder_schema(item) for item in value]

    if isinstance(value, str):
        if len(value) == 1 and value.isalpha():
            return "<POINT>"

        if value.isdigit():
            return "<MARKER>"

        return "<STRING>"

    if isinstance(value, bool):
        return "<BOOL>"

    if isinstance(value, (int, float)):
        return "<NUMBER>"

    if value is None:
        return None

    return "<VALUE>"


def build_semantic_prompt(
    concept: str,
    gold_entities: dict[str, Any],
) -> str:
    schema = placeholder_schema(gold_entities)

    return (
        "Identify the specific visible entities in the image that instantiate "
        "the already-selected geometry concept below.\n\n"
        f"Selected concept: {concept}\n\n"
        "Return ONLY the JSON object for `entities`. "
        "Do not add markdown or explanation.\n"
        "Use exactly the field structure shown below and replace placeholders "
        "with the point labels or markers visible in the image.\n\n"
        f"Required entity schema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n\n"
        "Rules:\n"
        "- Point labels must match the image exactly.\n"
        "- A two-point undirected segment may be written in either endpoint order.\n"
        "- Sets/lists of vertices, segments, arms, or markers may be written in any order.\n"
        "- For an angle [A, B, C], B is the vertex; "
        "[A, B, C] and [C, B, A] are equivalent.\n"
        "- Do not invent unlabeled latent quantities."
    )


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    text = str(text).strip()

    if not text:
        return None

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```$", "", text)

    decoder = json.JSONDecoder()

    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(
                text[match.start():]
            )
        except json.JSONDecodeError:
            continue

        if isinstance(value, dict):
            return value

    return None


def normalize_atom(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.strip().split()).upper()

    return value


def sort_json_list(values: list[Any]) -> list[Any]:
    return sorted(
        values,
        key=lambda item: json.dumps(
            item,
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


def canonicalize_semantic(
    value: Any,
    field_name: str | None = None,
) -> Any:
    if isinstance(value, dict):
        return {
            str(key): canonicalize_semantic(subvalue, str(key))
            for key, subvalue in sorted(
                value.items(),
                key=lambda item: str(item[0]),
            )
        }

    if isinstance(value, list):
        # Angle [A, B, C]:
        # B is the vertex, while the two arms may swap.
        if field_name == "angle" and len(value) == 3:
            first = normalize_atom(value[0])
            vertex = normalize_atom(value[1])
            third = normalize_atom(value[2])

            first, third = sorted((first, third), key=str)

            return [first, vertex, third]

        items = [
            canonicalize_semantic(item, None)
            for item in value
        ]

        if (
            field_name in SEMANTIC_UNDIRECTED_SEGMENT_FIELDS
            and len(items) == 2
        ):
            return sorted(items, key=str)

        if field_name in SEMANTIC_UNORDERED_STRING_LIST_FIELDS:
            return sorted(items, key=str)

        if field_name in SEMANTIC_UNORDERED_NESTED_LIST_FIELDS:
            normalized: list[Any] = []

            for item in items:
                if isinstance(item, list) and len(item) == 2:
                    item = sorted(item, key=str)

                normalized.append(item)

            return sort_json_list(normalized)

        # Other evaluation lists are treated as set-like containers.
        return sort_json_list(items)

    return normalize_atom(value)


def semantic_entities_from_response(
    text: str,
) -> dict[str, Any] | None:
    parsed = extract_first_json_object(text)

    if parsed is None:
        return None

    if (
        set(parsed) == {"entities"}
        and isinstance(parsed["entities"], dict)
    ):
        parsed = parsed["entities"]

    return parsed if isinstance(parsed, dict) else None


def semantic_exact_from_response(
    response: str,
    gold_entities: dict[str, Any],
) -> tuple[bool, dict[str, Any] | None]:
    pred_entities = semantic_entities_from_response(response)

    if pred_entities is None:
        return False, None

    correct = (
        canonicalize_semantic(pred_entities)
        == canonicalize_semantic(gold_entities)
    )

    return correct, pred_entities


# =============================================================================
# Errors / JSONL
# =============================================================================

def exception_record(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


def write_jsonl_record(
    file_handle,
    record: dict[str, Any],
) -> None:
    file_handle.write(
        json.dumps(
            record,
            ensure_ascii=False,
        )
        + "\n"
    )

    # Preserve completed generations even if a later sample fails.
    file_handle.flush()


# =============================================================================
# Aggregation
# =============================================================================

def empty_stats() -> Counter:
    return Counter(
        n=0,
        any_error=0,

        loose_correct=0,
        loose_error=0,

        constrained_correct=0,
        constrained_wrong=0,
        constrained_unparsed=0,
        constrained_error=0,

        semantic_valid=0,
        semantic_correct=0,
        semantic_error=0,
        semantic_constrained_valid=0,

        entity_n=0,
        entity_valid=0,
        entity_correct=0,
        entity_error=0,
        entity_constrained_valid=0,
    )


def safe_ratio(
    numerator: int | float,
    denominator: int | float,
) -> float | None:
    if denominator == 0:
        return None

    return float(numerator) / float(denominator)


def update_stats(
    stats: Counter,
    *,
    loose_status: str,
    constrained_status: str,
    semantic_status: str,
    has_entities: bool,
) -> None:
    stats["n"] += 1

    had_error = (
        loose_status == "error"
        or constrained_status == "error"
        or semantic_status == "error"
    )

    stats["any_error"] += int(had_error)

    # -------------------------------------------------------------------------
    # Loose
    # -------------------------------------------------------------------------
    if loose_status == "correct":
        stats["loose_correct"] += 1
    elif loose_status == "error":
        stats["loose_error"] += 1
    elif loose_status != "wrong":
        raise ValueError(f"Unknown loose status: {loose_status!r}")

    # -------------------------------------------------------------------------
    # Constrained
    # -------------------------------------------------------------------------
    if constrained_status == "correct":
        stats["constrained_correct"] += 1
    elif constrained_status == "wrong":
        stats["constrained_wrong"] += 1
    elif constrained_status == "unparsed":
        stats["constrained_unparsed"] += 1
    elif constrained_status == "error":
        stats["constrained_error"] += 1
    else:
        raise ValueError(
            f"Unknown constrained status: {constrained_status!r}"
        )

    # -------------------------------------------------------------------------
    # Semantic
    # -------------------------------------------------------------------------
    if semantic_status in {"correct", "wrong"}:
        stats["semantic_valid"] += 1

        if semantic_status == "correct":
            stats["semantic_correct"] += 1

        if constrained_status == "correct":
            stats["semantic_constrained_valid"] += 1

    elif semantic_status == "error":
        stats["semantic_error"] += 1

    elif semantic_status != "not_attempted":
        raise ValueError(
            f"Unknown semantic status: {semantic_status!r}"
        )

    # -------------------------------------------------------------------------
    # Entity-bearing subset
    # -------------------------------------------------------------------------
    if has_entities:
        stats["entity_n"] += 1

        if semantic_status in {"correct", "wrong"}:
            stats["entity_valid"] += 1

            if semantic_status == "correct":
                stats["entity_correct"] += 1

            if constrained_status == "correct":
                stats["entity_constrained_valid"] += 1

        elif semantic_status in {"error", "not_attempted"}:
            # not_attempted here occurs when constrained inference itself failed.
            if constrained_status == "error" or semantic_status == "error":
                stats["entity_error"] += 1


def finalize_stats(stats: Counter) -> dict[str, Any]:
    n = int(stats["n"])

    loose_error = int(stats["loose_error"])
    loose_valid = n - loose_error

    constrained_error = int(stats["constrained_error"])
    constrained_valid = n - constrained_error

    constrained_correct = int(stats["constrained_correct"])
    constrained_wrong = int(stats["constrained_wrong"])
    constrained_unparsed = int(stats["constrained_unparsed"])

    parsed = constrained_correct + constrained_wrong

    semantic_valid = int(stats["semantic_valid"])
    semantic_correct = int(stats["semantic_correct"])

    entity_n = int(stats["entity_n"])
    entity_valid = int(stats["entity_valid"])
    entity_correct = int(stats["entity_correct"])

    return {
        "n": n,
        "any_error": int(stats["any_error"]),

        "loose": {
            "correct": int(stats["loose_correct"]),
            "valid_n": loose_valid,
            "error": loose_error,
            "accuracy": safe_ratio(
                stats["loose_correct"],
                loose_valid,
            ),
            "error_rate": safe_ratio(loose_error, n),
        },

        "constrained": {
            "correct": constrained_correct,
            "wrong": constrained_wrong,
            "unparsed": constrained_unparsed,
            "error": constrained_error,
            "valid_n": constrained_valid,
            "parsed": parsed,

            "accuracy": safe_ratio(
                constrained_correct,
                constrained_valid,
            ),
            "wrong_rate": safe_ratio(
                constrained_wrong,
                constrained_valid,
            ),
            "unparsed_rate": safe_ratio(
                constrained_unparsed,
                constrained_valid,
            ),
            "parsed_rate": safe_ratio(
                parsed,
                constrained_valid,
            ),
            "error_rate": safe_ratio(
                constrained_error,
                n,
            ),
        },

        "semantic": {
            "correct": semantic_correct,
            "valid_n": semantic_valid,
            "error": int(stats["semantic_error"]),

            "accuracy": safe_ratio(
                semantic_correct,
                semantic_valid,
            ),

            "given_constrained": safe_ratio(
                semantic_correct,
                stats["semantic_constrained_valid"],
            ),
        },

        "entity_semantic": {
            "n": entity_n,
            "valid_n": entity_valid,
            "correct": entity_correct,
            "error": int(stats["entity_error"]),

            "accuracy": safe_ratio(
                entity_correct,
                entity_valid,
            ),

            "given_constrained": safe_ratio(
                entity_correct,
                stats["entity_constrained_valid"],
            ),
        },
    }


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_dataset(
    *,
    dataset_name: str,
    rows: list[dict[str, Any]],
    concepts: tuple[str, ...],
    model_kind: str,
    tokenizer,
    model,
    image_processor,
    prediction_file: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    overall_stats = empty_stats()

    per_concept: dict[str, Counter] = {
        concept: empty_stats()
        for concept in concepts
    }

    total_stage_errors = 0

    prediction_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    progress = tqdm(
        rows,
        desc=f"{model_kind} / {dataset_name}",
        unit="sample",
    )

    with prediction_file.open(
        "w",
        encoding="utf-8",
    ) as prediction_log:

        for index, row in enumerate(progress):
            gold_concept = str(row["concept"])
            sample_id = stable_sample_key(row, index)

            target = row["_semantic_target"]
            gold_entities = target["entities"]
            has_entities = bool(gold_entities)

            loose_status = "wrong"
            constrained_status = "unparsed"
            semantic_status = "not_attempted"

            loose_response = None
            loose_prediction = None

            constrained_response = None
            constrained_prediction = None

            semantic_response = None
            semantic_prediction = None

            errors: dict[str, Any] = {}

            # -----------------------------------------------------------------
            # Loose
            # -----------------------------------------------------------------
            loose_prompt = original_loose_prompt(
                row,
                dataset_name,
            )

            try:
                loose_response = generate_response(
                    prompt_text=loose_prompt,
                    tokenizer=tokenizer,
                    model=model,
                    image_processor=image_processor,
                    conv_mode=args.conv_mode,
                    max_new_tokens=args.max_new_tokens_loose,
                    image_value=row["image"],
                    image_mode="normal",
                )

                loose_prediction = predict_loose_concept(
                    loose_response,
                    concepts,
                )

                loose_status = (
                    "correct"
                    if loose_prediction == gold_concept
                    else "wrong"
                )

            except Exception as exc:
                if is_cuda_oom(exc):
                    raise

                loose_status = "error"
                errors["loose"] = exception_record(exc)
                total_stage_errors += 1

            # -----------------------------------------------------------------
            # Constrained
            #
            # This still runs even if Loose generation failed.
            # -----------------------------------------------------------------
            (
                constrained_prompt,
                candidate_order,
            ) = build_constrained_prompt(
                dataset_name,
                concepts,
                sample_id,
                args.constrained_seed,
            )

            try:
                constrained_response = generate_response(
                    prompt_text=constrained_prompt,
                    tokenizer=tokenizer,
                    model=model,
                    image_processor=image_processor,
                    conv_mode=args.conv_mode,
                    max_new_tokens=args.max_new_tokens_constrained,
                    image_value=row["image"],
                    image_mode="normal",
                )

                constrained_prediction = parse_constrained_prediction(
                    constrained_response,
                    concepts,
                )

                if constrained_prediction is None:
                    constrained_status = "unparsed"
                elif constrained_prediction == gold_concept:
                    constrained_status = "correct"
                else:
                    constrained_status = "wrong"

            except Exception as exc:
                if is_cuda_oom(exc):
                    raise

                constrained_status = "error"
                errors["constrained"] = exception_record(exc)
                total_stage_errors += 1

            # -----------------------------------------------------------------
            # Semantic Exact
            # -----------------------------------------------------------------
            semantic_prompt = None

            if constrained_status == "correct":
                if not has_entities:
                    semantic_status = "correct"

                else:
                    semantic_prompt = build_semantic_prompt(
                        gold_concept,
                        gold_entities,
                    )

                    try:
                        semantic_response = generate_response(
                            prompt_text=semantic_prompt,
                            tokenizer=tokenizer,
                            model=model,
                            image_processor=image_processor,
                            conv_mode=args.conv_mode,
                            max_new_tokens=args.max_new_tokens_semantic,
                            image_value=row["image"],
                            image_mode="normal",
                        )

                        (
                            semantic_correct,
                            semantic_prediction,
                        ) = semantic_exact_from_response(
                            semantic_response,
                            gold_entities,
                        )

                        semantic_status = (
                            "correct"
                            if semantic_correct
                            else "wrong"
                        )

                    except Exception as exc:
                        if is_cuda_oom(exc):
                            raise

                        semantic_status = "error"
                        errors["semantic"] = exception_record(exc)
                        total_stage_errors += 1

            elif constrained_status in {"wrong", "unparsed"}:
                # Semantic is a strict extension of constrained correctness.
                semantic_status = "wrong"

            else:
                # Constrained inference itself failed.
                semantic_status = "not_attempted"

            if total_stage_errors > args.max_sample_errors:
                raise RuntimeError(
                    "Aborting evaluation: more than "
                    f"{args.max_sample_errors} inference/scoring errors occurred "
                    f"during {model_kind}/{dataset_name}."
                )

            # -----------------------------------------------------------------
            # Aggregate
            # -----------------------------------------------------------------
            update_stats(
                overall_stats,
                loose_status=loose_status,
                constrained_status=constrained_status,
                semantic_status=semantic_status,
                has_entities=has_entities,
            )

            update_stats(
                per_concept[gold_concept],
                loose_status=loose_status,
                constrained_status=constrained_status,
                semantic_status=semantic_status,
                has_entities=has_entities,
            )

            # -----------------------------------------------------------------
            # Raw prediction log
            # -----------------------------------------------------------------
            record = {
                "dataset": dataset_name,
                "model": model_kind,
                "sample_id": sample_id,
                "figure_id": row.get("figure_id"),
                "gold_concept": gold_concept,
                "gold_entities": gold_entities,
                "has_entities": has_entities,

                "loose": {
                    "prompt": loose_prompt,
                    "response": loose_response,
                    "prediction": loose_prediction,
                    "status": loose_status,
                },

                "constrained": {
                    "prompt": constrained_prompt,
                    "candidate_order": candidate_order,
                    "response": constrained_response,
                    "prediction": constrained_prediction,
                    "status": constrained_status,
                },

                "semantic": {
                    "prompt": semantic_prompt,
                    "response": semantic_response,
                    "prediction": semantic_prediction,
                    "status": semantic_status,
                },

                "errors": errors,
            }

            write_jsonl_record(
                prediction_log,
                record,
            )

            current = finalize_stats(overall_stats)

            progress.set_postfix(
                loose=(
                    f"{100 * (current['loose']['accuracy'] or 0):.1f}%"
                ),
                constrained=(
                    f"{100 * (current['constrained']['accuracy'] or 0):.1f}%"
                ),
                semantic=(
                    f"{100 * (current['semantic']['accuracy'] or 0):.1f}%"
                ),
                errors=current["any_error"],
            )

    return {
        "n": len(rows),
        "stage_errors": total_stage_errors,
        "prediction_file": str(prediction_file),
        "overall": finalize_stats(overall_stats),
        "per_concept": {
            concept: finalize_stats(per_concept[concept])
            for concept in concepts
        },
    }


# =============================================================================
# Markdown formatting
# =============================================================================

def pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "-"

    return f"{value * 100:.2f}%"


def classification_table(
    *,
    title: str,
    dataset_results: dict[str, Any],
    model_order: tuple[str, ...],
) -> list[str]:
    lines = [
        f"## {title}",
        "",
        (
            "| Model | Loose Acc | Loose Error | "
            "Constrained Acc | Wrong Concept | Unparsed | Parsed Rate | "
            "Constrained Error | Valid N | N |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for model_name in model_order:
        result = dataset_results.get(model_name)

        if result is None:
            lines.append(
                f"| {model_name} | - | - | - | - | - | - | - | - | - |"
            )
            continue

        overall = result["overall"]
        loose = overall["loose"]
        constrained = overall["constrained"]

        lines.append(
            f"| {model_name} "
            f"| {pct(loose['accuracy'])} "
            f"| {pct(loose['error_rate'])} "
            f"| {pct(constrained['accuracy'])} "
            f"| {pct(constrained['wrong_rate'])} "
            f"| {pct(constrained['unparsed_rate'])} "
            f"| {pct(constrained['parsed_rate'])} "
            f"| {pct(constrained['error_rate'])} "
            f"| {constrained['valid_n']:,} "
            f"| {overall['n']:,} |"
        )

    lines.append("")
    return lines


def semantic_table(
    *,
    title: str,
    dataset_results: dict[str, Any],
    model_order: tuple[str, ...],
) -> list[str]:
    lines = [
        f"## {title}",
        "",
        (
            "| Model | Semantic Acc | Semantic \\| Constrained | "
            "Semantic Error N | Semantic Valid N | "
            "Entity Semantic Acc | Entity \\| Constrained | "
            "Entity Error N | Entity Valid N | Entity N |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for model_name in model_order:
        result = dataset_results.get(model_name)

        if result is None:
            lines.append(
                f"| {model_name} | - | - | - | - | - | - | - | - | - |"
            )
            continue

        semantic = result["overall"]["semantic"]
        entity = result["overall"]["entity_semantic"]

        lines.append(
            f"| {model_name} "
            f"| {pct(semantic['accuracy'])} "
            f"| {pct(semantic['given_constrained'])} "
            f"| {semantic['error']:,} "
            f"| {semantic['valid_n']:,} "
            f"| {pct(entity['accuracy'])} "
            f"| {pct(entity['given_constrained'])} "
            f"| {entity['error']:,} "
            f"| {entity['valid_n']:,} "
            f"| {entity['n']:,} |"
        )

    lines.append("")
    return lines


def per_concept_tables(
    *,
    title: str,
    dataset_results: dict[str, Any],
    concepts: tuple[str, ...],
    model_order: tuple[str, ...],
) -> list[str]:
    lines = [f"## {title}", ""]

    for model_name in model_order:
        result = dataset_results.get(model_name)

        if result is None:
            continue

        lines.extend(
            [
                f"### {model_name}",
                "",
                (
                    "| Concept | Loose | Constrained | Wrong | Unparsed | Parsed | "
                    "Con. Error | Semantic | Sem. \\| Con. | "
                    "Entity Sem. | Entity \\| Con. | Error Samples | N |"
                ),
                (
                    "|---|---:|---:|---:|---:|---:|---:|"
                    "---:|---:|---:|---:|---:|---:|"
                ),
            ]
        )

        for concept in concepts:
            item = result["per_concept"][concept]

            constrained = item["constrained"]
            semantic = item["semantic"]
            entity = item["entity_semantic"]

            lines.append(
                f"| {concept} "
                f"| {pct(item['loose']['accuracy'])} "
                f"| {pct(constrained['accuracy'])} "
                f"| {pct(constrained['wrong_rate'])} "
                f"| {pct(constrained['unparsed_rate'])} "
                f"| {pct(constrained['parsed_rate'])} "
                f"| {pct(constrained['error_rate'])} "
                f"| {pct(semantic['accuracy'])} "
                f"| {pct(semantic['given_constrained'])} "
                f"| {pct(entity['accuracy'])} "
                f"| {pct(entity['given_constrained'])} "
                f"| {item['any_error']:,} "
                f"| {item['n']:,} |"
            )

        lines.append("")

    return lines


def prediction_file_table(
    results: dict[str, dict[str, Any]],
) -> list[str]:
    lines = [
        "## Prediction Logs",
        "",
        "| Dataset | Model | JSONL |",
        "|---|---|---|",
    ]

    for dataset_name in ("stage1", "stage2"):
        for model_name in MODEL_ORDER:
            result = results[dataset_name].get(model_name)

            if result is None:
                continue

            lines.append(
                f"| {dataset_name} "
                f"| {model_name} "
                f"| `{result['prediction_file']}` |"
            )

    lines.append("")
    return lines


def write_markdown(
    *,
    results: dict[str, dict[str, Any]],
    output_file: Path,
    complete: bool,
) -> None:
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines: list[str] = [
        "# Geometry Accuracy Results",
        "",
        f"Status: **{'complete' if complete else 'partial'}**",
        "",
        "## Metric Definitions",
        "",
        (
            "- **Loose Acc**: permissive free-form concept scoring using "
            "predefined aliases."
        ),
        (
            "- **Constrained Acc**: canonical concept classification from the "
            "closed candidate set."
        ),
        (
            "- **Wrong Concept**: exactly one canonical concept was parsed, "
            "but it was not the gold concept."
        ),
        (
            "- **Unparsed**: constrained inference succeeded, but no unique "
            "canonical concept could be recovered."
        ),
        (
            "- **Parsed Rate**: constrained correct + wrong concept over valid "
            "constrained outputs. This is parse success, not strict format compliance."
        ),
        (
            "- **Error**: inference/scoring exception. Errors are reported "
            "separately and are not counted as Unparsed."
        ),
        (
            "- **Semantic Acc**: constrained concept correct AND visible entity "
            "JSON exactly correct."
        ),
        (
            "- **Semantic | Constrained**: semantic correctness among valid "
            "samples whose constrained concept was correct."
        ),
        (
            "- **Entity Semantic Acc**: Semantic Acc restricted to samples with "
            "non-empty entity targets."
        ),
        (
            "- **Entity | Constrained**: entity exactness among valid "
            "entity-bearing samples with a correct constrained concept."
        ),
        "",
        "Uniform random-choice references for Constrained Accuracy:",
        "",
        (
            f"- Stage1: 1 / {len(STAGE1_CONCEPTS)} "
            f"= {100 / len(STAGE1_CONCEPTS):.2f}%"
        ),
        (
            f"- Stage2: 1 / {len(STAGE2_CONCEPTS)} "
            f"= {100 / len(STAGE2_CONCEPTS):.2f}%"
        ),
        "",
        (
            "Loose and Semantic metrics do not have a comparable simple "
            "uniform random-choice reference."
        ),
        "",
    ]

    # Stage1
    lines.extend(
        classification_table(
            title="Stage 1 Dataset — Concept Classification",
            dataset_results=results["stage1"],
            model_order=STAGE1_MODEL_ORDER,
        )
    )

    lines.extend(
        semantic_table(
            title="Stage 1 Dataset — Semantic Grounding",
            dataset_results=results["stage1"],
            model_order=STAGE1_MODEL_ORDER,
        )
    )

    # Stage2
    lines.extend(
        classification_table(
            title="Stage 2 Dataset — Concept Classification",
            dataset_results=results["stage2"],
            model_order=STAGE2_MODEL_ORDER,
        )
    )

    lines.extend(
        semantic_table(
            title="Stage 2 Dataset — Semantic Grounding",
            dataset_results=results["stage2"],
            model_order=STAGE2_MODEL_ORDER,
        )
    )

    # Per-concept
    lines.extend(
        per_concept_tables(
            title="Stage 1 Dataset — Per-Concept Results",
            dataset_results=results["stage1"],
            concepts=STAGE1_CONCEPTS,
            model_order=STAGE1_MODEL_ORDER,
        )
    )

    lines.extend(
        per_concept_tables(
            title="Stage 2 Dataset — Per-Concept Results",
            dataset_results=results["stage2"],
            concepts=STAGE2_CONCEPTS,
            model_order=STAGE2_MODEL_ORDER,
        )
    )

    lines.extend(prediction_file_table(results))

    output_file.write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )

    print(f"Updated result file: {output_file}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    output_file = args.output_dir / args.output_file

    prediction_dir = (
        args.prediction_dir
        if args.prediction_dir is not None
        else args.output_dir / "accuracy_predictions"
    )

    model_paths = EvaluationModelPaths.from_values(
        base_model=args.base_model,
        stage1_dir=args.stage1_dir,
        stage2_dir=args.stage2_dir,
        stage2_only_dir=args.stage2_only_dir,
    )

    print(f"Accuracy result : {output_file}")
    print(f"Prediction logs : {prediction_dir}")

    # Validate all datasets before loading a 7B model.
    stage1_rows = prepare_stage1_rows(args)
    stage2_rows = prepare_stage2_rows(args)

    results: dict[str, dict[str, Any]] = {
        "stage1": {},
        "stage2": {},
    }

    write_markdown(
        results=results,
        output_file=output_file,
        complete=False,
    )

    # One model at a time.
    for model_kind in MODEL_ORDER:
        tokenizer = None
        model = None
        image_processor = None

        try:
            (
                tokenizer,
                model,
                image_processor,
                _,
            ) = load_eval_model(
                model_kind,
                model_paths,
            )

            # Stage1 — all four models, including Stage2-only.
            results["stage1"][model_kind] = evaluate_dataset(
                dataset_name="stage1",
                rows=stage1_rows,
                concepts=STAGE1_CONCEPTS,
                model_kind=model_kind,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                prediction_file=(
                    prediction_dir
                    / f"{model_kind}_stage1.jsonl"
                ),
                args=args,
            )

            write_markdown(
                results=results,
                output_file=output_file,
                complete=False,
            )

            # Stage2 — all four models.
            results["stage2"][model_kind] = evaluate_dataset(
                dataset_name="stage2",
                rows=stage2_rows,
                concepts=STAGE2_CONCEPTS,
                model_kind=model_kind,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                prediction_file=(
                    prediction_dir
                    / f"{model_kind}_stage2.jsonl"
                ),
                args=args,
            )

            write_markdown(
                results=results,
                output_file=output_file,
                complete=False,
            )

        finally:
            if model is not None:
                unload_model(
                    tokenizer,
                    model,
                    image_processor,
                )

    write_markdown(
        results=results,
        output_file=output_file,
        complete=True,
    )

    print("\nAccuracy evaluation completed.")
    print(f"Final result: {output_file}")


if __name__ == "__main__":
    main()