#!/usr/bin/env python3
from __future__ import annotations

"""
Visual-dependency evaluation for Stage2 geometry concepts.

Models
Default:
- stage2
- stage2_only

Conditions
normal:
    Correct image.

shuffled:
    Image is replaced by another evaluation image from a DIFFERENT concept.
    The replacement is deterministic and one-to-one across the dataset.

blank:
    White image with the original image size.

none:
    No image token / no image input.

Shared infrastructure:
    helper.py

Shared ontology:
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
from typing import Any, Sequence

from tqdm import tqdm

from helper import (
    EvaluationModelPaths,
    filter_task_rows,
    generate_response,
    is_cuda_oom,
    load_eval_model,
    read_parquet_files,
    stable_sample_key,
    stratified_limit,
    unload_model,
    validate_concepts,
    validate_required_columns,
    warn_expected_count,
)

from ontology import STAGE2_CONCEPTS


# =============================================================================
# Constants
# =============================================================================

DEFAULT_MODELS = (
    "stage2",
    "stage2_only",
)

SUPPORTED_MODELS = (
    "base",
    "stage1",
    "stage2",
    "stage2_only",
)

IMAGE_MODES = (
    "normal",
    "shuffled",
    "blank",
    "none",
)

DEFAULT_STAGE2_PATTERN = (
    "stage2_geometry_evaluate_evaluate_*.parquet"
)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Stage2 concept classification under "
            "normal/shuffled/blank/no-image conditions."
        )
    )

    # Data
    parser.add_argument(
        "--stage2-data-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stage2-pattern",
        default=DEFAULT_STAGE2_PATTERN,
    )

    # Models
    parser.add_argument(
        "--base-model",
        required=True,
    )
    parser.add_argument(
        "--stage1-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stage2-only-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=list(DEFAULT_MODELS),
        help=(
            "Models to evaluate. Default: stage2 stage2_only."
        ),
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-file",
        default="visual_dependency_results.md",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=None,
        help=(
            "Defaults to "
            "<output-dir>/visual_dependency_predictions."
        ),
    )

    # Runtime
    parser.add_argument(
        "--conv-mode",
        default="llava_v1",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--candidate-seed",
        type=int,
        default=20260822,
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=20260822,
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=20260822,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help=(
            "0 evaluates the full dataset. "
            "Positive values use concept-balanced sampling."
        ),
    )
    parser.add_argument(
        "--max-sample-errors",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--expected-stage2-local-rows",
        type=int,
        default=2700,
    )

    args = parser.parse_args()

    # Preserve order while removing duplicates.
    args.models = tuple(
        dict.fromkeys(args.models)
    )

    return args


# =============================================================================
# Dataset
# =============================================================================

def discover_parquets(
    directory: Path,
    pattern: str,
) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Stage2 data directory was not found: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Stage2 data path is not a directory: {directory}"
        )

    paths = sorted(
        path
        for path in directory.glob(pattern)
        if path.is_file()
    )

    if not paths:
        raise FileNotFoundError(
            "No Stage2 evaluation parquet matched:\n"
            f"  directory: {directory}\n"
            f"  pattern  : {pattern}"
        )

    print(
        f"Stage2: discovered {len(paths):,} shard(s)."
    )

    for path in paths:
        print(f"  - {path}")

    return paths


def sort_rows(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    indexed = list(enumerate(rows))

    indexed.sort(
        key=lambda item: (
            str(item[1].get("concept") or ""),
            stable_sample_key(item[1], item[0]),
        )
    )

    return [
        row
        for _, row in indexed
    ]


def validate_unique_figures(
    rows: Sequence[dict[str, Any]],
) -> None:
    figure_ids = [
        str(row.get("figure_id") or "")
        for row in rows
    ]

    if not figure_ids or not all(figure_ids):
        return

    duplicate_count = (
        len(figure_ids)
        - len(set(figure_ids))
    )

    if duplicate_count:
        raise ValueError(
            "Stage2 local evaluation set contains "
            f"{duplicate_count} duplicate figure_id value(s)."
        )


def prepare_stage2_rows(
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    paths = discover_parquets(
        args.stage2_data_dir,
        args.stage2_pattern,
    )

    rows = read_parquet_files(
        paths,
        "Stage2",
    )

    validate_required_columns(
        rows,
        {
            "concept",
            "image",
            "task_kind",
        },
        "Stage2",
    )

    rows = filter_task_rows(
        rows,
        "local",
    )

    if not rows:
        raise RuntimeError(
            "Stage2 contains no task_kind='local' rows."
        )

    validate_concepts(
        rows,
        STAGE2_CONCEPTS,
        "Stage2 local",
    )

    validate_unique_figures(rows)

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

    if len(rows) < 2:
        raise ValueError(
            "Visual-dependency evaluation requires at least two samples."
        )

    print(
        f"Stage2 visual-dependency rows: {len(rows):,}"
    )

    return rows


# =============================================================================
# Deterministic shuffled-image mapping
# =============================================================================

def build_shuffled_indices(
    rows: Sequence[dict[str, Any]],
) -> list[int]:
    """
    Construct a deterministic one-to-one image permutation in which every
    sample receives an image belonging to a different Stage2 concept.

    Rows are already sorted by concept, so each concept occupies one contiguous
    block. Rotating by the size of the largest block guarantees that no block
    overlaps itself as long as the largest concept occupies <= half the set.

    For the intended balanced Stage2 set:
        27 concepts x 100 figures
    this becomes a simple 100-position rotation.
    """

    n = len(rows)

    counts = Counter(
        str(row["concept"])
        for row in rows
    )

    max_count = max(
        counts.values()
    )

    if max_count * 2 > n:
        raise ValueError(
            "Cannot construct a different-concept one-to-one shuffle: "
            "one concept contains more than half of the evaluation set."
        )

    indices = [
        (index + max_count) % n
        for index in range(n)
    ]

    # Defensive verification.
    for source_index, target_index in enumerate(indices):
        source_concept = str(
            rows[source_index]["concept"]
        )

        target_concept = str(
            rows[target_index]["concept"]
        )

        if source_index == target_index:
            raise RuntimeError(
                "Shuffled-image mapping contains an identity mapping."
            )

        if source_concept == target_concept:
            raise RuntimeError(
                "Shuffled-image mapping failed to change concept: "
                f"row {source_index}, concept={source_concept!r}."
            )

    if len(set(indices)) != n:
        raise RuntimeError(
            "Shuffled-image mapping is not one-to-one."
        )

    return indices


# =============================================================================
# Constrained concept prompt / parser
# =============================================================================

def normalize_text(text: str) -> str:
    text = (
        str(text)
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    text = re.sub(
        r"[^a-z0-9 ]+",
        " ",
        text,
    )

    return " ".join(
        text.split()
    )


def shuffled_candidates(
    sample_id: str,
    seed: int,
) -> list[str]:
    """
    Candidate order is fixed per evaluation sample and therefore identical
    across image modes and checkpoints.
    """

    digest = hashlib.sha256(
        f"{seed}:{sample_id}".encode("utf-8")
    ).digest()

    sample_seed = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    rng = random.Random(
        sample_seed
    )

    candidates = list(
        STAGE2_CONCEPTS
    )

    rng.shuffle(
        candidates
    )

    return candidates


def build_constrained_prompt(
    sample_id: str,
    seed: int,
) -> tuple[str, list[str]]:
    candidates = shuffled_candidates(
        sample_id,
        seed,
    )

    options = "\n".join(
        f"- {label}"
        for label in candidates
    )

    prompt = (
        "Classify the geometric relation or property shown in the image.\n"
        "Choose exactly ONE label from the candidate list below.\n"
        "Return the canonical label only. Do not explain.\n\n"
        f"Candidates:\n{options}"
    )

    return prompt, candidates


def parse_constrained_prediction(
    text: str,
) -> str | None:
    """
    Recover exactly one canonical Stage2 concept.

    `unparsed` means inference succeeded but no unique concept could be
    recovered. It is distinct from an inference error.
    """

    normalized = normalize_text(text)

    found: list[str] = []

    for concept in STAGE2_CONCEPTS:
        label = normalize_text(
            concept
        )

        pattern = (
            rf"(?<![a-z0-9])"
            rf"{re.escape(label)}"
            rf"(?![a-z0-9])"
        )

        if re.search(
            pattern,
            normalized,
        ):
            found.append(
                concept
            )

    return (
        found[0]
        if len(found) == 1
        else None
    )


# =============================================================================
# Statistics
# =============================================================================

def empty_stats() -> Counter:
    return Counter(
        n=0,
        correct=0,
        wrong=0,
        unparsed=0,
        error=0,
    )


def update_stats(
    stats: Counter,
    status: str,
) -> None:
    stats["n"] += 1

    if status not in {
        "correct",
        "wrong",
        "unparsed",
        "error",
    }:
        raise ValueError(
            f"Unknown prediction status: {status!r}"
        )

    stats[status] += 1


def safe_ratio(
    numerator: int,
    denominator: int,
) -> float | None:
    if denominator == 0:
        return None

    return numerator / denominator


def finalize_stats(
    stats: Counter,
) -> dict[str, Any]:
    n = int(
        stats["n"]
    )

    error = int(
        stats["error"]
    )

    valid_n = (
        n - error
    )

    correct = int(
        stats["correct"]
    )

    wrong = int(
        stats["wrong"]
    )

    unparsed = int(
        stats["unparsed"]
    )

    parsed = (
        correct + wrong
    )

    return {
        "n": n,
        "valid_n": valid_n,

        "correct": correct,
        "wrong": wrong,
        "unparsed": unparsed,
        "error": error,
        "parsed": parsed,

        "accuracy": safe_ratio(
            correct,
            valid_n,
        ),

        "wrong_rate": safe_ratio(
            wrong,
            valid_n,
        ),

        "unparsed_rate": safe_ratio(
            unparsed,
            valid_n,
        ),

        "parsed_rate": safe_ratio(
            parsed,
            valid_n,
        ),

        "error_rate": safe_ratio(
            error,
            n,
        ),
    }


# =============================================================================
# JSONL
# =============================================================================

def exception_record(
    exc: BaseException,
) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


def write_jsonl_record(
    handle,
    record: dict[str, Any],
) -> None:
    handle.write(
        json.dumps(
            record,
            ensure_ascii=False,
        )
        + "\n"
    )

    handle.flush()


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(
    *,
    rows: list[dict[str, Any]],
    shuffled_indices: Sequence[int],
    model_kind: str,
    tokenizer,
    model,
    image_processor,
    prediction_file: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    overall = {
        mode: empty_stats()
        for mode in IMAGE_MODES
    }

    per_concept = {
        mode: {
            concept: empty_stats()
            for concept in STAGE2_CONCEPTS
        }
        for mode in IMAGE_MODES
    }

    total_errors = 0

    prediction_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_for_progress = (
        len(rows)
        * len(IMAGE_MODES)
    )

    progress = tqdm(
        total=total_for_progress,
        desc=f"{model_kind} / visual dependency",
        unit="generation",
    )

    with prediction_file.open(
        "w",
        encoding="utf-8",
    ) as prediction_log:

        for index, row in enumerate(rows):
            gold_concept = str(
                row["concept"]
            )

            sample_id = stable_sample_key(
                row,
                index,
            )

            prompt, candidate_order = (
                build_constrained_prompt(
                    sample_id,
                    args.candidate_seed,
                )
            )

            shuffled_index = int(
                shuffled_indices[index]
            )

            shuffled_row = rows[
                shuffled_index
            ]

            shuffled_sample_id = stable_sample_key(
                shuffled_row,
                shuffled_index,
            )

            shuffled_concept = str(
                shuffled_row[
                    "concept"
                ]
            )

            for mode in IMAGE_MODES:
                response = None
                prediction = None
                error = None
                status = "unparsed"

                # -------------------------------------------------------------
                # Configure visual input
                # -------------------------------------------------------------
                if mode == "normal":
                    image_value = row["image"]
                    helper_mode = "normal"
                    image_override = None

                elif mode == "shuffled":
                    # helper.py uses image_override to provide another image
                    # while retaining normal image preprocessing.
                    image_value = row["image"]
                    helper_mode = "normal"
                    image_override = shuffled_row["image"]

                elif mode == "blank":
                    image_value = row["image"]
                    helper_mode = "blank"
                    image_override = None

                elif mode == "none":
                    image_value = None
                    helper_mode = "none"
                    image_override = None

                else:
                    raise RuntimeError(
                        f"Unhandled image mode: {mode}"
                    )

                # -------------------------------------------------------------
                # Inference
                # -------------------------------------------------------------
                try:
                    response = generate_response(
                        prompt_text=prompt,
                        tokenizer=tokenizer,
                        model=model,
                        image_processor=image_processor,
                        conv_mode=args.conv_mode,
                        max_new_tokens=args.max_new_tokens,
                        image_value=image_value,
                        image_mode=helper_mode,
                        image_override=image_override,
                    )

                    prediction = (
                        parse_constrained_prediction(
                            response
                        )
                    )

                    if prediction is None:
                        status = "unparsed"

                    elif prediction == gold_concept:
                        status = "correct"

                    else:
                        status = "wrong"

                except Exception as exc:
                    if is_cuda_oom(exc):
                        raise

                    status = "error"
                    error = exception_record(
                        exc
                    )

                    total_errors += 1

                    print(
                        "\nWARNING: visual-dependency error "
                        f"[{model_kind}/{mode}/{sample_id}]: "
                        f"{type(exc).__name__}: {exc}"
                    )

                update_stats(
                    overall[mode],
                    status,
                )

                update_stats(
                    per_concept[
                        mode
                    ][gold_concept],
                    status,
                )

                # -------------------------------------------------------------
                # Per-sample log
                # -------------------------------------------------------------
                record = {
                    "dataset": "stage2",
                    "task_kind": "local",
                    "model": model_kind,
                    "image_mode": mode,

                    "sample_id": sample_id,
                    "figure_id": row.get("figure_id"),
                    "gold_concept": gold_concept,

                    "prompt": prompt,
                    "candidate_order": candidate_order,

                    "response": response,
                    "prediction": prediction,
                    "status": status,

                    "source_image": {
                        "sample_id": sample_id,
                        "concept": gold_concept,
                    },

                    "actual_image_input": (
                        None
                        if mode == "none"
                        else {
                            "sample_id": (
                                shuffled_sample_id
                                if mode == "shuffled"
                                else sample_id
                            ),
                            "concept": (
                                shuffled_concept
                                if mode == "shuffled"
                                else gold_concept
                            ),
                            "blank": (
                                mode == "blank"
                            ),
                        }
                    ),

                    "error": error,
                }

                write_jsonl_record(
                    prediction_log,
                    record,
                )

                if total_errors > args.max_sample_errors:
                    raise RuntimeError(
                        "Aborting visual-dependency evaluation: "
                        f"more than {args.max_sample_errors} errors "
                        f"occurred for {model_kind}."
                    )

                progress.update(1)

                current = finalize_stats(
                    overall[mode]
                )

                progress.set_postfix(
                    mode=mode,
                    acc=(
                        f"{100 * (current['accuracy'] or 0):.1f}%"
                    ),
                    errors=total_errors,
                )

    progress.close()

    return {
        "n": len(rows),
        "prediction_file": str(
            prediction_file
        ),

        "overall": {
            mode: finalize_stats(
                overall[mode]
            )
            for mode in IMAGE_MODES
        },

        "per_concept": {
            mode: {
                concept: finalize_stats(
                    per_concept[
                        mode
                    ][concept]
                )
                for concept in STAGE2_CONCEPTS
            }
            for mode in IMAGE_MODES
        },
    }


# =============================================================================
# Markdown
# =============================================================================

def pct(
    value: float | None,
) -> str:
    if value is None:
        return "-"

    if not math.isfinite(
        value
    ):
        return "-"

    return (
        f"{value * 100:.2f}%"
    )


def overall_table(
    results: dict[str, Any],
    models: Sequence[str],
) -> list[str]:
    lines = [
        "## Overall Results",
        "",
        (
            "| Model | Image Mode | Accuracy | Wrong | "
            "Unparsed | Parsed Rate | Error Rate | Valid N | N |"
        ),
        (
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for model_kind in models:
        result = results.get(
            model_kind
        )

        if result is None:
            for mode in IMAGE_MODES:
                lines.append(
                    f"| {model_kind} | {mode} "
                    "| - | - | - | - | - | - | - |"
                )

            continue

        for mode in IMAGE_MODES:
            item = result[
                "overall"
            ][mode]

            lines.append(
                f"| {model_kind} "
                f"| {mode} "
                f"| {pct(item['accuracy'])} "
                f"| {pct(item['wrong_rate'])} "
                f"| {pct(item['unparsed_rate'])} "
                f"| {pct(item['parsed_rate'])} "
                f"| {pct(item['error_rate'])} "
                f"| {item['valid_n']:,} "
                f"| {item['n']:,} |"
            )

    lines.append("")
    return lines


def per_concept_tables(
    results: dict[str, Any],
    models: Sequence[str],
) -> list[str]:
    lines = [
        "## Per-Concept Accuracy",
        "",
    ]

    for model_kind in models:
        result = results.get(
            model_kind
        )

        if result is None:
            continue

        lines.extend(
            [
                f"### {model_kind}",
                "",
                (
                    "| Concept | Normal | Shuffled | Blank | None | "
                    "Normal Errors | Shuffled Errors | "
                    "Blank Errors | None Errors |"
                ),
                (
                    "|---|---:|---:|---:|---:|"
                    "---:|---:|---:|---:|"
                ),
            ]
        )

        for concept in STAGE2_CONCEPTS:
            normal = result[
                "per_concept"
            ][
                "normal"
            ][concept]

            shuffled = result[
                "per_concept"
            ][
                "shuffled"
            ][concept]

            blank = result[
                "per_concept"
            ][
                "blank"
            ][concept]

            none = result[
                "per_concept"
            ][
                "none"
            ][concept]

            lines.append(
                f"| {concept} "
                f"| {pct(normal['accuracy'])} "
                f"| {pct(shuffled['accuracy'])} "
                f"| {pct(blank['accuracy'])} "
                f"| {pct(none['accuracy'])} "
                f"| {normal['error']:,} "
                f"| {shuffled['error']:,} "
                f"| {blank['error']:,} "
                f"| {none['error']:,} |"
            )

        lines.append("")

    return lines


def prediction_table(
    results: dict[str, Any],
    models: Sequence[str],
) -> list[str]:
    lines = [
        "## Prediction Logs",
        "",
        "| Model | JSONL |",
        "|---|---|",
    ]

    for model_kind in models:
        result = results.get(
            model_kind
        )

        if result is None:
            continue

        lines.append(
            f"| {model_kind} "
            f"| `{result['prediction_file']}` |"
        )

    lines.append("")
    return lines


def write_markdown(
    *,
    results: dict[str, Any],
    output_file: Path,
    models: Sequence[str],
    sample_n: int,
    complete: bool,
) -> None:
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    chance = (
        1.0
        / len(STAGE2_CONCEPTS)
    )

    lines: list[str] = [
        "# Visual Dependency Results",
        "",
        f"Status: **{'complete' if complete else 'partial'}**",
        "",
        f"Stage2 local samples: **{sample_n:,}**",
        "",
        "## Conditions",
        "",
        "- **normal**: correct image.",
        (
            "- **shuffled**: one-to-one replacement with an image "
            "from a different Stage2 concept."
        ),
        "- **blank**: white image with the original dimensions.",
        "- **none**: text-only input with no image token/input.",
        "",
        "## Metric",
        "",
        (
            "- **Accuracy**: 27-way constrained Stage2 concept "
            "classification accuracy over valid model outputs."
        ),
        (
            "- **Wrong**: exactly one canonical Stage2 concept was parsed, "
            "but it was incorrect."
        ),
        (
            "- **Unparsed**: inference succeeded but no unique canonical "
            "concept could be recovered."
        ),
        (
            "- **Error Rate**: inference/scoring exception. Errors are "
            "reported separately and are not counted as Unparsed."
        ),
        "",
        (
            f"Uniform random-choice reference: "
            f"1 / {len(STAGE2_CONCEPTS)} = {chance * 100:.2f}%."
        ),
        "",
        (
            "`shuffled`, `blank`, and `none` are empirical ablation "
            "conditions, not theoretical lower bounds."
        ),
        "",
    ]

    lines.extend(
        overall_table(
            results,
            models,
        )
    )

    lines.extend(
        per_concept_tables(
            results,
            models,
        )
    )

    lines.extend(
        prediction_table(
            results,
            models,
        )
    )

    output_file.write_text(
        "\n".join(lines).rstrip()
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Updated result file: "
        f"{output_file}"
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    output_file = (
        args.output_dir
        / args.output_file
    )

    prediction_dir = (
        args.prediction_dir
        if args.prediction_dir is not None
        else args.output_dir
        / "visual_dependency_predictions"
    )

    model_paths = (
        EvaluationModelPaths.from_values(
            base_model=args.base_model,
            stage1_dir=args.stage1_dir,
            stage2_dir=args.stage2_dir,
            stage2_only_dir=args.stage2_only_dir,
        )
    )

    print(
        f"Visual dependency result: "
        f"{output_file}"
    )

    print(
        f"Prediction logs        : "
        f"{prediction_dir}"
    )

    print(
        "Models                 : "
        + ", ".join(args.models)
    )

    # Validate/load data before loading any model.
    rows = prepare_stage2_rows(
        args
    )

    shuffled_indices = (
        build_shuffled_indices(
            rows
        )
    )

    results: dict[
        str,
        Any,
    ] = {}

    write_markdown(
        results=results,
        output_file=output_file,
        models=args.models,
        sample_n=len(rows),
        complete=False,
    )

    # One model at a time.
    for model_kind in args.models:
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

            results[
                model_kind
            ] = evaluate_model(
                rows=rows,
                shuffled_indices=shuffled_indices,
                model_kind=model_kind,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                prediction_file=(
                    prediction_dir
                    / f"{model_kind}_stage2_visual_dependency.jsonl"
                ),
                args=args,
            )

            write_markdown(
                results=results,
                output_file=output_file,
                models=args.models,
                sample_n=len(rows),
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
        models=args.models,
        sample_n=len(rows),
        complete=True,
    )

    print(
        "\nVisual-dependency evaluation completed."
    )

    print(
        f"Final result: {output_file}"
    )


if __name__ == "__main__":
    main()
