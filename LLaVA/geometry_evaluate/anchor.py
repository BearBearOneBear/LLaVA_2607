#!/usr/bin/env python3
from __future__ import annotations

"""
Geometry anchor evaluator.

Metrics
1. Overall Fact F1

2. Concept-wise Fact F1

3. Field-wise Fact F1

4. Parse Fail Rate

5. Error Rate

Outputs
1. anchor_results.md

2. Per-model JSONL prediction logs containing:

Shared infrastructure:
    helper.py

Shared ontology:
    ontology.py
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

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
    ANCHOR_DSL_FIELDS,
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

DEFAULT_STAGE2_PATTERN = (
    "stage2_geometry_evaluate_evaluate_*.parquet"
)

DEFAULT_ANCHOR_PROMPT = (
    "Encode the geometric configuration.\n"
    "Geometry representation:"
)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage2 geometry anchor reconstruction."
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

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-file",
        default="anchor_results.md",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-sample JSONL logs. "
            "Defaults to <output-dir>/anchor_predictions."
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
        default=768,
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
            "0 evaluates the complete set. "
            "Positive values use deterministic concept-balanced sampling."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=20260822,
    )
    parser.add_argument(
        "--expected-anchor-rows",
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
) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Stage2 evaluation directory was not found: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Stage2 evaluation path is not a directory: {directory}"
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
        f"Stage2: discovered {len(paths):,} parquet shard(s)."
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

    return [row for _, row in indexed]


# =============================================================================
# Anchor fact representation
# =============================================================================

@dataclass
class ParsedAnchor:
    parsed: bool
    facts_by_field: dict[str, set[str]]
    all_facts: set[str]


def normalized_space(text: str) -> str:
    return " ".join(
        str(text or "").strip().split()
    )


def canonical_segment(token: str) -> str:
    """
    Treat an undirected segment AB and BA as equivalent.
    """

    token = re.sub(
        r"[^A-Z]",
        "",
        str(token).upper(),
    )

    if len(token) == 2:
        return "".join(
            sorted(token)
        )

    return token


def canonical_point(token: str) -> str:
    token = re.sub(
        r"[^A-Z]",
        "",
        str(token).upper(),
    )

    return token


def split_function_calls(
    payload: str,
) -> list[tuple[str, list[str]]]:
    """
    Parse simple DSL function calls without altering payload case globally.

    Function names are case-insensitive.
    Arguments are normalized later according to their semantic role.
    """

    calls: list[
        tuple[str, list[str]]
    ] = []

    pattern = re.compile(
        r"([A-Za-z][A-Za-z0-9_]*)\s*\(([^()]*)\)"
    )

    for match in pattern.finditer(payload):
        name = match.group(1).upper()

        args = [
            arg.strip()
            for arg in match.group(2).split(",")
            if arg.strip()
        ]

        calls.append(
            (name, args)
        )

    return calls


def canonical_segment_group(
    segments: Sequence[str],
) -> str:
    normalized = sorted(
        {
            canonical_segment(segment)
            for segment in segments
            if canonical_segment(segment)
        }
    )

    return "|".join(normalized)


# =============================================================================
# Anchor DSL parser
# =============================================================================

def parse_anchor(text: str) -> ParsedAnchor:
    """
    Parse model-generated anchor DSL into canonical fact sets.

    Important parser rules
    ----------------------
    1. Only ANCHOR_DSL_FIELDS may create facts.

       Example:
           NOTE: ...
           ANSWER: ...

       are ignored completely.

    2. Field names are case-insensitive.

       Example:
           points: A B

       is equivalent to:
           POINTS: A B

    3. POINTS payload case is preserved before label extraction.

       Therefore:
           POINTS: A B
       -> points A and B

       but:
           POINTS: a and b
       does NOT accidentally become uppercase point labels.

    4. Undirected segment endpoint order is ignored.

    5. EQ/PARA marker IDs are arbitrary.

       EQ: TICK(AB,1) TICK(CD,1)

       and

       EQ: EQ(AB,CD)

       both canonicalize to:

           EQ:GROUP:AB|CD

    6. A sample is considered parseable when at least one native DSL
       field header appears.

       Partial DSL remains parseable. Missing gold facts are still FN.
    """

    raw = str(text or "")

    raw = re.sub(
        r"```(?:text|json)?",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    raw = raw.replace("```", "")

    # Permit semicolon-separated field lines.
    raw = raw.replace(";", "\n")

    payloads_by_field: dict[
        str,
        list[str],
    ] = defaultdict(list)

    native_header_seen = False

    for line in raw.splitlines():
        match = re.match(
            r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$",
            line,
        )

        if not match:
            continue

        field = match.group(1).upper()
        payload = match.group(2)

        # Critical: arbitrary prose headers must never create anchor facts.
        if field not in ANCHOR_DSL_FIELDS:
            continue

        native_header_seen = True
        payloads_by_field[field].append(payload)

    if not native_header_seen:
        return ParsedAnchor(
            parsed=False,
            facts_by_field={},
            all_facts=set(),
        )

    facts_by_field: dict[
        str,
        set[str],
    ] = defaultdict(set)

    for field, payloads in payloads_by_field.items():
        for original_payload in payloads:
            payload = normalized_space(
                original_payload
            )

            if not payload:
                continue

            # =================================================================
            # POINTS
            # =================================================================
            if field == "POINTS":
                # Deliberately case-sensitive.
                #
                # "A B" -> labels
                # "a and b" -> no uppercase labels
                points = set(
                    re.findall(
                        r"\b[A-Z]\b",
                        original_payload,
                    )
                )

                for point in points:
                    facts_by_field[field].add(
                        f"POINTS:POINT:{point}"
                    )

                continue

            # =================================================================
            # SEG
            # =================================================================
            if field == "SEG":
                upper_payload = payload.upper()

                segments = re.findall(
                    r"\b[A-Z]{2}\b",
                    upper_payload,
                )

                for segment in segments:
                    facts_by_field[field].add(
                        "SEG:SEG:"
                        + canonical_segment(segment)
                    )

                continue

            # =================================================================
            # EQ / PARA
            # =================================================================
            if field in {"EQ", "PARA"}:
                upper_payload = payload.upper()

                # -------------------------------------------------------------
                # Marker notation:
                #
                # EQ: TICK(AB,1) TICK(CD,1)
                # PARA: MARK(AB,2) MARK(CD,2)
                # -------------------------------------------------------------
                marker_calls = re.findall(
                    r"(?:MARK|TICK)\s*\(\s*"
                    r"([A-Z]{2})\s*,\s*([0-9]+)\s*\)",
                    upper_payload,
                )

                groups: dict[
                    str,
                    set[str],
                ] = defaultdict(set)

                for segment, marker_id in marker_calls:
                    groups[marker_id].add(
                        canonical_segment(segment)
                    )

                for segments in groups.values():
                    group = canonical_segment_group(
                        list(segments)
                    )

                    if group:
                        facts_by_field[field].add(
                            f"{field}:GROUP:{group}"
                        )

                # -------------------------------------------------------------
                # Direct notation:
                #
                # EQ: EQ(AB,CD)
                # PARA: PARA(AB,CD)
                #
                # Canonicalized to the same GROUP representation as markers.
                # -------------------------------------------------------------
                for name, args in split_function_calls(
                    payload
                ):
                    if name not in {"EQ", "PARA"}:
                        continue

                    if len(args) < 2:
                        continue

                    group = canonical_segment_group(
                        args
                    )

                    if group:
                        facts_by_field[field].add(
                            f"{field}:GROUP:{group}"
                        )

                # If we successfully found a structured representation,
                # do not also create RAW facts.
                if facts_by_field[field]:
                    continue

            # =================================================================
            # General function calls
            # =================================================================
            calls = split_function_calls(
                payload
            )

            if calls:
                for name, args in calls:
                    normalized_args = [
                        arg.strip()
                        for arg in args
                    ]

                    # ---------------------------------------------------------
                    # Perpendicular segments
                    # ---------------------------------------------------------
                    if name == "PERP" and len(normalized_args) >= 2:
                        pair = sorted(
                            canonical_segment(value)
                            for value in normalized_args[:2]
                        )

                        fact = (
                            f"{field}:{name}("
                            f"{','.join(pair)})"
                        )

                    # ---------------------------------------------------------
                    # Defensive support for direct EQ/PARA if they appear
                    # under another native field.
                    # ---------------------------------------------------------
                    elif name in {"EQ", "PARA"} and len(normalized_args) >= 2:
                        group = canonical_segment_group(
                            normalized_args
                        )

                        fact = (
                            f"{field}:{name}:GROUP:{group}"
                        )

                    # ---------------------------------------------------------
                    # Point lies on segment
                    # ---------------------------------------------------------
                    elif name == "ON" and len(normalized_args) >= 2:
                        point = canonical_point(
                            normalized_args[0]
                        )

                        segment = canonical_segment(
                            normalized_args[1]
                        )

                        fact = (
                            f"{field}:{name}("
                            f"{point},{segment})"
                        )

                    # ---------------------------------------------------------
                    # Circle argument order is treated as irrelevant for the
                    # native anchor representation.
                    # ---------------------------------------------------------
                    elif name == "CIRCLE":
                        values = sorted(
                            canonical_point(value)
                            for value in normalized_args
                        )

                        fact = (
                            f"{field}:{name}("
                            f"{','.join(values)})"
                        )

                    # ---------------------------------------------------------
                    # Angle/arc-like three-letter code:
                    # keep middle vertex fixed, allow arm reversal.
                    # ---------------------------------------------------------
                    elif name in {"ARC", "ARC2"} and len(normalized_args) == 1:
                        code = re.sub(
                            r"[^A-Z]",
                            "",
                            normalized_args[0].upper(),
                        )

                        if len(code) == 3:
                            first, vertex, third = code
                            first, third = sorted(
                                (first, third)
                            )
                            code = (
                                first
                                + vertex
                                + third
                            )

                        fact = (
                            f"{field}:{name}({code})"
                        )

                    # ---------------------------------------------------------
                    # Other known/simple functions.
                    # ---------------------------------------------------------
                    else:
                        values = [
                            str(value).strip().upper()
                            for value in normalized_args
                        ]

                        fact = (
                            f"{field}:{name}("
                            f"{','.join(values)})"
                        )

                    facts_by_field[field].add(
                        fact
                    )

                continue

            # =================================================================
            # Bare native-field tokens
            # =================================================================
            #
            # This keeps compatibility with visible_target_json fields whose
            # serialized values are simple bare symbols rather than functions.
            # Unknown top-level fields were already filtered above.
            # =================================================================
            for token in payload.split():
                normalized = (
                    token.strip()
                    .strip(",")
                    .upper()
                )

                if normalized:
                    facts_by_field[field].add(
                        f"{field}:RAW:{normalized}"
                    )

    all_facts = (
        set().union(
            *facts_by_field.values()
        )
        if facts_by_field
        else set()
    )

    return ParsedAnchor(
        parsed=True,
        facts_by_field=dict(facts_by_field),
        all_facts=all_facts,
    )


# =============================================================================
# Gold anchor conversion
# =============================================================================

def gold_anchor_from_visible_target(
    value: Any,
) -> ParsedAnchor:
    target = load_json_value(value)

    if not isinstance(target, dict):
        raise TypeError(
            "visible_target_json must decode to an object."
        )

    unknown_fields = sorted(
        set(map(str, target))
        - set(ANCHOR_DSL_FIELDS)
    )

    if unknown_fields:
        raise ValueError(
            "visible_target_json contains unknown field(s): "
            f"{unknown_fields}"
        )

    lines: list[str] = []

    for field in ANCHOR_DSL_FIELDS:
        values = target.get(
            field,
            [],
        )

        if not values:
            continue

        if not isinstance(values, list):
            raise TypeError(
                f"visible_target_json[{field!r}] "
                "must be a list."
            )

        lines.append(
            f"{field}: "
            + " ".join(
                str(item)
                for item in values
            )
        )

    gold = parse_anchor(
        "\n".join(lines)
    )

    if not gold.parsed:
        raise ValueError(
            "Gold visible_target_json did not contain "
            "a native anchor field."
        )

    if not gold.all_facts:
        raise ValueError(
            "Gold visible_target_json produced zero anchor facts."
        )

    return gold


# =============================================================================
# Dataset preparation
# =============================================================================

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
            "visible_target_json",
        },
        "Stage2",
    )

    rows = filter_task_rows(
        rows,
        "anchor",
    )

    if not rows:
        raise RuntimeError(
            "Stage2 contains no task_kind='anchor' rows."
        )

    validate_concepts(
        rows,
        STAGE2_CONCEPTS,
        "Stage2 anchor",
    )

    # -------------------------------------------------------------------------
    # Validate and cache all gold facts BEFORE loading any model.
    # -------------------------------------------------------------------------
    for index, row in enumerate(rows):
        try:
            row["_gold_anchor"] = (
                gold_anchor_from_visible_target(
                    row["visible_target_json"]
                )
            )

        except Exception as exc:
            raise ValueError(
                f"Invalid gold anchor at Stage2 row {index}."
            ) from exc

    warn_expected_count(
        len(rows),
        args.expected_anchor_rows,
        "Stage2 anchor evaluation set",
    )

    # Final dataset should contain one anchor row per figure.
    figure_ids = [
        str(row.get("figure_id") or "")
        for row in rows
    ]

    if figure_ids and all(figure_ids):
        duplicate_count = (
            len(figure_ids)
            - len(set(figure_ids))
        )

        if duplicate_count:
            raise ValueError(
                "Stage2 anchor evaluation set contains "
                f"{duplicate_count} duplicate figure_id value(s)."
            )

    rows = sort_rows(rows)

    rows = stratified_limit(
        rows,
        STAGE2_CONCEPTS,
        args.max_samples,
        seed=args.sample_seed,
    )

    print(
        f"Stage2 anchor rows: {len(rows):,}"
    )

    return rows


# =============================================================================
# Fact metrics
# =============================================================================

def fact_counts(
    pred: set[str],
    gold: set[str],
) -> Counter:
    return Counter(
        tp=len(pred & gold),
        fp=len(pred - gold),
        fn=len(gold - pred),
    )


def add_counts(
    destination: Counter,
    source: Counter,
) -> None:
    for key in (
        "tp",
        "fp",
        "fn",
    ):
        destination[key] += int(
            source.get(key, 0)
        )


def metric_from_counts(
    counts: Counter,
) -> dict[str, Any]:
    tp = int(counts.get("tp", 0))
    fp = int(counts.get("fp", 0))
    fn = int(counts.get("fn", 0))

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gold_facts": tp + fn,
        "pred_facts": tp + fp,
    }


# =============================================================================
# Anchor prompt
# =============================================================================

def anchor_prompt(
    row: dict[str, Any],
) -> str:
    prompt = str(
        row.get("prompt")
        or ""
    ).strip()

    return (
        prompt
        if prompt
        else DEFAULT_ANCHOR_PROMPT
    )


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


def parsed_anchor_json(
    anchor: ParsedAnchor,
) -> dict[str, Any]:
    return {
        "parsed": anchor.parsed,
        "facts": sorted(
            anchor.all_facts
        ),
        "facts_by_field": {
            field: sorted(
                anchor.facts_by_field.get(
                    field,
                    set(),
                )
            )
            for field in ANCHOR_DSL_FIELDS
            if anchor.facts_by_field.get(
                field
            )
        },
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

    # Keep already-generated outputs if a later sample crashes.
    handle.flush()


# =============================================================================
# Anchor evaluation
# =============================================================================

def evaluate_anchor(
    *,
    rows: list[dict[str, Any]],
    model_kind: str,
    tokenizer,
    model,
    image_processor,
    prediction_file: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    overall_counts = Counter(
        tp=0,
        fp=0,
        fn=0,
    )

    per_concept_counts = {
        concept: Counter(
            tp=0,
            fp=0,
            fn=0,
        )
        for concept in STAGE2_CONCEPTS
    }

    per_field_counts = {
        field: Counter(
            tp=0,
            fp=0,
            fn=0,
        )
        for field in ANCHOR_DSL_FIELDS
    }

    per_concept_total = Counter()
    per_concept_valid = Counter()
    per_concept_parse_success = Counter()
    per_concept_errors = Counter()

    valid_n = 0
    parse_success = 0
    error_n = 0

    prediction_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    progress = tqdm(
        rows,
        desc=f"{model_kind} / stage2 anchor",
        unit="sample",
    )

    with prediction_file.open(
        "w",
        encoding="utf-8",
    ) as prediction_log:

        for index, row in enumerate(
            progress
        ):
            concept = str(
                row["concept"]
            )

            sample_id = stable_sample_key(
                row,
                index,
            )

            figure_id = row.get(
                "figure_id"
            )

            per_concept_total[
                concept
            ] += 1

            gold: ParsedAnchor = row[
                "_gold_anchor"
            ]

            prompt = anchor_prompt(
                row
            )

            response = None

            pred = ParsedAnchor(
                parsed=False,
                facts_by_field={},
                all_facts=set(),
            )

            error = None
            sample_metric = None

            # =================================================================
            # Model inference + parse
            # =================================================================
            try:
                response = generate_response(
                    prompt_text=prompt,
                    tokenizer=tokenizer,
                    model=model,
                    image_processor=image_processor,
                    conv_mode=args.conv_mode,
                    max_new_tokens=args.max_new_tokens,
                    image_value=row["image"],
                    image_mode="normal",
                )

                pred = parse_anchor(
                    response
                )

            except Exception as exc:
                if is_cuda_oom(exc):
                    raise

                error = exception_record(
                    exc
                )

                error_n += 1

                per_concept_errors[
                    concept
                ] += 1

                print(
                    "\nWARNING: anchor error "
                    f"[{model_kind}/{sample_id}]: "
                    f"{type(exc).__name__}: {exc}"
                )

            # =================================================================
            # Valid output scoring
            # =================================================================
            if error is None:
                valid_n += 1

                per_concept_valid[
                    concept
                ] += 1

                if pred.parsed:
                    parse_success += 1

                    per_concept_parse_success[
                        concept
                    ] += 1

                # Parse-failed valid outputs intentionally have an empty
                # prediction set, so every gold fact becomes FN.
                sample_counts = fact_counts(
                    pred.all_facts,
                    gold.all_facts,
                )

                sample_metric = metric_from_counts(
                    sample_counts
                )

                add_counts(
                    overall_counts,
                    sample_counts,
                )

                add_counts(
                    per_concept_counts[
                        concept
                    ],
                    sample_counts,
                )

                # -------------------------------------------------------------
                # Field scoring
                # -------------------------------------------------------------
                for field in ANCHOR_DSL_FIELDS:
                    gold_facts = (
                        gold.facts_by_field.get(
                            field,
                            set(),
                        )
                    )

                    pred_facts = (
                        pred.facts_by_field.get(
                            field,
                            set(),
                        )
                    )

                    add_counts(
                        per_field_counts[
                            field
                        ],
                        fact_counts(
                            pred_facts,
                            gold_facts,
                        ),
                    )

            # =================================================================
            # Per-sample JSONL
            # =================================================================
            record = {
                "dataset": "stage2",
                "task_kind": "anchor",
                "model": model_kind,
                "sample_id": sample_id,
                "figure_id": figure_id,
                "concept": concept,
                "prompt": prompt,
                "response": response,
                "gold": parsed_anchor_json(
                    gold
                ),
                "prediction": parsed_anchor_json(
                    pred
                ),
                "sample_metric": sample_metric,
                "error": error,
            }

            write_jsonl_record(
                prediction_log,
                record,
            )

            if error_n > args.max_sample_errors:
                raise RuntimeError(
                    "Aborting anchor evaluation: "
                    f"more than {args.max_sample_errors} "
                    f"errors occurred for {model_kind}."
                )

            # =================================================================
            # Progress
            # =================================================================
            current_metric = metric_from_counts(
                overall_counts
            )

            parse_fail_rate = (
                1.0
                - parse_success / valid_n
                if valid_n
                else 0.0
            )

            progress.set_postfix(
                f1=(
                    f"{current_metric['f1'] * 100:.1f}%"
                ),
                parse_fail=(
                    f"{parse_fail_rate * 100:.1f}%"
                ),
                errors=error_n,
            )

    parse_fail = (
        valid_n
        - parse_success
    )

    n = len(rows)

    return {
        "n": n,
        "valid_n": valid_n,
        "error_n": error_n,
        "error_rate": (
            error_n / n
            if n
            else 0.0
        ),
        "parse_success": parse_success,
        "parse_fail": parse_fail,
        "parse_success_rate": (
            parse_success / valid_n
            if valid_n
            else 0.0
        ),
        "parse_fail_rate": (
            parse_fail / valid_n
            if valid_n
            else 0.0
        ),
        "prediction_file": str(
            prediction_file
        ),

        "overall": metric_from_counts(
            overall_counts
        ),

        "per_concept": {
            concept: {
                **metric_from_counts(
                    per_concept_counts[
                        concept
                    ]
                ),
                "n": int(
                    per_concept_total[
                        concept
                    ]
                ),
                "valid_n": int(
                    per_concept_valid[
                        concept
                    ]
                ),
                "error_n": int(
                    per_concept_errors[
                        concept
                    ]
                ),
                "parse_success": int(
                    per_concept_parse_success[
                        concept
                    ]
                ),
                "parse_fail": int(
                    per_concept_valid[
                        concept
                    ]
                    - per_concept_parse_success[
                        concept
                    ]
                ),
                "parse_fail_rate": (
                    (
                        per_concept_valid[
                            concept
                        ]
                        - per_concept_parse_success[
                            concept
                        ]
                    )
                    / per_concept_valid[
                        concept
                    ]
                    if per_concept_valid[
                        concept
                    ]
                    else 0.0
                ),
            }
            for concept in STAGE2_CONCEPTS
        },

        "per_field": {
            field: metric_from_counts(
                per_field_counts[
                    field
                ]
            )
            for field in ANCHOR_DSL_FIELDS
        },
    }


# =============================================================================
# Markdown formatting
# =============================================================================

def pct(
    value: float | None,
) -> str:
    if value is None:
        return "-"

    return (
        f"{value * 100:.2f}%"
    )


def average_table(
    results: dict[str, Any],
) -> list[str]:
    lines = [
        "## Stage 2 Dataset — Overall Anchor Score",
        "",
        (
            "| Model | Fact Precision | Fact Recall | Fact F1 | "
            "Parse Fail Rate | Error Rate | Valid N | N |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for model_name in MODEL_ORDER:
        result = results.get(
            model_name
        )

        if result is None:
            lines.append(
                f"| {model_name} "
                "| - | - | - | - | - | - | - |"
            )
            continue

        overall = result[
            "overall"
        ]

        lines.append(
            f"| {model_name} "
            f"| {pct(overall['precision'])} "
            f"| {pct(overall['recall'])} "
            f"| {pct(overall['f1'])} "
            f"| {pct(result['parse_fail_rate'])} "
            f"| {pct(result['error_rate'])} "
            f"| {result['valid_n']:,} "
            f"| {result['n']:,} |"
        )

    lines.append("")
    return lines


def concept_tables(
    results: dict[str, Any],
) -> list[str]:
    lines = [
        "## Stage 2 Dataset — Per-Concept Anchor Score",
        "",
    ]

    for model_name in MODEL_ORDER:
        result = results.get(
            model_name
        )

        if result is None:
            continue

        lines.extend(
            [
                f"### {model_name}",
                "",
                (
                    "| Concept | Precision | Recall | Fact F1 | "
                    "Parse Fail | Errors | Valid N | N |"
                ),
                (
                    "|---|---:|---:|---:|---:|---:|---:|---:|"
                ),
            ]
        )

        for concept in STAGE2_CONCEPTS:
            item = result[
                "per_concept"
            ][concept]

            lines.append(
                f"| {concept} "
                f"| {pct(item['precision'])} "
                f"| {pct(item['recall'])} "
                f"| {pct(item['f1'])} "
                f"| {pct(item['parse_fail_rate'])} "
                f"| {item['error_n']:,} "
                f"| {item['valid_n']:,} "
                f"| {item['n']:,} |"
            )

        lines.append("")

    return lines


def field_tables(
    results: dict[str, Any],
) -> list[str]:
    lines = [
        "## Stage 2 Dataset — Field-wise Anchor Score",
        "",
    ]

    for model_name in MODEL_ORDER:
        result = results.get(
            model_name
        )

        if result is None:
            continue

        lines.extend(
            [
                f"### {model_name}",
                "",
                (
                    "| Field | Precision | Recall | Fact F1 | "
                    "Gold Facts | Pred Facts |"
                ),
                (
                    "|---|---:|---:|---:|---:|---:|"
                ),
            ]
        )

        for field in ANCHOR_DSL_FIELDS:
            item = result[
                "per_field"
            ][field]

            # No denominator exists if neither gold nor predictions contain
            # this field anywhere in the valid evaluation set.
            if (
                item["gold_facts"] == 0
                and item["pred_facts"] == 0
            ):
                precision = None
                recall = None
                f1 = None
            else:
                precision = item[
                    "precision"
                ]
                recall = item[
                    "recall"
                ]
                f1 = item[
                    "f1"
                ]

            lines.append(
                f"| {field} "
                f"| {pct(precision)} "
                f"| {pct(recall)} "
                f"| {pct(f1)} "
                f"| {item['gold_facts']:,} "
                f"| {item['pred_facts']:,} |"
            )

        lines.append("")

    return lines


def prediction_file_table(
    results: dict[str, Any],
) -> list[str]:
    lines = [
        "## Prediction Logs",
        "",
        "| Model | JSONL |",
        "|---|---|",
    ]

    for model_name in MODEL_ORDER:
        result = results.get(
            model_name
        )

        if result is None:
            continue

        lines.append(
            f"| {model_name} "
            f"| `{result['prediction_file']}` |"
        )

    lines.append("")
    return lines


def write_markdown(
    *,
    results: dict[str, Any],
    output_file: Path,
    complete: bool,
) -> None:
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines: list[str] = [
        "# Geometry Anchor Results",
        "",
        f"Status: **{'complete' if complete else 'partial'}**",
        "",
        "## Metric Definitions",
        "",
        (
            "- **Fact Precision / Recall / F1**: micro scores over "
            "canonical facts reconstructed from the native Stage2 anchor DSL."
        ),
        (
            "- **Overall Fact F1** pools TP / FP / FN across all valid "
            "Stage2 anchor samples."
        ),
        (
            "- **Concept-wise Fact F1** pools TP / FP / FN only within "
            "samples of that Stage2 concept."
        ),
        (
            "- **Field-wise Fact F1** pools TP / FP / FN only for one "
            "native DSL field such as SEG, EQ, PARA, or ON."
        ),
        (
            "- **Parse Fail Rate**: valid inference outputs with no parseable "
            "native DSL representation. Parse-failed outputs remain in Fact F1 "
            "with an empty predicted fact set, so gold facts become FN."
        ),
        (
            "- **Error Rate**: inference/scoring exceptions. Errors are "
            "reported separately from parse failures and excluded from "
            "valid-output Fact F1."
        ),
        "",
        (
            "Anchor Fact F1 has a theoretical minimum of 0. "
            "There is no natural uniform random-choice chance level analogous "
            "to closed-set concept classification."
        ),
        "",
    ]

    lines.extend(
        average_table(
            results
        )
    )

    lines.extend(
        concept_tables(
            results
        )
    )

    lines.extend(
        field_tables(
            results
        )
    )

    lines.extend(
        prediction_file_table(
            results
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
        / "anchor_predictions"
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
        f"Anchor result   : "
        f"{output_file}"
    )

    print(
        f"Prediction logs : "
        f"{prediction_dir}"
    )

    # Validate all gold targets before loading a 7B checkpoint.
    anchor_rows = prepare_stage2_rows(
        args
    )

    results: dict[
        str,
        Any,
    ] = {}

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

            results[
                model_kind
            ] = evaluate_anchor(
                rows=anchor_rows,
                model_kind=model_kind,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                prediction_file=(
                    prediction_dir
                    / f"{model_kind}_stage2_anchor.jsonl"
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

    print(
        "\nAnchor evaluation completed."
    )

    print(
        f"Final result: "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()