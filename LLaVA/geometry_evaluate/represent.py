#!/usr/bin/env python3
from __future__ import annotations

"""
Geometry representation evaluator.

1. Primitive preservation
   -> Stage1 checkpoint-to-checkpoint RSA

2. Concept formation
   -> Concept Intra / Inter / Separation

3. Primitive scaffolding
   -> Parent Margin

4. Cross-modal semantic alignment
   -> Text Anchor Margin
   -> sample-level Linear CKA

Representations
raw:
    h(image, prompt)

delta:
    h(image, prompt) - h(blank image, prompt)

For text:
raw:
    h(description + representation prompt)

delta:
    h(description + representation prompt)
    - h(representation prompt)

Shared infrastructure:
    helper.py

Shared geometry definitions:
    ontology.py
"""

import argparse
import gc
import hashlib
import math
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from helper import (
    EvaluationModelPaths,
    decode_image,
    filter_task_rows,
    forward_multimodal_hidden_states,
    forward_text_hidden_states,
    load_eval_model,
    read_parquet_files,
    stable_sample_key,
    stratified_limit,
    unload_model,
    validate_concepts,
    validate_nonempty_column,
    validate_required_columns,
    warn_expected_count,
)

from ontology import (
    PARENT_ONTOLOGY,
    STAGE1_CONCEPTS,
    STAGE2_CONCEPTS,
)


# =============================================================================
# Constants
# =============================================================================

MODEL_ORDER = (
    "base",
    "stage1",
    "stage2",
    "stage2_only",
)

REP_TYPES = (
    "raw",
    "delta",
)

DEFAULT_LAYERS = (
    8,
    16,
    24,
    32,
)

DEFAULT_PRIMARY_LAYER = 24

DEFAULT_STAGE1_PATTERN = (
    "stage1_geometry_evaluate_evaluate_*.parquet"
)

DEFAULT_STAGE2_PATTERN = (
    "stage2_geometry_evaluate_evaluate_*.parquet"
)

DEFAULT_REP_PROMPT = (
    "Represent the geometric configuration."
)

DEFAULT_TEXT_COLUMN = (
    "neutral_description_no_labels"
)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate geometry representation structure."
    )

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--stage1-data-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--stage2-data-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--stage1-pattern",
        default=DEFAULT_STAGE1_PATTERN,
    )

    parser.add_argument(
        "--stage2-pattern",
        default=DEFAULT_STAGE2_PATTERN,
    )

    # -------------------------------------------------------------------------
    # Models
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-file",
        default="representation_results.md",
    )

    # -------------------------------------------------------------------------
    # Representation extraction
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--conv-mode",
        default="llava_v1",
    )

    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(DEFAULT_LAYERS),
    )

    parser.add_argument(
        "--primary-layer",
        type=int,
        default=DEFAULT_PRIMARY_LAYER,
    )

    parser.add_argument(
        "--representation-prompt",
        default=DEFAULT_REP_PROMPT,
    )

    parser.add_argument(
        "--text-column",
        default=DEFAULT_TEXT_COLUMN,
    )

    # -------------------------------------------------------------------------
    # Null tests
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--permutations",
        type=int,
        default=999,
    )

    parser.add_argument(
        "--null-pair-samples",
        type=int,
        default=50_000,
        help=(
            "Number of random sample pairs used by the "
            "concept-separation permutation test."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260822,
    )

    # -------------------------------------------------------------------------
    # Smoke testing
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help=(
            "0 evaluates the full dataset. Positive values use "
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

    args = parser.parse_args()

    args.layers = tuple(
        dict.fromkeys(args.layers)
    )

    if args.primary_layer not in args.layers:
        raise ValueError(
            "--primary-layer must also appear in --layers."
        )

    if args.permutations < 1:
        raise ValueError(
            "--permutations must be >= 1."
        )

    if args.null_pair_samples < 1:
        raise ValueError(
            "--null-pair-samples must be >= 1."
        )

    # Concept structure requires at least two examples per Stage2 concept.
    if (
        0 < args.max_samples
        < 2 * len(STAGE2_CONCEPTS)
    ):
        raise ValueError(
            "--max-samples must be 0 or at least "
            f"{2 * len(STAGE2_CONCEPTS)} for representation evaluation."
        )

    return args


# =============================================================================
# Data discovery
# =============================================================================

def discover_parquets(
    directory: Path,
    pattern: str,
    dataset_name: str,
) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(
            f"{dataset_name} data directory was not found: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"{dataset_name} data path is not a directory: {directory}"
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


def sort_rows(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    indexed = list(
        enumerate(rows)
    )

    indexed.sort(
        key=lambda item: (
            str(
                item[1].get("concept")
                or ""
            ),
            stable_sample_key(
                item[1],
                item[0],
            ),
        )
    )

    return [
        row
        for _, row in indexed
    ]


def validate_unique_figures(
    rows: Sequence[dict[str, Any]],
    dataset_name: str,
) -> None:
    figure_ids = [
        str(
            row.get("figure_id")
            or ""
        )
        for row in rows
    ]

    if not figure_ids or not all(
        figure_ids
    ):
        return

    duplicate_count = (
        len(figure_ids)
        - len(set(figure_ids))
    )

    if duplicate_count:
        raise ValueError(
            f"{dataset_name} contains "
            f"{duplicate_count} duplicate figure_id value(s)."
        )


# =============================================================================
# Dataset preparation
# =============================================================================

def prepare_stage1_rows(
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    paths = discover_parquets(
        args.stage1_data_dir,
        args.stage1_pattern,
        "Stage1",
    )

    rows = read_parquet_files(
        paths,
        "Stage1",
    )

    validate_required_columns(
        rows,
        {
            "concept",
            "image",
        },
        "Stage1",
    )

    validate_concepts(
        rows,
        STAGE1_CONCEPTS,
        "Stage1",
    )

    warn_expected_count(
        len(rows),
        args.expected_stage1_rows,
        "Stage1 representation set",
    )

    rows = sort_rows(
        rows
    )

    rows = stratified_limit(
        rows,
        STAGE1_CONCEPTS,
        args.max_samples,
        seed=args.seed,
    )

    print(
        f"Stage1 representation rows: "
        f"{len(rows):,}"
    )

    return rows


def prepare_stage2_rows(
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    paths = discover_parquets(
        args.stage2_data_dir,
        args.stage2_pattern,
        "Stage2",
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
            args.text_column,
        },
        "Stage2",
    )

    # One local row per figure is sufficient.
    # The anchor row contains the same image.
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

    validate_nonempty_column(
        rows,
        args.text_column,
        "Stage2 local",
    )

    validate_unique_figures(
        rows,
        "Stage2 local representation set",
    )

    warn_expected_count(
        len(rows),
        args.expected_stage2_local_rows,
        "Stage2 local representation set",
    )

    rows = sort_rows(
        rows
    )

    rows = stratified_limit(
        rows,
        STAGE2_CONCEPTS,
        args.max_samples,
        seed=args.seed,
    )

    print(
        f"Stage2 representation rows: "
        f"{len(rows):,}"
    )

    return rows


# =============================================================================
# Representation extraction
# =============================================================================

def empty_rep_storage(
    layers: Sequence[int],
) -> dict[str, dict[int, list[torch.Tensor]]]:
    return {
        rep: {
            layer: []
            for layer in layers
        }
        for rep in REP_TYPES
    }


def stack_rep_storage(
    storage: dict[
        str,
        dict[
            int,
            list[torch.Tensor],
        ],
    ],
    layers: Sequence[int],
) -> dict[str, dict[int, torch.Tensor]]:
    """
    Store extracted representations as CPU float16 tensors.

    Metrics convert them to float32/float64 as needed.
    """

    return {
        rep: {
            layer: torch.stack(
                storage[rep][layer]
            ).half()
            for layer in layers
        }
        for rep in REP_TYPES
    }


def extract_image_representations(
    *,
    rows: Sequence[dict[str, Any]],
    description: str,
    tokenizer,
    model,
    image_processor,
    args: argparse.Namespace,
) -> dict[str, dict[int, torch.Tensor]]:
    storage = empty_rep_storage(
        args.layers
    )

    # Same prompt + same blank-image size yields the same control input.
    blank_cache: dict[
        tuple[int, int],
        dict[int, torch.Tensor],
    ] = {}

    for row in tqdm(
        rows,
        desc=f"{description} image",
        unit="sample",
    ):
        image = decode_image(
            row["image"]
        )

        normal = forward_multimodal_hidden_states(
            image_value=image,
            prompt_text=args.representation_prompt,
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            conv_mode=args.conv_mode,
            layers=args.layers,
            image_mode="normal",
        )

        if image.size not in blank_cache:
            blank_cache[
                image.size
            ] = forward_multimodal_hidden_states(
                image_value=image,
                prompt_text=args.representation_prompt,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                conv_mode=args.conv_mode,
                layers=args.layers,
                image_mode="blank",
            )

        blank = blank_cache[
            image.size
        ]

        for layer in args.layers:
            storage[
                "raw"
            ][layer].append(
                normal[layer]
            )

            storage[
                "delta"
            ][layer].append(
                normal[layer]
                - blank[layer]
            )

    return stack_rep_storage(
        storage,
        args.layers,
    )


def extract_text_representations(
    *,
    rows: Sequence[dict[str, Any]],
    description: str,
    tokenizer,
    model,
    args: argparse.Namespace,
) -> dict[str, dict[int, torch.Tensor]]:
    storage = empty_rep_storage(
        args.layers
    )

    # Prompt-only control for text delta.
    baseline = forward_text_hidden_states(
        prompt_text=args.representation_prompt,
        tokenizer=tokenizer,
        model=model,
        conv_mode=args.conv_mode,
        layers=args.layers,
    )

    for row in tqdm(
        rows,
        desc=f"{description} text",
        unit="sample",
    ):
        text = str(
            row[
                args.text_column
            ]
        ).strip()

        prompt = (
            f"{text}\n\n"
            f"{args.representation_prompt}"
        )

        state = forward_text_hidden_states(
            prompt_text=prompt,
            tokenizer=tokenizer,
            model=model,
            conv_mode=args.conv_mode,
            layers=args.layers,
        )

        for layer in args.layers:
            storage[
                "raw"
            ][layer].append(
                state[layer]
            )

            storage[
                "delta"
            ][layer].append(
                state[layer]
                - baseline[layer]
            )

    return stack_rep_storage(
        storage,
        args.layers,
    )


# =============================================================================
# Numerical helpers
# =============================================================================

def normalize_rows(
    vectors: torch.Tensor,
) -> torch.Tensor:
    """
    F.normalize leaves exact zero vectors at zero when eps is supplied.
    """

    return F.normalize(
        vectors.float(),
        dim=1,
        eps=1e-12,
    )


def safe_mean(
    values: Sequence[float],
) -> float | None:
    if not values:
        return None

    return float(
        sum(values)
        / len(values)
    )


def format_value(
    value: float | None,
    digits: int = 4,
) -> str:
    if value is None:
        return "-"

    if math.isnan(value):
        return "nan"

    if math.isinf(value):
        return "inf"

    return f"{value:.{digits}f}"


# =============================================================================
# Deterministic derived seeds
# =============================================================================

def derived_seed(
    base_seed: int,
    *parts: Any,
) -> int:
    """
    Stable seed independent of Python's randomized hash().
    """

    payload = ":".join(
        [
            str(base_seed),
            *[
                str(part)
                for part in parts
            ],
        ]
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    ) % (2**31 - 1)


# =============================================================================
# Concept structure
# =============================================================================

def concept_structure(
    vectors: torch.Tensor,
    labels: Sequence[str],
    concepts: Sequence[str],
) -> dict[str, Any]:
    """
    Cosine structure.

    Intra:
        average cosine similarity between samples with the same concept.

    Inter:
        average cosine similarity between samples with different concepts.

    Separation:
        Intra - Inter

    Pair sums are accumulated in float64 and subtract the actual
    sum_i ||x_i||^2 rather than assuming every normalized row has norm 1.
    This keeps the formula correct even for exact zero vectors.
    """

    x = normalize_rows(
        vectors
    ).double()

    n = len(labels)

    if n < 2:
        raise ValueError(
            "Concept structure requires at least two samples."
        )

    total_sum = x.sum(
        dim=0
    )

    total_self_sum = (
        x.square()
        .sum()
        .item()
    )

    total_pair_sum = (
        (
            total_sum
            @ total_sum
        ).item()
        - total_self_sum
    ) / 2.0

    total_pair_count = (
        n
        * (n - 1)
        // 2
    )

    same_sum = 0.0
    same_count = 0

    per_concept: dict[
        str,
        dict[str, Any],
    ] = {}

    for concept in concepts:
        indices = [
            index
            for index, label
            in enumerate(labels)
            if label == concept
        ]

        group = x[
            indices
        ]

        group_n = len(
            indices
        )

        if group_n == 0:
            per_concept[
                concept
            ] = {
                "n": 0,
                "intra": None,
                "inter": None,
                "separation": None,
            }
            continue

        group_sum = group.sum(
            dim=0
        )

        group_self_sum = (
            group.square()
            .sum()
            .item()
        )

        within_sum = (
            (
                group_sum
                @ group_sum
            ).item()
            - group_self_sum
        ) / 2.0

        within_count = (
            group_n
            * (group_n - 1)
            // 2
        )

        intra = (
            within_sum / within_count
            if within_count
            else None
        )

        outside_n = (
            n - group_n
        )

        inter_count = (
            group_n
            * outside_n
        )

        inter = (
            (
                group_sum
                @ (
                    total_sum
                    - group_sum
                )
            ).item()
            / inter_count
            if inter_count
            else None
        )

        separation = (
            intra - inter
            if (
                intra is not None
                and inter is not None
            )
            else None
        )

        per_concept[
            concept
        ] = {
            "n": group_n,
            "intra": intra,
            "inter": inter,
            "separation": separation,
        }

        same_sum += (
            within_sum
        )

        same_count += (
            within_count
        )

    different_sum = (
        total_pair_sum
        - same_sum
    )

    different_count = (
        total_pair_count
        - same_count
    )

    if same_count == 0:
        raise ValueError(
            "No within-concept sample pair exists."
        )

    if different_count == 0:
        raise ValueError(
            "No between-concept sample pair exists."
        )

    intra = (
        same_sum
        / same_count
    )

    inter = (
        different_sum
        / different_count
    )

    return {
        "intra": intra,
        "inter": inter,
        "separation": (
            intra - inter
        ),
        "concept": per_concept,
    }


# =============================================================================
# Permutation null helpers
# =============================================================================

def null_summary(
    observed: float,
    null_values: Sequence[float],
) -> dict[str, float]:
    values = np.asarray(
        null_values,
        dtype=np.float64,
    )

    null_mean = float(
        values.mean()
    )

    null_std = float(
        values.std()
    )

    if null_std > 0:
        z = (
            observed
            - null_mean
        ) / null_std

    elif observed > null_mean:
        z = math.inf

    elif observed < null_mean:
        z = -math.inf

    else:
        z = 0.0

    # One-sided empirical p:
    # H1 = observed statistic is larger than the permutation null.
    p = (
        1
        + int(
            np.sum(
                values
                >= observed
            )
        )
    ) / (
        len(values)
        + 1
    )

    return {
        "observed": float(
            observed
        ),
        "mean": null_mean,
        "std": null_std,
        "z": float(z),
        "p": float(p),
    }


def separation_null(
    *,
    vectors: torch.Tensor,
    labels: Sequence[str],
    pair_samples: int,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    """
    Pair-sampled permutation test for Concept Separation.

    The main Concept Separation reported elsewhere is exact.

    The permutation null uses a fixed random sample of valid sample pairs
    so 999 permutations remain inexpensive.

    `observed` in the returned null object is therefore the separation
    measured on the same sampled pairs used by the null distribution.
    """

    x = normalize_rows(
        vectors
    )

    n = len(
        labels
    )

    if n < 2:
        raise ValueError(
            "Separation null requires at least two samples."
        )

    max_pairs = (
        n
        * (n - 1)
        // 2
    )

    pair_count = min(
        pair_samples,
        max_pairs,
    )

    generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    # Random directed pairs without self-pairs.
    # Repeated pairs are allowed; at 50k / millions of available Stage2 pairs
    # the resulting approximation remains small relative to the full space.
    first = torch.randint(
        0,
        n,
        (pair_count,),
        generator=generator,
    )

    second = torch.randint(
        0,
        n - 1,
        (pair_count,),
        generator=generator,
    )

    second += (
        second >= first
    ).long()

    similarities = (
        x[first]
        * x[second]
    ).sum(
        dim=1
    )

    label_to_id = {
        label: index
        for index, label
        in enumerate(
            sorted(
                set(labels)
            )
        )
    }

    label_ids = torch.tensor(
        [
            label_to_id[label]
            for label in labels
        ],
        dtype=torch.long,
    )

    same_observed = (
        label_ids[first]
        == label_ids[second]
    )

    if (
        not same_observed.any()
        or same_observed.all()
    ):
        raise RuntimeError(
            "Pair sample did not contain both same- and "
            "different-concept pairs."
        )

    observed = float(
        similarities[
            same_observed
        ].mean()
        - similarities[
            ~same_observed
        ].mean()
    )

    permutation_generator = (
        torch.Generator()
        .manual_seed(
            seed + 1
        )
    )

    null_values: list[
        float
    ] = []

    for _ in range(
        permutations
    ):
        shuffled = label_ids[
            torch.randperm(
                n,
                generator=permutation_generator,
            )
        ]

        same = (
            shuffled[first]
            == shuffled[second]
        )

        if (
            not same.any()
            or same.all()
        ):
            continue

        value = float(
            similarities[
                same
            ].mean()
            - similarities[
                ~same
            ].mean()
        )

        null_values.append(
            value
        )

    if not null_values:
        raise RuntimeError(
            "No valid concept-separation permutation was produced."
        )

    return null_summary(
        observed,
        null_values,
    )


# =============================================================================
# Concept centroids
# =============================================================================

def concept_centroids(
    vectors: torch.Tensor,
    labels: Sequence[str],
    concepts: Sequence[str],
) -> torch.Tensor:
    vectors = vectors.float()

    centroids: list[
        torch.Tensor
    ] = []

    for concept in concepts:
        indices = [
            index
            for index, label
            in enumerate(labels)
            if label == concept
        ]

        if not indices:
            raise ValueError(
                f"No samples were found for concept {concept!r}."
            )

        centroid = vectors[
            indices
        ].mean(
            dim=0
        )

        centroids.append(
            centroid
        )

    return F.normalize(
        torch.stack(
            centroids
        ),
        dim=1,
        eps=1e-12,
    )


# =============================================================================
# RSA
# =============================================================================

def average_ranks(
    values: np.ndarray,
) -> np.ndarray:
    """
    Average ranks with tie handling.
    """

    if len(values) == 0:
        return np.asarray(
            [],
            dtype=np.float64,
        )

    order = np.argsort(
        values,
        kind="mergesort",
    )

    sorted_values = values[
        order
    ]

    ranks = np.empty(
        len(values),
        dtype=np.float64,
    )

    start = 0

    while start < len(values):
        end = (
            start + 1
        )

        while (
            end < len(values)
            and sorted_values[end]
            == sorted_values[start]
        ):
            end += 1

        average_rank = (
            (
                start
                + end
                - 1
            )
            / 2.0
            + 1.0
        )

        ranks[
            order[start:end]
        ] = average_rank

        start = end

    return ranks


def pearson(
    first: np.ndarray,
    second: np.ndarray,
) -> float | None:
    if (
        len(first) == 0
        or len(second) == 0
        or len(first) != len(second)
    ):
        return None

    first = (
        first
        - first.mean()
    )

    second = (
        second
        - second.mean()
    )

    denominator = math.sqrt(
        float(
            (first @ first)
            * (second @ second)
        )
    )

    if denominator == 0:
        return None

    return float(
        (first @ second)
        / denominator
    )


def rsa_signature(
    vectors: torch.Tensor,
) -> np.ndarray:
    """
    Spearman-RSA signature:
    rank-transform the upper-triangle within-sample cosine similarities.
    """

    if len(vectors) < 2:
        return np.asarray(
            [],
            dtype=np.float64,
        )

    x = normalize_rows(
        vectors
    )

    similarity = (
        x
        @ x.T
    )

    indices = torch.triu_indices(
        len(x),
        len(x),
        offset=1,
    )

    values = (
        similarity[
            indices[0],
            indices[1],
        ]
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )

    return average_ranks(
        values
    )


def build_stage1_rsa_signatures(
    *,
    representations: dict[
        str,
        dict[int, torch.Tensor],
    ],
    labels: Sequence[str],
    layers: Sequence[int],
    primary_layer: int,
) -> dict[str, dict[int, Any]]:
    signatures: dict[
        str,
        dict[int, Any],
    ] = {}

    for rep in REP_TYPES:
        signatures[
            rep
        ] = {}

        for layer in layers:
            item: dict[
                str,
                Any,
            ] = {
                "all": rsa_signature(
                    representations[
                        rep
                    ][layer]
                ),
                "concept": {},
            }

            if layer == primary_layer:
                for concept in STAGE1_CONCEPTS:
                    indices = [
                        index
                        for index, label
                        in enumerate(labels)
                        if label == concept
                    ]

                    item[
                        "concept"
                    ][concept] = rsa_signature(
                        representations[
                            rep
                        ][layer][
                            indices
                        ]
                    )

            signatures[
                rep
            ][layer] = item

    return signatures


def compare_rsa_signatures(
    first: dict[str, dict[int, Any]],
    second: dict[str, dict[int, Any]],
    layers: Sequence[int],
    primary_layer: int,
) -> dict[str, Any]:
    overall: list[
        tuple[int, str, float | None]
    ] = []

    per_concept: list[
        tuple[str, str, float | None]
    ] = []

    for rep in REP_TYPES:
        for layer in layers:
            score = pearson(
                first[
                    rep
                ][layer]["all"],
                second[
                    rep
                ][layer]["all"],
            )

            overall.append(
                (
                    layer,
                    rep,
                    score,
                )
            )

        for concept in STAGE1_CONCEPTS:
            score = pearson(
                first[
                    rep
                ][primary_layer][
                    "concept"
                ][concept],
                second[
                    rep
                ][primary_layer][
                    "concept"
                ][concept],
            )

            per_concept.append(
                (
                    rep,
                    concept,
                    score,
                )
            )

    return {
        "overall": overall,
        "concept": per_concept,
    }


# =============================================================================
# Parent Margin
# =============================================================================

def parent_margin_from_assignment(
    similarity: torch.Tensor,
    assignments: Sequence[Sequence[str]],
) -> tuple[
    float,
    float,
    float,
    dict[
        str,
        tuple[
            float,
            float,
            float,
        ],
    ],
]:
    """
    Compute Parent Margin as a macro average over Stage2 concepts.

    Each concept contributes equally regardless of whether its ontology
    contains two or three primitive parents.
    """

    primitive_index = {
        primitive: index
        for index, primitive
        in enumerate(
            STAGE1_CONCEPTS
        )
    }

    parent_means: list[
        float
    ] = []

    nonparent_means: list[
        float
    ] = []

    margins: list[
        float
    ] = []

    detail: dict[
        str,
        tuple[
            float,
            float,
            float,
        ],
    ] = {}

    for relation_index, concept in enumerate(
        STAGE2_CONCEPTS
    ):
        parents = set(
            assignments[
                relation_index
            ]
        )

        parent_values = [
            float(
                similarity[
                    relation_index,
                    primitive_index[
                        primitive
                    ],
                ]
            )
            for primitive in STAGE1_CONCEPTS
            if primitive in parents
        ]

        nonparent_values = [
            float(
                similarity[
                    relation_index,
                    primitive_index[
                        primitive
                    ],
                ]
            )
            for primitive in STAGE1_CONCEPTS
            if primitive not in parents
        ]

        if (
            not parent_values
            or not nonparent_values
        ):
            raise ValueError(
                f"Invalid parent assignment for {concept!r}."
            )

        parent_mean = float(
            np.mean(
                parent_values
            )
        )

        nonparent_mean = float(
            np.mean(
                nonparent_values
            )
        )

        margin = (
            parent_mean
            - nonparent_mean
        )

        parent_means.append(
            parent_mean
        )

        nonparent_means.append(
            nonparent_mean
        )

        margins.append(
            margin
        )

        detail[
            concept
        ] = (
            parent_mean,
            nonparent_mean,
            margin,
        )

    return (
        float(
            np.mean(
                parent_means
            )
        ),
        float(
            np.mean(
                nonparent_means
            )
        ),
        float(
            np.mean(
                margins
            )
        ),
        detail,
    )


def parent_metric(
    *,
    stage2_vectors: torch.Tensor,
    stage2_labels: Sequence[str],
    stage1_vectors: torch.Tensor,
    stage1_labels: Sequence[str],
) -> dict[str, Any]:
    relation_centroids = concept_centroids(
        stage2_vectors,
        stage2_labels,
        STAGE2_CONCEPTS,
    )

    primitive_centroids = concept_centroids(
        stage1_vectors,
        stage1_labels,
        STAGE1_CONCEPTS,
    )

    similarity = (
        relation_centroids
        @ primitive_centroids.T
    )

    assignments = [
        PARENT_ONTOLOGY[
            concept
        ]
        for concept in STAGE2_CONCEPTS
    ]

    (
        parent_mean,
        nonparent_mean,
        margin,
        detail,
    ) = parent_margin_from_assignment(
        similarity,
        assignments,
    )

    return {
        "parent": parent_mean,
        "nonparent": nonparent_mean,
        "margin": margin,
        "concept": detail,
        "matrix": similarity,
    }


def parent_null(
    *,
    similarity: torch.Tensor,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    """
    Shuffle whole Stage2 parent sets across Stage2 concepts.

    This preserves the empirical distribution of ontology parent-set sizes
    and parent combinations.
    """

    assignments = [
        tuple(
            PARENT_ONTOLOGY[
                concept
            ]
        )
        for concept in STAGE2_CONCEPTS
    ]

    observed = parent_margin_from_assignment(
        similarity,
        assignments,
    )[2]

    rng = random.Random(
        seed
    )

    null_values: list[
        float
    ] = []

    for _ in range(
        permutations
    ):
        shuffled = (
            assignments[:]
        )

        rng.shuffle(
            shuffled
        )

        margin = (
            parent_margin_from_assignment(
                similarity,
                shuffled,
            )[2]
        )

        null_values.append(
            margin
        )

    return null_summary(
        observed,
        null_values,
    )


# =============================================================================
# Text Anchor Margin
# =============================================================================

def text_anchor_from_similarity(
    similarity: torch.Tensor,
) -> tuple[
    float,
    float,
    float,
    dict[
        str,
        tuple[
            float,
            float,
            float,
        ],
    ],
]:
    """
    For each image concept:
        matched = similarity to same text concept
        mismatched = mean similarity to all other text concepts
        margin = matched - mismatched

    Overall statistics are macro averages over Stage2 concepts.
    """

    n = len(
        STAGE2_CONCEPTS
    )

    if similarity.shape != (
        n,
        n,
    ):
        raise ValueError(
            "Text-anchor similarity matrix has unexpected shape: "
            f"{tuple(similarity.shape)}"
        )

    matched_values: list[
        float
    ] = []

    mismatched_values: list[
        float
    ] = []

    margins: list[
        float
    ] = []

    detail: dict[
        str,
        tuple[
            float,
            float,
            float,
        ],
    ] = {}

    for index, concept in enumerate(
        STAGE2_CONCEPTS
    ):
        matched = float(
            similarity[
                index,
                index,
            ]
        )

        other = torch.cat(
            [
                similarity[
                    index,
                    :index,
                ],
                similarity[
                    index,
                    index + 1:,
                ],
            ]
        )

        mismatched = float(
            other.mean()
        )

        margin = (
            matched
            - mismatched
        )

        matched_values.append(
            matched
        )

        mismatched_values.append(
            mismatched
        )

        margins.append(
            margin
        )

        detail[
            concept
        ] = (
            matched,
            mismatched,
            margin,
        )

    return (
        float(
            np.mean(
                matched_values
            )
        ),
        float(
            np.mean(
                mismatched_values
            )
        ),
        float(
            np.mean(
                margins
            )
        ),
        detail,
    )


def text_anchor_metric(
    *,
    image_vectors: torch.Tensor,
    text_vectors: torch.Tensor,
    labels: Sequence[str],
) -> dict[str, Any]:
    image_centroids = concept_centroids(
        image_vectors,
        labels,
        STAGE2_CONCEPTS,
    )

    text_centroids = concept_centroids(
        text_vectors,
        labels,
        STAGE2_CONCEPTS,
    )

    similarity = (
        image_centroids
        @ text_centroids.T
    )

    (
        matched,
        mismatched,
        margin,
        detail,
    ) = text_anchor_from_similarity(
        similarity
    )

    return {
        "matched": matched,
        "mismatched": mismatched,
        "margin": margin,
        "concept": detail,
        "matrix": similarity,
    }


def text_anchor_null(
    *,
    similarity: torch.Tensor,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    """
    Shuffle the one-to-one correspondence between image concepts
    and text concepts.
    """

    n = len(
        STAGE2_CONCEPTS
    )

    indices = torch.arange(
        n,
        dtype=torch.long,
    )

    observed = text_anchor_from_similarity(
        similarity
    )[2]

    generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    null_values: list[
        float
    ] = []

    for _ in range(
        permutations
    ):
        permutation = torch.randperm(
            n,
            generator=generator,
        )

        matched = similarity[
            indices,
            permutation,
        ]

        margins: list[
            float
        ] = []

        for row_index in range(
            n
        ):
            matched_column = int(
                permutation[
                    row_index
                ]
            )

            mask = torch.ones(
                n,
                dtype=torch.bool,
            )

            mask[
                matched_column
            ] = False

            mismatched = similarity[
                row_index,
                mask,
            ].mean()

            margins.append(
                float(
                    matched[
                        row_index
                    ]
                    - mismatched
                )
            )

        null_values.append(
            float(
                np.mean(
                    margins
                )
            )
        )

    return null_summary(
        observed,
        null_values,
    )


# =============================================================================
# Linear CKA
# =============================================================================

def linear_cka(
    image_vectors: torch.Tensor,
    text_vectors: torch.Tensor,
) -> float:
    """
    Exact sample-level linear CKA.

    X and Y contain the same Stage2 figures in the same order.

    Because n_samples < hidden_dim for this experiment, sample-space
    Gram matrices are smaller than feature-space cross-products.

    CUDA is used for this matrix calculation when available. The loaded
    model remains untouched.
    """

    if len(image_vectors) != len(
        text_vectors
    ):
        raise ValueError(
            "Image/text CKA sample counts do not match."
        )

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    x = image_vectors.to(
        device=device,
        dtype=torch.float32,
    )

    y = text_vectors.to(
        device=device,
        dtype=torch.float32,
    )

    x = (
        x
        - x.mean(
            dim=0,
            keepdim=True,
        )
    )

    y = (
        y
        - y.mean(
            dim=0,
            keepdim=True,
        )
    )

    image_gram = (
        x
        @ x.T
    )

    text_gram = (
        y
        @ y.T
    )

    numerator = (
        image_gram.double()
        .mul(
            text_gram.double()
        )
        .sum()
    )

    image_norm = (
        image_gram.double()
        .square()
        .sum()
    )

    text_norm = (
        text_gram.double()
        .square()
        .sum()
    )

    denominator = torch.sqrt(
        image_norm
        * text_norm
    )

    if float(
        denominator
    ) == 0.0:
        result = 0.0

    else:
        result = float(
            numerator
            / denominator
        )

    del x
    del y
    del image_gram
    del text_gram

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# =============================================================================
# One-model evaluation
# =============================================================================

def evaluate_one_model(
    *,
    model_kind: str,
    stage1_rows: Sequence[dict[str, Any]],
    stage2_rows: Sequence[dict[str, Any]],
    tokenizer,
    model,
    image_processor,
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, dict[int, Any]],
]:
    stage1_labels = [
        str(
            row[
                "concept"
            ]
        )
        for row in stage1_rows
    ]

    stage2_labels = [
        str(
            row[
                "concept"
            ]
        )
        for row in stage2_rows
    ]

    result: dict[
        str,
        Any,
    ] = {
        "stage1": {},
        "stage2": {},
        "parent": {},
        "text": {},
    }

    # =========================================================================
    # Stage1 visual representation
    # =========================================================================
    stage1_reps = extract_image_representations(
        rows=stage1_rows,
        description=f"{model_kind}/Stage1",
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        args=args,
    )

    for rep in REP_TYPES:
        result[
            "stage1"
        ][rep] = {}

        for layer in args.layers:
            structure = concept_structure(
                stage1_reps[
                    rep
                ][layer],
                stage1_labels,
                STAGE1_CONCEPTS,
            )

            if layer == args.primary_layer:
                structure[
                    "null"
                ] = separation_null(
                    vectors=stage1_reps[
                        rep
                    ][layer],
                    labels=stage1_labels,
                    pair_samples=args.null_pair_samples,
                    permutations=args.permutations,
                    seed=derived_seed(
                        args.seed,
                        "stage1",
                        "separation",
                        rep,
                        layer,
                    ),
                )

            result[
                "stage1"
            ][rep][layer] = structure

    rsa_signatures = build_stage1_rsa_signatures(
        representations=stage1_reps,
        labels=stage1_labels,
        layers=args.layers,
        primary_layer=args.primary_layer,
    )

    # =========================================================================
    # Stage2 visual representation
    # =========================================================================
    stage2_reps = extract_image_representations(
        rows=stage2_rows,
        description=f"{model_kind}/Stage2",
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        args=args,
    )

    # =========================================================================
    # Stage2 text-only representation
    # =========================================================================
    text_reps = extract_text_representations(
        rows=stage2_rows,
        description=f"{model_kind}/Stage2",
        tokenizer=tokenizer,
        model=model,
        args=args,
    )

    for rep in REP_TYPES:
        result[
            "stage2"
        ][rep] = {}

        result[
            "parent"
        ][rep] = {}

        result[
            "text"
        ][rep] = {}

        for layer in args.layers:
            # =================================================================
            # Stage2 concept structure
            # =================================================================
            structure = concept_structure(
                stage2_reps[
                    rep
                ][layer],
                stage2_labels,
                STAGE2_CONCEPTS,
            )

            if layer == args.primary_layer:
                structure[
                    "null"
                ] = separation_null(
                    vectors=stage2_reps[
                        rep
                    ][layer],
                    labels=stage2_labels,
                    pair_samples=args.null_pair_samples,
                    permutations=args.permutations,
                    seed=derived_seed(
                        args.seed,
                        "stage2",
                        "separation",
                        rep,
                        layer,
                    ),
                )

            result[
                "stage2"
            ][rep][layer] = structure

            # =================================================================
            # Parent Margin
            # =================================================================
            parent = parent_metric(
                stage2_vectors=stage2_reps[
                    rep
                ][layer],
                stage2_labels=stage2_labels,
                stage1_vectors=stage1_reps[
                    rep
                ][layer],
                stage1_labels=stage1_labels,
            )

            if layer == args.primary_layer:
                parent[
                    "null"
                ] = parent_null(
                    similarity=parent[
                        "matrix"
                    ],
                    permutations=args.permutations,
                    seed=derived_seed(
                        args.seed,
                        "parent",
                        rep,
                        layer,
                    ),
                )

            # Matrix is only required for the permutation test.
            parent.pop(
                "matrix"
            )

            result[
                "parent"
            ][rep][layer] = parent

            # =================================================================
            # Text Anchor Margin
            # =================================================================
            text = text_anchor_metric(
                image_vectors=stage2_reps[
                    rep
                ][layer],
                text_vectors=text_reps[
                    rep
                ][layer],
                labels=stage2_labels,
            )

            if layer == args.primary_layer:
                text[
                    "null"
                ] = text_anchor_null(
                    similarity=text[
                        "matrix"
                    ],
                    permutations=args.permutations,
                    seed=derived_seed(
                        args.seed,
                        "text-anchor",
                        rep,
                        layer,
                    ),
                )

                # CKA is the expensive sample-level structural metric.
                # It is therefore evaluated only at the pre-registered
                # primary analysis layer.
                text[
                    "cka"
                ] = linear_cka(
                    stage2_reps[
                        rep
                    ][layer],
                    text_reps[
                        rep
                    ][layer],
                )

            else:
                text[
                    "cka"
                ] = None

            text.pop(
                "matrix"
            )

            result[
                "text"
            ][rep][layer] = text

    del stage1_reps
    del stage2_reps
    del text_reps

    gc.collect()

    return (
        result,
        rsa_signatures,
    )


# =============================================================================
# Markdown helpers
# =============================================================================

def null_columns(
    metric: dict[str, Any],
) -> tuple[
    str,
    str,
    str,
    str,
]:
    null = metric.get(
        "null"
    )

    if null is None:
        return (
            "-",
            "-",
            "-",
            "-",
        )

    return (
        format_value(
            null[
                "observed"
            ]
        ),
        format_value(
            null[
                "mean"
            ]
        ),
        format_value(
            null[
                "z"
            ],
            2,
        ),
        format_value(
            null[
                "p"
            ],
            4,
        ),
    )


def append_structure_tables(
    *,
    lines: list[str],
    results: dict[str, Any],
    result_key: str,
    title: str,
    concepts: Sequence[str],
    args: argparse.Namespace,
) -> None:
    lines.extend(
        [
            f"## {title}",
            "",
            (
                "| Model | Layer | Rep | Intra | Inter | Separation | "
                "Null Obs* | Null Mean* | Z* | p* |"
            ),
            (
                "|---|---:|---|---:|---:|---:|"
                "---:|---:|---:|---:|"
            ),
        ]
    )

    for model_kind in MODEL_ORDER:
        if model_kind not in results:
            continue

        for rep in REP_TYPES:
            for layer in args.layers:
                metric = results[
                    model_kind
                ][result_key][rep][layer]

                (
                    null_observed,
                    null_mean,
                    z,
                    p,
                ) = null_columns(
                    metric
                )

                lines.append(
                    f"| {model_kind} "
                    f"| L{layer} "
                    f"| {rep} "
                    f"| {format_value(metric['intra'])} "
                    f"| {format_value(metric['inter'])} "
                    f"| {format_value(metric['separation'])} "
                    f"| {null_observed} "
                    f"| {null_mean} "
                    f"| {z} "
                    f"| {p} |"
                )

    lines.extend(
        [
            "",
            (
                f"### Per-concept — "
                f"L{args.primary_layer}"
            ),
            "",
            (
                "| Model | Rep | Concept | "
                "Intra | Inter | Separation | N |"
            ),
            (
                "|---|---|---|---:|---:|---:|---:|"
            ),
        ]
    )

    for model_kind in MODEL_ORDER:
        if model_kind not in results:
            continue

        for rep in REP_TYPES:
            metrics = results[
                model_kind
            ][result_key][rep][
                args.primary_layer
            ][
                "concept"
            ]

            for concept in concepts:
                item = metrics[
                    concept
                ]

                lines.append(
                    f"| {model_kind} "
                    f"| {rep} "
                    f"| {concept} "
                    f"| {format_value(item['intra'])} "
                    f"| {format_value(item['inter'])} "
                    f"| {format_value(item['separation'])} "
                    f"| {item['n']:,} |"
                )

    lines.append("")


# =============================================================================
# Markdown writer
# =============================================================================

def write_markdown(
    *,
    results: dict[str, Any],
    rsa_results: dict[str, Any],
    output_file: Path,
    stage1_n: int,
    stage2_n: int,
    complete: bool,
    args: argparse.Namespace,
) -> None:
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    layers_text = ", ".join(
        f"L{layer}"
        for layer in args.layers
    )

    lines: list[str] = [
        "# Geometry Representation Results",
        "",
        f"Status: **{'complete' if complete else 'partial'}**",
        "",
        "## Evaluation Configuration",
        "",
        f"- Stage1 samples: **{stage1_n:,}**",
        f"- Stage2 samples: **{stage2_n:,}**",
        f"- Layers: **{layers_text}**",
        f"- Primary analysis layer: **L{args.primary_layer}**",
        f"- Text field: `{args.text_column}`",
        f"- Permutations: **{args.permutations:,}**",
        (
            "- Concept-separation permutation pair sample: "
            f"**{args.null_pair_samples:,}**"
        ),
        "",
        "Representations:",
        "",
        "- `raw = h_normal`",
        "- `delta = h_normal - h_blank` for image input",
        "- `text_delta = h(description + prompt) - h(prompt)` for text input",
        "",
        (
            "`delta` is treated as an input-dependent hidden-state difference, "
            "not as mathematically pure visual information."
        ),
        "",
        "## Metric Definitions",
        "",
        (
            "- **Concept Separation = Intra - Inter**. "
            "Higher means same-concept samples are relatively more compact "
            "than different-concept samples."
        ),
        (
            "- **Checkpoint RSA**: Spearman correlation between the upper "
            "triangles of within-sample cosine-similarity matrices on the "
            "same Stage1 samples."
        ),
        (
            "- **Parent Margin**: macro-average similarity of each Stage2 "
            "concept to its registered Stage1 parents minus similarity to "
            "its non-parent Stage1 primitives."
        ),
        (
            "- **Text Anchor Margin**: similarity between matching visual/text "
            "concept centroids minus similarity to non-matching text concepts."
        ),
        (
            "- **Linear CKA**: exact sample-level linear CKA between Stage2 "
            "visual and text representations. CKA is computed only at the "
            f"primary layer L{args.primary_layer}."
        ),
        (
            "- **Permutation nulls** are computed only at the primary layer. "
            "They are empirical references, not theoretical chance levels."
        ),
        (
            "- Concept Separation's exact value and its permutation-test "
            "`Null Obs` may differ slightly because the permutation test uses "
            "the fixed pair sample described above."
        ),
        "",
        "## Parent Ontology",
        "",
        "| Stage2 Concept | Stage1 Parent Primitives |",
        "|---|---|",
    ]

    for concept in STAGE2_CONCEPTS:
        lines.append(
            f"| {concept} "
            f"| {', '.join(PARENT_ONTOLOGY[concept])} |"
        )

    lines.append("")

    # =========================================================================
    # Stage1 concept structure
    # =========================================================================
    append_structure_tables(
        lines=lines,
        results=results,
        result_key="stage1",
        title="Stage 1 — Primitive Concept Structure",
        concepts=STAGE1_CONCEPTS,
        args=args,
    )

    # =========================================================================
    # Stage1 RSA
    # =========================================================================
    lines.extend(
        [
            "## Stage 1 — Checkpoint RSA",
            "",
            "| Checkpoint Pair | Layer | Rep | RSA |",
            "|---|---:|---|---:|",
        ]
    )

    for pair_name, comparison in rsa_results.items():
        for (
            layer,
            rep,
            score,
        ) in comparison[
            "overall"
        ]:
            lines.append(
                f"| {pair_name} "
                f"| L{layer} "
                f"| {rep} "
                f"| {format_value(score)} |"
            )

    lines.extend(
        [
            "",
            (
                f"### Per-concept RSA — "
                f"L{args.primary_layer}"
            ),
            "",
            "| Checkpoint Pair | Rep | Concept | RSA |",
            "|---|---|---|---:|",
        ]
    )

    for pair_name, comparison in rsa_results.items():
        for (
            rep,
            concept,
            score,
        ) in comparison[
            "concept"
        ]:
            lines.append(
                f"| {pair_name} "
                f"| {rep} "
                f"| {concept} "
                f"| {format_value(score)} |"
            )

    lines.append("")

    # =========================================================================
    # Stage2 concept structure
    # =========================================================================
    append_structure_tables(
        lines=lines,
        results=results,
        result_key="stage2",
        title="Stage 2 — Relation Concept Structure",
        concepts=STAGE2_CONCEPTS,
        args=args,
    )

    # =========================================================================
    # Parent Margin
    # =========================================================================
    lines.extend(
        [
            "## Stage 2 — Primitive Parent Margin",
            "",
            (
                "| Model | Layer | Rep | Parent | Non-parent | Margin | "
                "Null Obs* | Null Mean* | Z* | p* |"
            ),
            (
                "|---|---:|---|---:|---:|---:|"
                "---:|---:|---:|---:|"
            ),
        ]
    )

    for model_kind in MODEL_ORDER:
        if model_kind not in results:
            continue

        for rep in REP_TYPES:
            for layer in args.layers:
                metric = results[
                    model_kind
                ][
                    "parent"
                ][rep][layer]

                (
                    null_observed,
                    null_mean,
                    z,
                    p,
                ) = null_columns(
                    metric
                )

                lines.append(
                    f"| {model_kind} "
                    f"| L{layer} "
                    f"| {rep} "
                    f"| {format_value(metric['parent'])} "
                    f"| {format_value(metric['nonparent'])} "
                    f"| {format_value(metric['margin'])} "
                    f"| {null_observed} "
                    f"| {null_mean} "
                    f"| {z} "
                    f"| {p} |"
                )

    lines.extend(
        [
            "",
            (
                f"### Per-concept Parent Margin — "
                f"L{args.primary_layer}"
            ),
            "",
            (
                "| Model | Rep | Concept | "
                "Parents | Parent | Non-parent | Margin |"
            ),
            (
                "|---|---|---|---|---:|---:|---:|"
            ),
        ]
    )

    for model_kind in MODEL_ORDER:
        if model_kind not in results:
            continue

        for rep in REP_TYPES:
            detail = results[
                model_kind
            ][
                "parent"
            ][rep][
                args.primary_layer
            ][
                "concept"
            ]

            for concept in STAGE2_CONCEPTS:
                (
                    parent,
                    nonparent,
                    margin,
                ) = detail[
                    concept
                ]

                lines.append(
                    f"| {model_kind} "
                    f"| {rep} "
                    f"| {concept} "
                    f"| {', '.join(PARENT_ONTOLOGY[concept])} "
                    f"| {format_value(parent)} "
                    f"| {format_value(nonparent)} "
                    f"| {format_value(margin)} |"
                )

    lines.append("")

    # =========================================================================
    # Text anchoring
    # =========================================================================
    lines.extend(
        [
            "## Stage 2 — Text Anchoring",
            "",
            (
                "| Model | Layer | Rep | Matched | Mismatched | Margin | "
                "Linear CKA | Null Obs* | Null Mean* | Z* | p* |"
            ),
            (
                "|---|---:|---|---:|---:|---:|---:|"
                "---:|---:|---:|---:|"
            ),
        ]
    )

    for model_kind in MODEL_ORDER:
        if model_kind not in results:
            continue

        for rep in REP_TYPES:
            for layer in args.layers:
                metric = results[
                    model_kind
                ][
                    "text"
                ][rep][layer]

                (
                    null_observed,
                    null_mean,
                    z,
                    p,
                ) = null_columns(
                    metric
                )

                lines.append(
                    f"| {model_kind} "
                    f"| L{layer} "
                    f"| {rep} "
                    f"| {format_value(metric['matched'])} "
                    f"| {format_value(metric['mismatched'])} "
                    f"| {format_value(metric['margin'])} "
                    f"| {format_value(metric['cka'])} "
                    f"| {null_observed} "
                    f"| {null_mean} "
                    f"| {z} "
                    f"| {p} |"
                )

    lines.extend(
        [
            "",
            (
                f"### Per-concept Text Anchor Margin — "
                f"L{args.primary_layer}"
            ),
            "",
            (
                "| Model | Rep | Concept | "
                "Matched | Mismatched | Margin |"
            ),
            (
                "|---|---|---|---:|---:|---:|"
            ),
        ]
    )

    for model_kind in MODEL_ORDER:
        if model_kind not in results:
            continue

        for rep in REP_TYPES:
            detail = results[
                model_kind
            ][
                "text"
            ][rep][
                args.primary_layer
            ][
                "concept"
            ]

            for concept in STAGE2_CONCEPTS:
                (
                    matched,
                    mismatched,
                    margin,
                ) = detail[
                    concept
                ]

                lines.append(
                    f"| {model_kind} "
                    f"| {rep} "
                    f"| {concept} "
                    f"| {format_value(matched)} "
                    f"| {format_value(mismatched)} "
                    f"| {format_value(margin)} |"
                )

    lines.append("")

    output_file.write_text(
        "\n".join(
            lines
        ).rstrip()
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

    model_paths = EvaluationModelPaths.from_values(
        base_model=args.base_model,
        stage1_dir=args.stage1_dir,
        stage2_dir=args.stage2_dir,
        stage2_only_dir=args.stage2_only_dir,
    )

    print(
        f"Representation result: "
        f"{output_file}"
    )

    # Validate/load datasets before loading a 7B checkpoint.
    stage1_rows = prepare_stage1_rows(
        args
    )

    stage2_rows = prepare_stage2_rows(
        args
    )

    results: dict[
        str,
        Any,
    ] = {}

    rsa_signatures: dict[
        str,
        dict[str, dict[int, Any]],
    ] = {}

    rsa_results: dict[
        str,
        Any,
    ] = {}

    write_markdown(
        results=results,
        rsa_results=rsa_results,
        output_file=output_file,
        stage1_n=len(stage1_rows),
        stage2_n=len(stage2_rows),
        complete=False,
        args=args,
    )

    # =========================================================================
    # One model at a time
    # =========================================================================

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

            (
                results[
                    model_kind
                ],
                rsa_signatures[
                    model_kind
                ],
            ) = evaluate_one_model(
                model_kind=model_kind,
                stage1_rows=stage1_rows,
                stage2_rows=stage2_rows,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                args=args,
            )

            # Compare this checkpoint to every earlier completed checkpoint.
            for earlier in MODEL_ORDER:
                if earlier == model_kind:
                    break

                if earlier not in rsa_signatures:
                    continue

                pair_name = (
                    f"{earlier} <-> {model_kind}"
                )

                rsa_results[
                    pair_name
                ] = compare_rsa_signatures(
                    rsa_signatures[
                        earlier
                    ],
                    rsa_signatures[
                        model_kind
                    ],
                    layers=args.layers,
                    primary_layer=args.primary_layer,
                )

            write_markdown(
                results=results,
                rsa_results=rsa_results,
                output_file=output_file,
                stage1_n=len(stage1_rows),
                stage2_n=len(stage2_rows),
                complete=False,
                args=args,
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
        rsa_results=rsa_results,
        output_file=output_file,
        stage1_n=len(stage1_rows),
        stage2_n=len(stage2_rows),
        complete=True,
        args=args,
    )

    print(
        "\nRepresentation evaluation completed."
    )

    print(
        f"Final result: "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()