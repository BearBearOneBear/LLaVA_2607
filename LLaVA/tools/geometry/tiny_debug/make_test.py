from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = "./debug_data"
DEFAULT_NUM_SAMPLES = 100
DEFAULT_SEED = 20260806


def parse_args() -> argparse.Namespace:
    """명령행 인자를 읽는다."""

    parser = argparse.ArgumentParser(
        description=(
            "Create balanced tiny datasets for Stage 1 and Stage 2. "
            "Stage 2 local-anchor pairs are kept together."
        )
    )

    parser.add_argument(
        "--stage1_train_path",
        type=Path,
        required=True,
        help="Path to the Stage 1 training JSON.",
    )

    parser.add_argument(
        "--stage2_train_path",
        type=Path,
        required=True,
        help="Path to the Stage 2 training JSON.",
    )

    parser.add_argument(
        "--stage2_eval_path",
        type=Path,
        required=True,
        help="Path to the Stage 2 validation JSON.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Directory for generated tiny datasets.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=(
            "Maximum number of rows to keep from each dataset. "
            "A Stage 2 pair is never split, so an odd limit may "
            "produce one fewer row."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for balanced subset sampling.",
    )

    return parser.parse_args()


def load_json(input_path: Path) -> list[dict[str, Any]]:
    """JSON list 형식의 데이터셋을 읽는다."""

    resolved_path = input_path.resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Dataset was not found: {resolved_path}"
        )

    with resolved_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError(
            f"Dataset must be a JSON list: {resolved_path}"
        )

    if not dataset:
        raise ValueError(
            f"Dataset is empty: {resolved_path}"
        )

    for index, record in enumerate(dataset):
        if not isinstance(record, dict):
            raise ValueError(
                "Every dataset row must be a JSON object. "
                f"Invalid row index: {index}"
            )

    return dataset


def build_record_groups(
    dataset: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """동일 이미지를 사용하는 행들을 하나의 그룹으로 묶는다.

    Stage 1:
      이미지마다 행이 하나이므로 사실상 한 행이 한 그룹이다.

    Stage 2:
      같은 image_sha256을 가진 local/anchor 두 행이 한 그룹이다.
    """

    groups: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for index, record in enumerate(dataset):
        image_hash = record.get("image_sha256")

        if image_hash:
            group_key = f"image:{image_hash}"
        else:
            group_key = f"single-row:{index}"

        groups[group_key].append(record)

    for group_key, group in groups.items():
        concepts = {
            str(record.get("concept", "unknown"))
            for record in group
        }

        if len(concepts) != 1:
            raise ValueError(
                "Records sharing one image have different concepts: "
                f"{group_key}, concepts={sorted(concepts)}"
            )

        task_kinds = {
            str(record["task_kind"])
            for record in group
            if record.get("task_kind") is not None
        }

        # task_kind가 존재하면 Stage 2 pair로 간주한다.
        if task_kinds:
            if len(group) != 2:
                raise ValueError(
                    "Stage 2 image group must contain exactly "
                    f"two rows: {group_key}, rows={len(group)}"
                )

            if task_kinds != {"local", "anchor"}:
                raise ValueError(
                    "Stage 2 image group must contain one local "
                    f"and one anchor row: {group_key}, "
                    f"task_kinds={sorted(task_kinds)}"
                )

    return dict(groups)


def sample_dataset(
    dataset: list[dict[str, Any]],
    num_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """concept을 고르게 포함하면서 그룹 단위로 추출한다.

    각 concept의 이미지 그룹을 무작위로 섞은 뒤
    concept별 round-robin 방식으로 선택한다.

    Stage 2 local-anchor pair는 항상 함께 선택된다.
    """

    rng = random.Random(seed)

    record_groups = build_record_groups(dataset)

    concept_group_keys: dict[str, list[str]] = defaultdict(list)

    for group_key, group in record_groups.items():
        concept = str(group[0].get("concept", "unknown"))
        concept_group_keys[concept].append(group_key)

    concepts = sorted(concept_group_keys)
    rng.shuffle(concepts)

    for group_keys in concept_group_keys.values():
        rng.shuffle(group_keys)

    subset: list[dict[str, Any]] = []

    while True:
        added_any_group = False

        for concept in concepts:
            group_keys = concept_group_keys[concept]

            while group_keys:
                group_key = group_keys.pop()
                group = record_groups[group_key]

                # 마지막 pair를 반으로 자르지 않는다.
                if len(subset) + len(group) > num_samples:
                    continue

                subset.extend(group)
                added_any_group = True
                break

            if len(subset) == num_samples:
                return subset

        if not added_any_group:
            break

    return subset


def print_subset_statistics(
    subset: list[dict[str, Any]],
) -> None:
    """생성된 tiny subset의 concept/task 분포를 출력한다."""

    concept_counts = Counter(
        str(record.get("concept", "unknown"))
        for record in subset
    )

    task_kind_counts = Counter(
        str(record["task_kind"])
        for record in subset
        if record.get("task_kind") is not None
    )

    unique_images = {
        str(record.get("image_sha256") or record.get("image"))
        for record in subset
    }

    print(f"Rows: {len(subset)}")
    print(f"Unique images: {len(unique_images)}")
    print(
        "Concept counts: "
        f"{dict(sorted(concept_counts.items()))}"
    )

    if task_kind_counts:
        print(
            "Task-kind counts: "
            f"{dict(sorted(task_kind_counts.items()))}"
        )


def save_subset(
    input_path: Path,
    output_path: Path,
    num_samples: int,
    seed: int,
) -> int:
    """입력 데이터의 균형 잡힌 부분집합을 저장한다."""

    dataset = load_json(input_path)

    subset = sample_dataset(
        dataset=dataset,
        num_samples=num_samples,
        seed=seed,
    )

    if not subset:
        raise ValueError(
            "No records could be selected from: "
            f"{input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            subset,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"Source: {input_path.resolve()}")
    print(f"Output: {output_path.resolve()}")
    print(f"Requested rows: {num_samples}")
    print_subset_statistics(subset)

    return len(subset)


def main() -> None:
    args = parse_args()

    if args.num_samples <= 0:
        raise ValueError(
            "--num_samples must be greater than zero."
        )

    output_dir = args.output_dir.resolve()

    stage1_output_dir = output_dir / "stage1"
    stage2_output_dir = output_dir / "stage2"

    stage1_output_path = (
        stage1_output_dir
        / f"train_{args.num_samples}.json"
    )

    stage2_train_output_path = (
        stage2_output_dir
        / f"train_{args.num_samples}.json"
    )

    stage2_eval_output_path = (
        stage2_output_dir
        / f"validation_{args.num_samples}.json"
    )

    print("Creating tiny Stage 1 and Stage 2 datasets.")
    print(f"Maximum rows per dataset: {args.num_samples}")
    print(f"Sampling seed: {args.seed}")
    print(f"Output directory: {output_dir}")

    # Stage 1 train-only
    save_subset(
        input_path=args.stage1_train_path,
        output_path=stage1_output_path,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    # Stage 2 train
    save_subset(
        input_path=args.stage2_train_path,
        output_path=stage2_train_output_path,
        num_samples=args.num_samples,
        seed=args.seed + 1,
    )

    # Stage 2 validation
    save_subset(
        input_path=args.stage2_eval_path,
        output_path=stage2_eval_output_path,
        num_samples=args.num_samples,
        seed=args.seed + 2,
    )

    print()
    print("Tiny dataset creation completed.")
    print(f"Stage 1 train: {stage1_output_path}")
    print(f"Stage 2 train: {stage2_train_output_path}")
    print(f"Stage 2 validation: {stage2_eval_output_path}")


if __name__ == "__main__":
    main()