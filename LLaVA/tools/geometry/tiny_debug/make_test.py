from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = "./debug_data"
DEFAULT_NUM_SAMPLES = 100


def parse_args() -> argparse.Namespace:
    """명령행 인자를 읽는다."""

    parser = argparse.ArgumentParser(
        description=(
            "Create small Stage 1 train-only and "
            "Stage 2 train/validation datasets."
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
        help="Directory for the tiny datasets.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of samples to keep from each dataset.",
    )

    return parser.parse_args()


def load_json(input_path: Path) -> list[dict[str, Any]]:
    """JSON list 형식의 데이터셋을 읽는다."""

    input_path = input_path.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Dataset was not found: {input_path}"
        )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError(
            f"Dataset must be a JSON list: {input_path}"
        )

    return dataset


def save_subset(
    input_path: Path,
    output_path: Path,
    num_samples: int,
) -> int:
    """입력 데이터의 앞부분을 tiny JSON으로 저장한다."""

    dataset = load_json(input_path)

    subset = dataset[:num_samples]

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

    print(
        f"Created {output_path}: "
        f"{len(subset)} of {len(dataset)} samples."
    )

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

    print("Creating tiny Stage 1 and Stage 2 datasets.")
    print(f"Number of samples: {args.num_samples}")
    print(f"Output directory: {output_dir}")

    # Stage 1은 train-only다.
    save_subset(
        input_path=args.stage1_train_path,
        output_path=stage1_output_dir / "train_100.json",
        num_samples=args.num_samples,
    )

    # Stage 2는 train과 validation을 모두 사용한다.
    save_subset(
        input_path=args.stage2_train_path,
        output_path=stage2_output_dir / "train_100.json",
        num_samples=args.num_samples,
    )

    save_subset(
        input_path=args.stage2_eval_path,
        output_path=stage2_output_dir / "validation_100.json",
        num_samples=args.num_samples,
    )

    print("Tiny dataset creation completed.")


if __name__ == "__main__":
    main()