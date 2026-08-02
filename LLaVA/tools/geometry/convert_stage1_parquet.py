'''
outputs/data/stage1_geometry_grounding/
├── train_parquet/
│   ├── stage1_geometry_grounding_train_000.parquet
│   └── stage1_geometry_grounding_train_001.parquet
└── validation_parquet/
    └── stage1_geometry_grounding_validation_000.parquet

->

geometry_data/stage1/
├── train.json
├── validation.json
└── images/
    ├── train/
    └── validation/
'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """path"""

    parser = argparse.ArgumentParser(
        description="Convert Stage 1 Parquet data to LLaVA format."
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        required=True,
        help="Root directory containing train_parquet and validation_parquet.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for converted images and JSON files.",
    )
    return parser.parse_args()


def load_records(parquet_dir: Path) -> list[dict]:
    """parquets to records"""

    records: list[dict] = []

    for parquet_path in sorted(parquet_dir.glob("*.parquet")):
        table = pq.read_table(parquet_path)
        records.extend(table.to_pylist())

    return records


def convert_split(
    split: str,
    parquet_dir: Path,
    output_dir: Path,
) -> None:
    """llava format split"""

    records = load_records(parquet_dir)

    image_dir = output_dir / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)

    llava_records: list[dict] = []

    for record in tqdm(records, desc=f"Converting {split}"):
        sample_id = record["id"]
        image_name = f"{sample_id}.png"

        image_path = image_dir / image_name
        image_path.write_bytes(record["image"])

        relative_image_path = Path(split) / image_name

        llava_records.append(
            {
                "id": sample_id,
                "image": relative_image_path.as_posix(),
                "conversations": record["conversations"],
            }
        )

    json_path = output_dir / f"{split}.json"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            llava_records,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Converted {len(llava_records)} {split} samples.")
    print(f"JSON saved to: {json_path}")
    print(f"Images saved to: {image_dir}")


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    convert_split(
        split="train",
        parquet_dir=args.input_root / "train_parquet",
        output_dir=args.output_dir,
    )

    convert_split(
        split="validation",
        parquet_dir=args.input_root / "validation_parquet",
        output_dir=args.output_dir,
    )

    print("Stage 1 dataset conversion completed.")


if __name__ == "__main__":
    main()