'''
outputs/data/stage_geometry_grounding/
├── train_parquet/
│   ├── stage1_geometry_grounding_train_000.parquet
│   └── stage1_geometry_grounding_train_001.parquet
└── validation_parquet/
    └── stage1_geometry_grounding_validation_000.parquet

->

geometry_data/stage/
├── train.json
├── validation.json
└── images/
    ├── train/
    └── validation/
'''

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """명령행 인자를 읽는다."""

    parser = argparse.ArgumentParser(
        description="Convert geometry Parquet data to LLaVA JSON/PNG format."
    )

    parser.add_argument(
        "--input_root",
        type=Path,
        required=True,
        help="Root directory containing split Parquet directories.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for converted images and JSON files.",
    )

    # 새로 추가되는 세 번째 인자다.
    # 위의 --input_root 또는 --output_dir를 대체하는 것이 아니다.
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation"),
        default=("train", "validation"),
        help="Dataset splits to convert.",
    )

    return parser.parse_args()


def load_records(parquet_dir: Path) -> list[dict]:
    """한 split 폴더 안의 모든 Parquet 파일을 읽는다."""

    if not parquet_dir.is_dir():
        raise FileNotFoundError(
            f"Parquet directory was not found: {parquet_dir}"
        )

    parquet_paths = sorted(parquet_dir.glob("*.parquet"))

    if not parquet_paths:
        raise FileNotFoundError(
            f"No Parquet files were found in: {parquet_dir}"
        )

    records: list[dict] = []

    for parquet_path in parquet_paths:
        print(f"Reading: {parquet_path}")
        table = pq.read_table(parquet_path)
        records.extend(table.to_pylist())

    return records


def validate_record(
    record: dict,
    split: str,
    seen_ids: set[str],
) -> None:
    """변환에 필요한 필드와 sample ID를 검사한다."""

    required_fields = {
        "id",
        "image",
        "conversations",
    }

    missing_fields = required_fields - set(record)

    if missing_fields:
        raise ValueError(
            f"Missing required fields: {sorted(missing_fields)}"
        )

    sample_id = str(record["id"])

    if sample_id in seen_ids:
        raise ValueError(f"Duplicate sample id: {sample_id}")

    seen_ids.add(sample_id)

    record_split = record.get("split")

    if record_split is not None and record_split != split:
        raise ValueError(
            f"Split mismatch for sample {sample_id}: "
            f"expected={split}, actual={record_split}"
        )

    conversations = record["conversations"]

    if not isinstance(conversations, list) or not conversations:
        raise ValueError(
            f"Invalid conversations for sample: {sample_id}"
        )

    first_message = conversations[0]

    if "<image>" not in str(first_message.get("value", "")):
        raise ValueError(
            f"The first conversation does not contain <image>: {sample_id}"
        )


def validate_stage2_pairs(
    records: list[dict],
    split: str,
) -> None:
    """Stage 2 local-anchor pair 구성이 정확한지 검사한다."""

    # Stage 1에는 task_kind 필드가 없으므로 검사하지 않는다.
    if not any("task_kind" in record for record in records):
        return

    pair_groups: dict[
        tuple[str, str, int],
        list[dict],
    ] = defaultdict(list)

    for record in records:
        sample_id = str(record.get("id", ""))
        task_kind = record.get("task_kind")

        if task_kind not in {"local", "anchor"}:
            raise ValueError(
                f"Invalid task_kind for {sample_id}: {task_kind}"
            )

        image_hash = record.get("image_sha256")
        code = record.get("code")
        seed = record.get("seed")

        if not image_hash:
            raise ValueError(
                f"Missing image_sha256 for Stage 2 sample: {sample_id}"
            )

        pair_key = (
            str(image_hash),
            str(code),
            int(seed),
        )
        pair_groups[pair_key].append(record)

    invalid_pairs: list[str] = []

    for pair_key, pair_records in pair_groups.items():
        task_kinds = {
            record.get("task_kind")
            for record in pair_records
        }
        concepts = {
            record.get("concept")
            for record in pair_records
        }

        if len(pair_records) != 2:
            invalid_pairs.append(
                f"{pair_key[0]}: expected 2 rows, "
                f"found {len(pair_records)}"
            )
            continue

        if task_kinds != {"local", "anchor"}:
            invalid_pairs.append(
                f"{pair_key[0]}: task kinds={sorted(task_kinds)}"
            )
            continue

        if len(concepts) != 1:
            invalid_pairs.append(
                f"{pair_key[0]}: concepts={sorted(concepts)}"
            )

    if invalid_pairs:
        preview = "\n".join(
            f"  - {message}"
            for message in invalid_pairs[:20]
        )
        raise ValueError(
            f"Invalid Stage 2 pairs in {split}:\n{preview}"
        )

    print(
        f"Validated {len(pair_groups)} complete "
        f"local-anchor pairs for {split}."
    )

def convert_split(
    split: str,
    parquet_dir: Path,
    output_dir: Path,
) -> None:
    """한 split의 Parquet을 LLaVA JSON과 PNG로 변환한다."""

    records = load_records(parquet_dir)

    # Stage 2이면 local-anchor pair 구성을 검사한다.
    validate_stage2_pairs(
        records=records,
        split=split,
    )

    image_dir = output_dir / "images" / split

    # 이전 변환 결과가 남지 않도록 해당 split 이미지를 제거한다.
    if image_dir.exists():
        shutil.rmtree(image_dir)

    image_dir.mkdir(parents=True, exist_ok=True)

    llava_records: list[dict] = []
    seen_ids: set[str] = set()

    concept_counts: Counter[str] = Counter()
    task_kind_counts: Counter[str] = Counter()

    written_image_hashes: set[str] = set()

    for record in tqdm(records, desc=f"Converting {split}"):
        validate_record(
            record=record,
            split=split,
            seen_ids=seen_ids,
        )

        sample_id = str(record["id"])
        image_hash = str(
            record.get("image_sha256") or sample_id
        )

        # local과 anchor가 같은 그림을 사용하므로
        # image hash를 파일명으로 사용하여 PNG를 한 번만 저장한다.
        image_name = f"{image_hash}.png"
        image_path = image_dir / image_name

        if image_hash not in written_image_hashes:
            image_path.write_bytes(record["image"])
            written_image_hashes.add(image_hash)

        relative_image_path = Path(split) / image_name

        llava_record = {
            "id": sample_id,
            "image": relative_image_path.as_posix(),
            "conversations": record["conversations"],
        }

        # 학습에는 직접 사용하지 않지만
        # 데이터 감사와 local/anchor 분석을 위해 보존한다.
        if "task_kind" in record:
            llava_record["task_kind"] = record["task_kind"]

        if "concept" in record:
            llava_record["concept"] = record["concept"]

        if "image_sha256" in record:
            llava_record["image_sha256"] = record["image_sha256"]

        llava_records.append(llava_record)

        concept = str(record.get("concept", "unknown"))
        concept_counts[concept] += 1

        task_kind = record.get("task_kind")
        if task_kind is not None:
            task_kind_counts[str(task_kind)] += 1

    json_path = output_dir / f"{split}.json"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            llava_records,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"Converted rows: {len(llava_records)}")
    print(f"Unique images: {len(written_image_hashes)}")
    print(
        "Concept counts: "
        f"{dict(sorted(concept_counts.items()))}"
    )

    if task_kind_counts:
        print(
            "Task-kind counts: "
            f"{dict(sorted(task_kind_counts.items()))}"
        )

    print(f"JSON saved to: {json_path}")
    print(f"Images saved to: {image_dir}")


def remove_unused_validation_output(output_dir: Path) -> None:
    """train-only 변환 시 과거 validation 결과를 삭제한다."""

    validation_json = output_dir / "validation.json"
    validation_image_dir = output_dir / "images" / "validation"

    if validation_json.exists():
        validation_json.unlink()
        print(f"Removed old file: {validation_json}")

    if validation_image_dir.exists():
        shutil.rmtree(validation_image_dir)
        print(f"Removed old directory: {validation_image_dir}")


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1에서 --splits train으로 실행하면
    # 이전 validation 변환 결과를 제거한다.
    if "validation" not in args.splits:
        remove_unused_validation_output(args.output_dir)

    for split in args.splits:
        convert_split(
            split=split,
            parquet_dir=args.input_root / f"{split}_parquet",
            output_dir=args.output_dir,
        )

    print("Dataset conversion completed.")


if __name__ == "__main__":
    main()