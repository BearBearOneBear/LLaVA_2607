"""
Convert geometry Parquet datasets to the LLaVA JSON/PNG format.

Expected input:

input_root/
├── train_parquet/
│   ├── *.parquet
│   └── ...
└── validation_parquet/
    ├── *.parquet
    └── ...

Generated output:

output_dir/
├── train.json
├── validation.json
└── images/
    ├── train/
    └── validation/

Stage 1 is normally converted with:

    --splits train

Stage 2 is normally converted with:

    --splits train validation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
from tqdm import tqdm


ALLOWED_REMOVABLE_PATHS = {
    Path("images/train"),
    Path("images/validation"),
    Path("validation.json"),
}

METADATA_COLUMNS = [
    "id",
    "split",
    "task_kind",
    "concept",
    "code",
    "seed",
    "image_sha256",
]


def parse_args() -> argparse.Namespace:
    """명령행 인자를 읽는다."""

    parser = argparse.ArgumentParser(
        description=(
            "Convert geometry Parquet data to "
            "the LLaVA JSON/PNG format."
        )
    )

    parser.add_argument(
        "--input_root",
        type=Path,
        required=True,
        help=(
            "Root directory containing train_parquet and/or "
            "validation_parquet."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for converted JSON and PNG files.",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation"),
        default=("train", "validation"),
        help="Dataset splits to convert.",
    )

    return parser.parse_args()


def validate_output_root(output_dir: Path) -> Path:
    """위험한 위치를 output root로 사용하는 것을 방지한다."""

    resolved_output = output_dir.resolve()
    filesystem_root = Path(resolved_output.anchor).resolve()
    home_directory = Path.home().resolve()

    if resolved_output == filesystem_root:
        raise ValueError(
            f"Refusing to use the filesystem root: {resolved_output}"
        )

    if resolved_output == home_directory:
        raise ValueError(
            f"Refusing to use the home directory: {resolved_output}"
        )

    return resolved_output


def validate_removable_path(
    target: Path,
    root: Path,
) -> Path:
    """변환기가 생성하는 것으로 확정된 경로만 허용한다."""

    resolved_target = target.resolve()
    resolved_root = root.resolve()

    try:
        relative_target = resolved_target.relative_to(
            resolved_root
        )
    except ValueError as error:
        raise ValueError(
            "Refusing to remove a path outside output_dir: "
            f"{resolved_target}"
        ) from error

    if relative_target not in ALLOWED_REMOVABLE_PATHS:
        raise ValueError(
            "Refusing to remove an unexpected generated path: "
            f"{resolved_target}"
        )

    return resolved_target


def safe_remove_directory(
    target: Path,
    root: Path,
) -> None:
    """허용된 생성 디렉터리만 삭제한다."""

    if not target.exists():
        return

    resolved_target = validate_removable_path(
        target=target,
        root=root,
    )

    if not resolved_target.is_dir():
        raise ValueError(
            f"Expected a directory: {resolved_target}"
        )

    shutil.rmtree(resolved_target)

    print(f"Removed old directory: {resolved_target}")


def safe_remove_file(
    target: Path,
    root: Path,
) -> None:
    """허용된 생성 파일만 삭제한다."""

    if not target.exists():
        return

    resolved_target = validate_removable_path(
        target=target,
        root=root,
    )

    if not resolved_target.is_file():
        raise ValueError(
            f"Expected a file: {resolved_target}"
        )

    resolved_target.unlink()

    print(f"Removed old file: {resolved_target}")


def list_parquet_paths(
    parquet_dir: Path,
) -> list[Path]:
    """한 split 폴더의 Parquet 파일을 정렬해 반환한다."""

    if not parquet_dir.is_dir():
        raise FileNotFoundError(
            f"Parquet directory was not found: {parquet_dir}"
        )

    parquet_paths = sorted(
        parquet_dir.glob("*.parquet")
    )

    if not parquet_paths:
        raise FileNotFoundError(
            f"No Parquet files were found in: {parquet_dir}"
        )

    return parquet_paths


def iter_records(
    parquet_paths: list[Path],
    columns: list[str] | None = None,
) -> Iterator[dict]:
    """Parquet을 batch 단위로 읽어 record를 순차 반환한다.

    columns=None이면 이미지가 포함된 전체 컬럼을 읽는다.

    metadata 검증 단계에서는 columns를 지정해 이미지 bytes를
    메모리에 올리지 않는다.
    """

    for parquet_path in parquet_paths:
        print(f"Reading: {parquet_path}")

        parquet_file = pq.ParquetFile(
            parquet_path
        )

        available_columns = set(
            parquet_file.schema_arrow.names
        )

        selected_columns = None

        if columns is not None:
            selected_columns = [
                column
                for column in columns
                if column in available_columns
            ]

        for batch in parquet_file.iter_batches(
            batch_size=256,
            columns=selected_columns,
        ):
            yield from batch.to_pylist()


def count_records(
    parquet_paths: list[Path],
) -> int:
    """여러 Parquet 파일의 전체 행 수를 센다."""

    return sum(
        pq.ParquetFile(path).metadata.num_rows
        for path in parquet_paths
    )


def validate_conversations(
    conversations: object,
    sample_id: str,
) -> None:
    """LLaVA conversation 형식을 검사한다."""

    if not isinstance(conversations, list):
        raise ValueError(
            f"Conversations must be a list: {sample_id}"
        )

    if not conversations:
        raise ValueError(
            f"Conversations are empty: {sample_id}"
        )

    first_message = conversations[0]

    if not isinstance(first_message, dict):
        raise ValueError(
            "The first conversation message must be an object: "
            f"{sample_id}"
        )

    first_value = str(
        first_message.get("value", "")
    )

    if "<image>" not in first_value:
        raise ValueError(
            "The first conversation does not contain <image>: "
            f"{sample_id}"
        )


def validate_record(
    record: dict,
    split: str,
    seen_ids: set[str],
) -> str:
    """필수 필드와 PNG를 검사하고 실제 이미지 hash를 반환한다."""

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

    if not sample_id:
        raise ValueError("A sample has an empty id.")

    if sample_id in seen_ids:
        raise ValueError(
            f"Duplicate sample id: {sample_id}"
        )

    image_bytes = record["image"]

    if not isinstance(
        image_bytes,
        (bytes, bytearray),
    ):
        raise TypeError(
            f"The image column must be raw bytes for {sample_id}. "
            f"Received: {type(image_bytes).__name__}"
        )

    if not image_bytes.startswith(
        b"\x89PNG\r\n\x1a\n"
    ):
        raise ValueError(
            f"The image is not a PNG: {sample_id}"
        )

    actual_image_hash = hashlib.sha256(
        image_bytes
    ).hexdigest()

    declared_image_hash = record.get(
        "image_sha256"
    )

    if declared_image_hash is not None:
        declared_image_hash = str(
            declared_image_hash
        )

        if declared_image_hash != actual_image_hash:
            raise ValueError(
                "image_sha256 mismatch for "
                f"{sample_id}: "
                f"declared={declared_image_hash}, "
                f"actual={actual_image_hash}"
            )

    record_split = record.get("split")

    if (
        record_split is not None
        and str(record_split) != split
    ):
        raise ValueError(
            f"Split mismatch for {sample_id}: "
            f"expected={split}, actual={record_split}"
        )

    validate_conversations(
        conversations=record["conversations"],
        sample_id=sample_id,
    )

    seen_ids.add(sample_id)

    return actual_image_hash


def validate_stage2_pairs(
    parquet_paths: list[Path],
    split: str,
) -> None:
    """Stage 2 local-anchor pair 구성을 검사한다.

    metadata 컬럼만 읽으며 이미지 bytes는 읽지 않는다.
    """

    records = list(
        iter_records(
            parquet_paths=parquet_paths,
            columns=METADATA_COLUMNS,
        )
    )

    has_task_kind = any(
        record.get("task_kind") is not None
        for record in records
    )

    # Stage 1에는 task_kind가 없으므로 pair 검사를 생략한다.
    if not has_task_kind:
        return

    pair_groups: dict[
        tuple[str, str, int],
        list[dict],
    ] = defaultdict(list)

    for record in records:
        sample_id = str(
            record.get("id", "")
        )

        task_kind = record.get(
            "task_kind"
        )

        if task_kind not in {
            "local",
            "anchor",
        }:
            raise ValueError(
                f"Invalid task_kind for {sample_id}: "
                f"{task_kind}"
            )

        image_hash = record.get(
            "image_sha256"
        )

        code = record.get("code")
        seed = record.get("seed")

        if not image_hash:
            raise ValueError(
                "Missing image_sha256 for Stage 2 sample: "
                f"{sample_id}"
            )

        if code is None:
            raise ValueError(
                "Missing code for Stage 2 sample: "
                f"{sample_id}"
            )

        if seed is None:
            raise ValueError(
                "Missing seed for Stage 2 sample: "
                f"{sample_id}"
            )

        pair_key = (
            str(image_hash),
            str(code),
            int(seed),
        )

        pair_groups[pair_key].append(
            record
        )

    invalid_pairs: list[str] = []

    for pair_key, pair_records in pair_groups.items():
        task_kinds = {
            str(record.get("task_kind"))
            for record in pair_records
        }

        concepts = {
            str(record.get("concept"))
            for record in pair_records
        }

        if len(pair_records) != 2:
            invalid_pairs.append(
                f"{pair_key[0]}: expected 2 rows, "
                f"found {len(pair_records)}"
            )
            continue

        if task_kinds != {
            "local",
            "anchor",
        }:
            invalid_pairs.append(
                f"{pair_key[0]}: "
                f"task kinds={sorted(task_kinds)}"
            )
            continue

        if len(concepts) != 1:
            invalid_pairs.append(
                f"{pair_key[0]}: "
                f"concepts={sorted(concepts)}"
            )

    if invalid_pairs:
        preview = "\n".join(
            f"  - {message}"
            for message in invalid_pairs[:20]
        )

        raise ValueError(
            f"Invalid Stage 2 pairs in {split}:\n"
            f"{preview}"
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
    """한 split을 LLaVA JSON과 PNG로 변환한다."""

    parquet_paths = list_parquet_paths(
        parquet_dir
    )

    validate_stage2_pairs(
        parquet_paths=parquet_paths,
        split=split,
    )

    image_dir = (
        output_dir
        / "images"
        / split
    )

    safe_remove_directory(
        target=image_dir,
        root=output_dir,
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    llava_records: list[dict] = []
    seen_ids: set[str] = set()

    concept_counts: Counter[str] = Counter()
    task_kind_counts: Counter[str] = Counter()

    written_image_hashes: set[str] = set()

    total_records = count_records(
        parquet_paths
    )

    for record in tqdm(
        iter_records(parquet_paths),
        total=total_records,
        desc=f"Converting {split}",
    ):
        image_hash = validate_record(
            record=record,
            split=split,
            seen_ids=seen_ids,
        )

        sample_id = str(record["id"])
        image_name = f"{image_hash}.png"

        image_path = (
            image_dir
            / image_name
        )

        if image_hash not in written_image_hashes:
            image_path.write_bytes(
                record["image"]
            )

            written_image_hashes.add(
                image_hash
            )

        relative_image_path = (
            Path(split)
            / image_name
        )

        llava_record: dict = {
            "id": sample_id,
            "image": relative_image_path.as_posix(),
            "conversations": record["conversations"],
            "image_sha256": image_hash,
        }

        if record.get("task_kind") is not None:
            llava_record["task_kind"] = str(
                record["task_kind"]
            )

        if record.get("concept") is not None:
            llava_record["concept"] = str(
                record["concept"]
            )

        llava_records.append(
            llava_record
        )

        concept = str(
            record.get(
                "concept",
                "unknown",
            )
        )

        concept_counts[concept] += 1

        task_kind = record.get(
            "task_kind"
        )

        if task_kind is not None:
            task_kind_counts[
                str(task_kind)
            ] += 1

    json_path = (
        output_dir
        / f"{split}.json"
    )

    temporary_json_path = (
        output_dir
        / f".{split}.json.tmp"
    )

    try:
        with temporary_json_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                llava_records,
                file,
                ensure_ascii=False,
                indent=2,
            )

        temporary_json_path.replace(
            json_path
        )
    finally:
        if temporary_json_path.exists():
            temporary_json_path.unlink()

    print()
    print(
        f"Converted rows: {len(llava_records)}"
    )

    print(
        f"Unique images: {len(written_image_hashes)}"
    )

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


def remove_unused_validation_output(
    output_dir: Path,
) -> None:
    """train-only 변환 시 과거 validation 결과를 제거한다."""

    validation_json = (
        output_dir
        / "validation.json"
    )

    validation_image_dir = (
        output_dir
        / "images"
        / "validation"
    )

    safe_remove_file(
        target=validation_json,
        root=output_dir,
    )

    safe_remove_directory(
        target=validation_image_dir,
        root=output_dir,
    )


def main() -> None:
    """요청된 split들을 순서대로 변환한다."""

    args = parse_args()

    input_root = args.input_root.resolve()
    output_dir = validate_output_root(
        args.output_dir
    )

    if not input_root.is_dir():
        raise FileNotFoundError(
            f"Input root was not found: {input_root}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if "validation" not in args.splits:
        remove_unused_validation_output(
            output_dir
        )

    for split in args.splits:
        convert_split(
            split=split,
            parquet_dir=(
                input_root
                / f"{split}_parquet"
            ),
            output_dir=output_dir,
        )

    print("Dataset conversion completed.")


if __name__ == "__main__":
    main()