#!/usr/bin/env python3
from __future__ import annotations

"""
Shared utilities for geometry evaluation.

Used by:
- accuracy.py
- anchor.py
- represent.py
- later ablation evaluators

This module intentionally contains NO:
- geometry ontology
- metric definition
- dataset path
- checkpoint path
- output path
- experiment-specific result formatting

All paths and runtime settings are supplied by each evaluator,
normally from arguments passed by the top-level .sh runner.
"""

import gc
import json
import random
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Sequence


# =============================================================================
# Repository bootstrap
# =============================================================================

THIS_FILE = Path(__file__).resolve()
LLAVA_ROOT = THIS_FILE.parents[1]

if str(LLAVA_ROOT) not in sys.path:
    sys.path.insert(0, str(LLAVA_ROOT))


import pyarrow.parquet as pq
import torch
from PIL import Image

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


# =============================================================================
# Common model configuration
# =============================================================================

SUPPORTED_MODEL_KINDS = (
    "base",
    "stage1",
    "stage2",
    "stage2_only",
)


@dataclass(frozen=True)
class EvaluationModelPaths:
    """
    Model/checkpoint locations supplied by an evaluator or shell runner.

    No default path is defined here.
    """

    base_model: str
    stage1_dir: Path
    stage2_dir: Path
    stage2_only_dir: Path

    @classmethod
    def from_values(
        cls,
        *,
        base_model: str,
        stage1_dir: str | Path,
        stage2_dir: str | Path,
        stage2_only_dir: str | Path,
    ) -> "EvaluationModelPaths":
        return cls(
            base_model=str(base_model),
            stage1_dir=Path(stage1_dir),
            stage2_dir=Path(stage2_dir),
            stage2_only_dir=Path(stage2_only_dir),
        )


# =============================================================================
# JSON
# =============================================================================

def load_json_value(value: Any) -> Any:
    """
    Decode a JSON-valued parquet cell.

    Bad input is not silently coerced.
    """

    if isinstance(value, str):
        return json.loads(value)

    if isinstance(value, (dict, list)):
        return value

    raise TypeError(
        "Expected a JSON string/dict/list, "
        f"got {type(value)!r}."
    )


# =============================================================================
# Parquet loading / validation
# =============================================================================

def as_path_list(
    paths: Sequence[str | Path],
) -> list[Path]:
    return [
        Path(path)
        for path in paths
    ]


def validate_parquet_file(
    path: str | Path,
    *,
    minimum_bytes: int = 100,
) -> Path:
    """
    Verify that a parquet shard exists and is not obviously truncated.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation parquet was not found: {path}"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluation parquet path is not a file: {path}"
        )

    size = path.stat().st_size

    if size < minimum_bytes:
        raise RuntimeError(
            "Evaluation parquet is suspiciously small "
            f"({size} bytes): {path}\n"
            "This usually indicates a truncated/corrupted binary file."
        )

    return path


def read_parquet_files(
    paths: Sequence[str | Path],
    dataset_name: str,
    *,
    minimum_bytes: int = 100,
    require_same_schema: bool = True,
) -> list[dict[str, Any]]:
    """
    Read multiple parquet shards.

    When require_same_schema=True, every shard must expose the same
    top-level columns. This catches accidental shard/schema mismatches.
    """

    resolved_paths = as_path_list(paths)

    if not resolved_paths:
        raise ValueError(
            f"No parquet paths were supplied for {dataset_name}."
        )

    rows: list[dict[str, Any]] = []
    reference_columns: set[str] | None = None

    for path in resolved_paths:
        path = validate_parquet_file(
            path,
            minimum_bytes=minimum_bytes,
        )

        try:
            table = pq.read_table(path)

        except Exception as exc:
            raise RuntimeError(
                f"Failed to read parquet shard {path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        shard_columns = set(
            table.schema.names
        )

        if reference_columns is None:
            reference_columns = shard_columns

        elif (
            require_same_schema
            and shard_columns != reference_columns
        ):
            missing = sorted(
                reference_columns - shard_columns
            )

            extra = sorted(
                shard_columns - reference_columns
            )

            raise ValueError(
                f"Parquet schema mismatch in {path}. "
                f"Missing columns: {missing}; "
                f"extra columns: {extra}."
            )

        shard = table.to_pylist()

        if not shard:
            raise RuntimeError(
                f"Parquet shard contains zero rows: {path}"
            )

        rows.extend(shard)

        print(
            f"Loaded {len(shard):,} rows from {path}"
        )

    if not rows:
        raise RuntimeError(
            f"{dataset_name} contains no rows."
        )

    print(
        f"{dataset_name}: "
        f"{len(rows):,} rows loaded "
        f"from {len(resolved_paths):,} shard(s)."
    )

    return rows


def validate_required_columns(
    rows: Sequence[dict[str, Any]],
    required: Iterable[str],
    dataset_name: str,
    *,
    max_examples: int = 5,
) -> None:
    """
    Validate required fields across every row.

    This intentionally does not inspect rows[0] only.
    """

    required_set = set(required)

    bad: list[
        tuple[int, list[str]]
    ] = []

    for index, row in enumerate(rows):
        missing = sorted(
            required_set - set(row)
        )

        if missing:
            bad.append(
                (index, missing)
            )

            if len(bad) >= max_examples:
                break

    if bad:
        examples = "; ".join(
            f"row {index}: {missing}"
            for index, missing in bad
        )

        raise ValueError(
            f"{dataset_name} has row(s) "
            f"with missing columns: {examples}"
        )


def validate_concepts(
    rows: Sequence[dict[str, Any]],
    allowed_concepts: Iterable[str],
    dataset_name: str,
    *,
    concept_column: str = "concept",
) -> None:
    """
    Check that every concept belongs to the evaluator ontology.
    """

    allowed = set(
        allowed_concepts
    )

    unknown = sorted(
        {
            str(row.get(concept_column))
            for row in rows
            if str(row.get(concept_column))
            not in allowed
        }
    )

    if unknown:
        raise ValueError(
            f"Unexpected {dataset_name} "
            f"concept(s): {unknown}"
        )


def validate_nonempty_column(
    rows: Sequence[dict[str, Any]],
    column: str,
    dataset_name: str,
    *,
    max_examples: int = 5,
) -> None:
    """
    Verify that a text-like column is not empty.
    """

    bad: list[int] = []

    for index, row in enumerate(rows):
        value = row.get(column)

        if (
            value is None
            or not str(value).strip()
        ):
            bad.append(index)

            if len(bad) >= max_examples:
                break

    if bad:
        raise ValueError(
            f"{dataset_name} has empty "
            f"{column!r} values at row(s): {bad}"
        )


def filter_task_rows(
    rows: Sequence[dict[str, Any]],
    task_kind: str,
) -> list[dict[str, Any]]:
    """
    Select local / anchor rows without duplicating the filter in evaluators.
    """

    target = (
        str(task_kind)
        .strip()
        .lower()
    )

    return [
        row
        for row in rows
        if (
            str(row.get("task_kind") or "")
            .strip()
            .lower()
            == target
        )
    ]


def stable_sample_key(
    row: dict[str, Any],
    index: int,
) -> str:
    """
    Stable sample identifier shared across evaluators.

    figure_id is preferred so local/anchor rows can later be joined.
    """

    for key in (
        "figure_id",
        "image_sha256",
        "id",
        "problem_id",
    ):
        value = row.get(key)

        if value not in (
            None,
            "",
        ):
            return str(value)

    return str(index)


def deduplicate_rows(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Keep the first row for each stable sample key.

    Useful when only one representation per figure is needed.
    """

    unique: dict[
        str,
        dict[str, Any],
    ] = {}

    for index, row in enumerate(rows):
        unique.setdefault(
            stable_sample_key(
                row,
                index,
            ),
            row,
        )

    return list(
        unique.values()
    )


def stratified_limit(
    rows: Sequence[dict[str, Any]],
    concepts: Sequence[str],
    limit: int,
    *,
    seed: int,
    concept_column: str = "concept",
) -> list[dict[str, Any]]:
    """
    Deterministic approximately-balanced subset for smoke tests.

    Selection is round-robin across concepts instead of rows[:N].
    Therefore a small smoke-test limit does not collapse onto the
    first few concepts in parquet order.
    """

    rows = list(rows)

    if (
        limit <= 0
        or len(rows) <= limit
    ):
        return rows

    rng = random.Random(seed)

    groups: dict[
        str,
        list[dict[str, Any]],
    ] = {
        concept: []
        for concept in concepts
    }

    for row in rows:
        concept = str(
            row.get(concept_column)
        )

        if concept in groups:
            groups[concept].append(
                row
            )

    for group in groups.values():
        rng.shuffle(group)

    selected: list[
        dict[str, Any]
    ] = []

    depth = 0

    while len(selected) < limit:
        added = False

        for concept in concepts:
            group = groups[concept]

            if depth < len(group):
                selected.append(
                    group[depth]
                )

                added = True

                if len(selected) >= limit:
                    break

        if not added:
            break

        depth += 1

    return selected


def warn_expected_count(
    actual: int,
    expected: int,
    description: str,
) -> None:
    """
    Expected evaluation sizes remain warnings, not hard constraints.
    """

    if actual != expected:
        print(
            f"WARNING: {description} has "
            f"{actual:,} rows; "
            f"expected {expected:,}."
        )


# =============================================================================
# Model loading / unloading
# =============================================================================

def require_file(
    path: str | Path,
    description: str,
) -> Path:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"{description} was not found: {path}"
        )

    return path


def load_eval_model(
    model_kind: str,
    model_paths: EvaluationModelPaths,
):
    """
    Load one evaluation checkpoint.

    Layout
    ------
    base:
        full liuhaotian/llava-v1.5-7b

    stage1:
        base LLaVA
        + Stage1 config
        + mm_projector.bin

    stage2:
        full Stage1 -> Stage2 checkpoint

    stage2_only:
        full Stage2-only checkpoint
    """

    if model_kind not in SUPPORTED_MODEL_KINDS:
        raise ValueError(
            f"Unknown model kind: {model_kind}. "
            f"Expected one of "
            f"{SUPPORTED_MODEL_KINDS}."
        )

    disable_torch_init()

    if model_kind == "base":
        model_path = (
            model_paths.base_model
        )

        model_base = None

        model_name = (
            "llava-v1.5-7b-base"
        )

    elif model_kind == "stage1":
        require_file(
            model_paths.stage1_dir
            / "config.json",
            "Stage 1 config",
        )

        require_file(
            model_paths.stage1_dir
            / "mm_projector.bin",
            "Stage 1 projector",
        )

        model_path = str(
            model_paths.stage1_dir
        )

        model_base = (
            model_paths.base_model
        )

        model_name = (
            "llava-v1.5-7b-stage1"
        )

    elif model_kind == "stage2":
        require_file(
            model_paths.stage2_dir
            / "config.json",
            "Stage 2 config",
        )

        model_path = str(
            model_paths.stage2_dir
        )

        model_base = None

        model_name = (
            "llava-v1.5-7b-stage2"
        )

    else:
        # stage2_only

        require_file(
            model_paths.stage2_only_dir
            / "config.json",
            "Stage 2-only config",
        )

        model_path = str(
            model_paths.stage2_only_dir
        )

        model_base = None

        model_name = (
            "llava-v1.5-7b-stage2-only"
        )

    print(
        "\n"
        + "=" * 78
    )

    print(
        f"Loading model: {model_kind}"
    )

    print(
        f"model_path: {model_path}"
    )

    print(
        f"model_base: {model_base}"
    )

    print(
        "=" * 78
    )

    (
        tokenizer,
        model,
        image_processor,
        context_len,
    ) = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        load_8bit=False,
        load_4bit=False,
        device_map="auto",
    )

    model.eval()

    return (
        tokenizer,
        model,
        image_processor,
        context_len,
    )


def model_input_device(
    model: torch.nn.Module,
) -> torch.device:
    try:
        return (
            model
            .get_input_embeddings()
            .weight
            .device
        )

    except Exception:
        return next(
            model.parameters()
        ).device


def model_dtype(
    model: torch.nn.Module,
) -> torch.dtype:
    try:
        return next(
            parameter.dtype
            for parameter
            in model.parameters()
            if parameter.is_floating_point()
        )

    except StopIteration:
        return torch.float16


def unload_model(
    tokenizer,
    model,
    image_processor,
) -> None:
    del tokenizer
    del model
    del image_processor

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def is_cuda_oom(
    exc: BaseException,
) -> bool:
    text = str(exc).lower()

    return (
        isinstance(
            exc,
            torch.cuda.OutOfMemoryError,
        )
        or "out of memory" in text
    )


# =============================================================================
# Image
# =============================================================================

def decode_image(
    image_value: Any,
) -> Image.Image:
    if isinstance(
        image_value,
        bytes,
    ):
        return Image.open(
            BytesIO(image_value)
        ).convert("RGB")

    if isinstance(
        image_value,
        dict,
    ):
        if (
            image_value.get("bytes")
            is not None
        ):
            return Image.open(
                BytesIO(
                    image_value["bytes"]
                )
            ).convert("RGB")

        if image_value.get("path"):
            return Image.open(
                image_value["path"]
            ).convert("RGB")

    if isinstance(
        image_value,
        str,
    ):
        return Image.open(
            image_value
        ).convert("RGB")

    if isinstance(
        image_value,
        Image.Image,
    ):
        return image_value.convert(
            "RGB"
        )

    raise TypeError(
        "Unsupported image field type: "
        f"{type(image_value)!r}."
    )


def resolve_image(
    image_value: Any,
    *,
    image_mode: str = "normal",
    image_override: Any | None = None,
) -> Image.Image:
    """
    Resolve normal / blank image input.

    image_override can later be used for shuffled-image ablations.
    """

    if image_mode not in {
        "normal",
        "blank",
    }:
        raise ValueError(
            "resolve_image only supports "
            "'normal' or 'blank', "
            f"got {image_mode!r}."
        )

    source = (
        image_value
        if image_override is None
        else image_override
    )

    image = decode_image(
        source
    )

    if image_mode == "blank":
        image = Image.new(
            "RGB",
            image.size,
            (255, 255, 255),
        )

    return image


def prepare_image_tensor(
    image: Image.Image,
    *,
    image_processor,
    model,
) -> torch.Tensor:
    device = model_input_device(
        model
    )

    image_tensor = process_images(
        [image],
        image_processor,
        model.config,
    )[0]

    return (
        image_tensor
        .unsqueeze(0)
        .to(
            device=device,
            dtype=model_dtype(model),
        )
    )


# =============================================================================
# Conversation rendering
# =============================================================================

def render_conversation(
    user_text: str,
    *,
    model,
    conv_mode: str,
    use_image: bool,
) -> str:
    """
    Render one LLaVA conversation prompt.

    Existing <image> tokens are stripped first so the helper owns image-token
    placement consistently.
    """

    text = (
        str(user_text)
        .replace(
            DEFAULT_IMAGE_TOKEN,
            "",
        )
        .strip()
    )

    if not text:
        raise ValueError(
            "Evaluation prompt is empty."
        )

    if use_image:
        if getattr(
            model.config,
            "mm_use_im_start_end",
            False,
        ):
            text = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + text
            )

        else:
            text = (
                DEFAULT_IMAGE_TOKEN
                + "\n"
                + text
            )

    conv = conv_templates[
        conv_mode
    ].copy()

    conv.append_message(
        conv.roles[0],
        text,
    )

    conv.append_message(
        conv.roles[1],
        None,
    )

    return conv.get_prompt()


# =============================================================================
# Generation
# =============================================================================

def generate_response(
    *,
    prompt_text: str,
    tokenizer,
    model,
    image_processor,
    conv_mode: str,
    max_new_tokens: int,
    image_value: Any | None = None,
    image_mode: str = "normal",
    image_override: Any | None = None,
) -> str:
    """
    Generate one deterministic model response.

    image_mode
    ----------
    normal:
        normal image input

    blank:
        white image with the same dimensions

    none:
        text-only input

    image_override
    --------------
    Can later be used to supply another figure for shuffled-image ablation.
    """

    if image_mode not in {
        "normal",
        "blank",
        "none",
    }:
        raise ValueError(
            f"Unsupported image_mode: "
            f"{image_mode!r}"
        )

    use_image = (
        image_mode != "none"
    )

    rendered = render_conversation(
        prompt_text,
        model=model,
        conv_mode=conv_mode,
        use_image=use_image,
    )

    device = model_input_device(
        model
    )

    generate_kwargs: dict[
        str,
        Any,
    ] = {
        "do_sample": False,
        "temperature": 0.0,
        "num_beams": 1,
        "max_new_tokens": int(
            max_new_tokens
        ),
        "use_cache": True,
    }

    if use_image:
        if image_value is None:
            raise ValueError(
                "image_value is required "
                "when image_mode is not 'none'."
            )

        input_ids = (
            tokenizer_image_token(
                rendered,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )
            .unsqueeze(0)
            .to(device)
        )

        image = resolve_image(
            image_value,
            image_mode=image_mode,
            image_override=image_override,
        )

        generate_kwargs[
            "images"
        ] = prepare_image_tensor(
            image,
            image_processor=image_processor,
            model=model,
        )

        generate_kwargs[
            "image_sizes"
        ] = [
            image.size
        ]

    else:
        input_ids = tokenizer(
            rendered,
            return_tensors="pt",
        ).input_ids.to(
            device
        )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            **generate_kwargs,
        )

    return (
        tokenizer
        .batch_decode(
            output_ids,
            skip_special_tokens=True,
        )[0]
        .strip()
    )


# =============================================================================
# Hidden-state forward
# =============================================================================

def validate_hidden_layers(
    hidden_states: Sequence[
        torch.Tensor
    ],
    layers: Sequence[int],
) -> None:
    """
    hidden_states[0] = embedding output
    hidden_states[k] = output after transformer block k

    Therefore user-facing L8/L16/L24/L32 map directly to indices
    8/16/24/32.
    """

    max_layer = (
        len(hidden_states)
        - 1
    )

    bad = [
        int(layer)
        for layer in layers
        if (
            int(layer) < 1
            or int(layer) > max_layer
        )
    ]

    if bad:
        raise ValueError(
            f"Requested layer(s) {bad}, "
            f"but model exposes transformer "
            f"layers 1..{max_layer}."
        )


def last_token_hidden_states(
    outputs,
    layers: Sequence[int],
) -> dict[int, torch.Tensor]:
    hidden_states = (
        outputs.hidden_states
    )

    validate_hidden_layers(
        hidden_states,
        layers,
    )

    return {
        int(layer): (
            hidden_states[
                int(layer)
            ][0, -1, :]
            .detach()
            .float()
            .cpu()
        )
        for layer in layers
    }


def forward_multimodal_hidden_states(
    *,
    image_value: Any,
    prompt_text: str,
    tokenizer,
    model,
    image_processor,
    conv_mode: str,
    layers: Sequence[int],
    image_mode: str = "normal",
    image_override: Any | None = None,
) -> dict[int, torch.Tensor]:
    """
    Forward one image + text sample and return the last-token hidden state
    from all requested transformer layers.
    """

    if image_mode not in {
        "normal",
        "blank",
    }:
        raise ValueError(
            "forward_multimodal_hidden_states "
            "supports only 'normal' or 'blank', "
            f"got {image_mode!r}."
        )

    rendered = render_conversation(
        prompt_text,
        model=model,
        conv_mode=conv_mode,
        use_image=True,
    )

    device = model_input_device(
        model
    )

    input_ids = (
        tokenizer_image_token(
            rendered,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )
        .unsqueeze(0)
        .to(device)
    )

    image = resolve_image(
        image_value,
        image_mode=image_mode,
        image_override=image_override,
    )

    image_tensor = prepare_image_tensor(
        image,
        image_processor=image_processor,
        model=model,
    )

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            images=image_tensor,
            image_sizes=[
                image.size
            ],
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    return last_token_hidden_states(
        outputs,
        layers,
    )


def forward_text_hidden_states(
    *,
    prompt_text: str,
    tokenizer,
    model,
    conv_mode: str,
    layers: Sequence[int],
) -> dict[int, torch.Tensor]:
    """
    Forward one text-only sample and return the last-token hidden state
    from all requested transformer layers.
    """

    rendered = render_conversation(
        prompt_text,
        model=model,
        conv_mode=conv_mode,
        use_image=False,
    )

    device = model_input_device(
        model
    )

    input_ids = tokenizer(
        rendered,
        return_tensors="pt",
    ).input_ids.to(
        device
    )

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    return last_token_hidden_states(
        outputs,
        layers,
    )