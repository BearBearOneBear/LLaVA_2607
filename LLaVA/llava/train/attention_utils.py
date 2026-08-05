from __future__ import annotations

import os
from typing import Optional


ALLOWED_IMPLEMENTATIONS = {
    "auto",
    "flash_attention_2",
    "sdpa",
    "eager",
    "none",
    "default",
    "",
}

MIN_FLASH_ATTENTION_CAPABILITY = 8


def _get_local_rank() -> int:
    """DeepSpeed가 전달한 local rank를 안전하게 읽는다."""

    value = os.environ.get("LOCAL_RANK", "-1")

    try:
        return int(value)
    except ValueError:
        return -1


def rank0_print(message: str) -> None:
    """단일 프로세스 또는 local rank 0에서만 출력한다."""

    if _get_local_rank() in {-1, 0}:
        print(message)


def _flash_attention_2_usable() -> tuple[bool, str]:
    """FlashAttention-2를 실제로 선택해도 되는지 검사한다."""

    try:
        import torch
        from transformers.utils import (
            is_flash_attn_2_available,
        )
    except ImportError as error:
        return False, f"required import failed: {error!r}"

    if not is_flash_attn_2_available():
        return (
            False,
            "transformers reports FlashAttention-2 as unavailable",
        )

    if not torch.cuda.is_available():
        return False, "CUDA is unavailable"

    # CUDA FlashAttention-2는 Ampere(SM80) 이상에서 사용한다.
    # ROCm에서는 CUDA compute capability 검사를 적용하지 않는다.
    if torch.version.cuda is not None:
        device_count = torch.cuda.device_count()

        if device_count <= 0:
            return False, "no CUDA device was detected"

        local_rank = _get_local_rank()
        device_index = local_rank if local_rank >= 0 else 0

        if device_index >= device_count:
            return (
                False,
                f"LOCAL_RANK={local_rank} exceeds the visible "
                f"device count ({device_count})",
            )

        major, minor = torch.cuda.get_device_capability(
            device_index
        )

        if major < MIN_FLASH_ATTENTION_CAPABILITY:
            return (
                False,
                f"compute capability {major}.{minor} is below "
                f"{MIN_FLASH_ATTENTION_CAPABILITY}.0 (Ampere)",
            )

    return True, ""


def _sdpa_usable() -> bool:
    """현재 Transformers/PyTorch 조합의 SDPA 지원 여부를 확인한다."""

    try:
        from transformers.utils import is_torch_sdpa_available
    except ImportError:
        return False

    return is_torch_sdpa_available()


def resolve_attention_implementation() -> Optional[str]:
    """환경변수로 attention backend를 선택한다.

    ATTENTION_IMPLEMENTATION 허용값:
      auto, flash_attention_2, sdpa, eager, none, default, ""

    auto 선택 순서:
      flash_attention_2 -> sdpa -> Transformers 기본 구현
    """

    requested = os.environ.get(
        "ATTENTION_IMPLEMENTATION",
        "auto",
    ).strip().lower()

    if requested not in ALLOWED_IMPLEMENTATIONS:
        raise ValueError(
            "ATTENTION_IMPLEMENTATION must be one of: "
            "auto, flash_attention_2, sdpa, eager, none, "
            f"default. Received: {requested!r}"
        )

    if requested in {"", "none", "default"}:
        rank0_print(
            "Attention implementation: transformers default"
        )
        return None

    if requested == "eager":
        rank0_print("Attention implementation: eager")
        return "eager"

    if requested == "sdpa":
        if not _sdpa_usable():
            raise RuntimeError(
                "ATTENTION_IMPLEMENTATION=sdpa was explicitly "
                "requested, but this Transformers/PyTorch build "
                "does not support it."
            )

        rank0_print("Attention implementation: sdpa")
        return "sdpa"

    flash_usable, flash_reason = (
        _flash_attention_2_usable()
    )

    if requested == "flash_attention_2":
        if not flash_usable:
            raise RuntimeError(
                "ATTENTION_IMPLEMENTATION=flash_attention_2 was "
                "explicitly requested, but it cannot be used: "
                f"{flash_reason}"
            )

        rank0_print(
            "Attention implementation: flash_attention_2"
        )
        return "flash_attention_2"

    # requested == "auto"
    if flash_usable:
        rank0_print(
            "Attention implementation: flash_attention_2"
        )
        return "flash_attention_2"

    rank0_print(
        "FlashAttention-2 is unavailable. "
        f"Reason: {flash_reason}."
    )

    if _sdpa_usable():
        rank0_print("Attention implementation: sdpa")
        return "sdpa"

    rank0_print(
        "SDPA is unavailable. "
        "Using the Transformers default attention implementation."
    )

    return None