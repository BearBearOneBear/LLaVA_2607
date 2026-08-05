from __future__ import annotations

import argparse
import functools
import sys
from typing import Any

import torch

import llava.train.train as llava_train
import llava.train.llava_trainer as llava_trainer
from llava.constants import IGNORE_INDEX
from llava.model.llava_arch import LlavaMetaForCausalLM
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
from llava.train.llava_trainer import LLaVATrainer


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """arguments"""

    parser = argparse.ArgumentParser(
        description="Run Stage 1 or Stage 2 with a tiny LLaVA model."
    )

    parser.add_argument(
        "--stage",
        type=int,
        choices=(1, 2),
        required=True,
        help="Training stage to run.",
    )

    return parser.parse_known_args()


def patch_multimodal_labels() -> None:
    """check labels after image expansion"""

    original_prepare = (
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal
    )

    @functools.wraps(original_prepare)
    def checked_prepare(
        self: LlavaMetaForCausalLM,
        *args: Any,
        **kwargs: Any,
    ):
        outputs = original_prepare(
            self,
            *args,
            **kwargs,
        )

        labels = outputs[-1]

        if labels is not None:
            if labels.ndim == 1:
                labels = labels.unsqueeze(0)

            supervised_tokens = (
                labels != IGNORE_INDEX
            ).sum(dim=-1)

            if torch.any(supervised_tokens == 0):
                raise RuntimeError(
                    "No supervised tokens remain after "
                    "image expansion and truncation. "
                    f"model_max_length="
                    f"{getattr(self.config, 'tokenizer_model_max_length', None)}, "
                    f"supervised_tokens="
                    f"{supervised_tokens.detach().cpu().tolist()}"
                )

        return outputs

    LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = (
        checked_prepare
    )


def patch_loss() -> None:
    """stop non-finite loss"""

    original_compute_loss = (
        LLaVATrainer.compute_loss
    )

    @functools.wraps(original_compute_loss)
    def checked_compute_loss(
        self: LLaVATrainer,
        *args: Any,
        **kwargs: Any,
    ):
        result = original_compute_loss(
            self,
            *args,
            **kwargs,
        )

        if isinstance(result, tuple):
            loss = result[0]
        else:
            loss = result

        if not torch.isfinite(loss).all():
            raise FloatingPointError(
                f"Non-finite loss was detected: "
                f"{loss.detach().float().cpu()}"
            )

        return result

    LLaVATrainer.compute_loss = (
        checked_compute_loss
    )


def patch_cpu_vision_tower() -> None:
    """keep clip in float32 on cpu"""

    if torch.cuda.is_available():
        return

    original_to = CLIPVisionTower.to

    @functools.wraps(original_to)
    def cpu_to(
        self: CLIPVisionTower,
        *args: Any,
        **kwargs: Any,
    ):
        kwargs["dtype"] = torch.float32

        return original_to(
            self,
            *args,
            **kwargs,
        )

    CLIPVisionTower.to = cpu_to


def patch_non_deepspeed_save() -> None:
    """save normal parameters without deepspeed"""

    def clone_parameter(
        parameter: torch.Tensor,
        ignore_status: bool = False,
        name: str | None = None,
    ) -> torch.Tensor:
        del ignore_status
        del name

        return parameter.detach().cpu().clone()

    llava_trainer.maybe_zero_3 = clone_parameter
    llava_train.maybe_zero_3 = clone_parameter

def patch_trainer_output() -> None:
    """print trainable parameters"""

    original_init = LLaVATrainer.__init__

    @functools.wraps(original_init)
    def checked_init(
        self: LLaVATrainer,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        original_init(
            self,
            *args,
            **kwargs,
        )

        total_parameters = sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )

        if trainable_parameters == 0:
            raise RuntimeError(
                "No trainable parameters were found."
            )

        print(
            f"Total parameters: {total_parameters:,}"
        )
        print(
            f"Trainable parameters: {trainable_parameters:,}"
        )

    LLaVATrainer.__init__ = checked_init


def install_patches() -> None:
    """install test patches"""

    patch_multimodal_labels()
    patch_loss()
    patch_cpu_vision_tower()
    patch_non_deepspeed_save()
    patch_trainer_output()

    print("Tiny training patches were installed.")

def run_stage1() -> None:
    """run original stage1 code"""

    from llava.train.train_stage1 import train_stage1

    print("Starting tiny Stage 1 training.")

    train_stage1()


def run_stage2() -> None:
    """run original stage2 code"""

    from llava.train.train_stage2 import train_stage2

    print("Starting tiny Stage 2 training.")

    train_stage2()


def main() -> None:
    """run tiny training"""

    args, training_args = parse_args()

    # 기존 LLaVA HfArgumentParser에는
    # --stage를 제외한 학습 인자만 전달한다.
    sys.argv = [
        sys.argv[0],
        *training_args,
    ]

    install_patches()

    if args.stage == 1:
        run_stage1()
    else:
        run_stage2()


if __name__ == "__main__":
    main()