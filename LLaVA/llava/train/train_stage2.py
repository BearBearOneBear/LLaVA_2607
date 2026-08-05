from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import transformers

import llava.train.train as llava_train
from llava.train.attention_utils import (
    rank0_print,
    resolve_attention_implementation,
)


@dataclass
class Stage2ModelArguments(llava_train.ModelArguments):
    """Stage 2에서는 LLM과 multimodal projector를 학습한다."""

    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)


@dataclass
class Stage2DataArguments(llava_train.DataArguments):
    """Stage 2 train/validation 데이터 경로."""

    eval_data_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the validation data."
        },
    )


def make_stage2_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: Stage2DataArguments,
) -> Dict:
    """Stage 2 train/eval dataset과 collator를 생성한다."""

    train_dataset = llava_train.LazySupervisedDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        data_args=data_args,
    )

    eval_dataset = None

    if data_args.eval_data_path is not None:
        eval_dataset = llava_train.LazySupervisedDataset(
            data_path=data_args.eval_data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )

    data_collator = (
        llava_train.DataCollatorForSupervisedDataset(
            tokenizer=tokenizer
        )
    )

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }


def train_stage2() -> None:
    """Stage 2 LLM 및 projector 학습을 실행한다."""

    llava_train.ModelArguments = Stage2ModelArguments
    llava_train.DataArguments = Stage2DataArguments
    llava_train.make_supervised_data_module = (
        make_stage2_data_module
    )

    attention_implementation = (
        resolve_attention_implementation()
    )

    rank0_print(
        "Starting Stage 2 LLM and projector training."
    )

    llava_train.train(
        attn_implementation=attention_implementation
    )


if __name__ == "__main__":
    train_stage2()