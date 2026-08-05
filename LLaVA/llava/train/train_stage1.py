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
class Stage1ModelArguments(llava_train.ModelArguments):
    """Stage 1에서는 multimodal projector만 학습한다."""

    tune_mm_mlp_adapter: bool = field(default=True)


@dataclass
class Stage1DataArguments(llava_train.DataArguments):
    """Stage 1 데이터 경로."""

    eval_data_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional path to validation data."
        },
    )


def make_stage1_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: Stage1DataArguments,
) -> Dict:
    """Stage 1 train/eval dataset과 collator를 생성한다."""

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


def train_stage1() -> None:
    """Stage 1 projector 학습을 실행한다."""

    llava_train.ModelArguments = Stage1ModelArguments
    llava_train.DataArguments = Stage1DataArguments
    llava_train.make_supervised_data_module = (
        make_stage1_data_module
    )

    attention_implementation = (
        resolve_attention_implementation()
    )

    rank0_print("Starting Stage 1 projector training.")

    llava_train.train(
        attn_implementation=attention_implementation
    )


if __name__ == "__main__":
    train_stage1()