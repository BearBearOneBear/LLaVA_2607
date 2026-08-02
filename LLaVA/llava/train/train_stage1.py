from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import transformers

import llava.train.train as llava_train


@dataclass
class Stage1ModelArguments(llava_train.ModelArguments):
    """train mlp only"""

    tune_mm_mlp_adapter: bool = field(default=True)


@dataclass
class Stage1DataArguments(llava_train.DataArguments):
    """data path"""

    eval_data_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the validation data."
        },
    )


def make_stage1_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: Stage1DataArguments,
) -> Dict:
    """dataset"""

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

    data_collator = llava_train.DataCollatorForSupervisedDataset(
        tokenizer=tokenizer
    )

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }


def train_stage1() -> None:
    """extend to original train code"""

    llava_train.ModelArguments = Stage1ModelArguments
    llava_train.DataArguments = Stage1DataArguments

    llava_train.make_supervised_data_module = make_stage1_data_module

    print("Starting Stage 1 projector training.")

    llava_train.train()


if __name__ == "__main__":
    train_stage1()