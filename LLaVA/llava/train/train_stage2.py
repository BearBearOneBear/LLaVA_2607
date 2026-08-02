from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import transformers

import llava.train.train as llava_train


@dataclass
class Stage2ModelArguments(llava_train.ModelArguments):
    """train llm and mlp"""

    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)


@dataclass
class Stage2DataArguments(llava_train.DataArguments):
    """data path"""

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


def train_stage2() -> None:
    """extend to original train code"""

    llava_train.ModelArguments = Stage2ModelArguments
    llava_train.DataArguments = Stage2DataArguments

    llava_train.make_supervised_data_module = make_stage2_data_module

    print("Starting Stage 2 LLM and projector training.")

    llava_train.train()


if __name__ == "__main__":
    train_stage2()