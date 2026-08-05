from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from transformers import AutoTokenizer

from llava.model.language_model.llava_llama import (
    LlavaConfig,
    LlavaLlamaForCausalLM,
)


DEFAULT_TOKENIZER = "liuhaotian/llava-v1.5-7b"
DEFAULT_OUTPUT_DIR = "./debug_assets/tiny_llava"

HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 256
NUM_HIDDEN_LAYERS = 3
NUM_ATTENTION_HEADS = 4
MAX_POSITION_EMBEDDINGS = 4096


def parse_args() -> argparse.Namespace:
    """arguments"""

    parser = argparse.ArgumentParser(
        description="Create a tiny LLaVA model for training tests."
    )

    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="Tokenizer checkpoint or local tokenizer path.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the tiny model.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the existing output directory.",
    )

    return parser.parse_args()


def prepare_output_dir(
    output_dir: Path,
    overwrite: bool,
) -> None:
    """output path"""

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite to recreate it."
            )

        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_tokenizer(
    tokenizer_name_or_path: str,
) -> AutoTokenizer:
    """tokenizer"""

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    return tokenizer


def create_tiny_model(
    tokenizer: AutoTokenizer,
) -> LlavaLlamaForCausalLM:
    """three layer llava model"""

    config = LlavaConfig(
        vocab_size=len(tokenizer),
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_key_value_heads=NUM_ATTENTION_HEADS,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        initializer_range=0.02,
        pretraining_tp=1,
        use_cache=False,
        tie_word_embeddings=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    config.tokenizer_model_max_length = 2048
    config.tokenizer_padding_side = "right"

    return LlavaLlamaForCausalLM(config)


def count_parameters(
    model: LlavaLlamaForCausalLM,
) -> int:
    """parameter count"""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def save_tiny_model(
    model: LlavaLlamaForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: Path,
) -> None:
    """save model and tokenizer"""

    model.save_pretrained(
        output_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(
        output_dir
    )


def main() -> None:
    """create tiny checkpoint"""

    args = parse_args()

    output_dir = Path(
        args.output_dir
    ).resolve()

    prepare_output_dir(
        output_dir=output_dir,
        overwrite=args.overwrite,
    )

    print("Loading the LLaVA tokenizer.")

    tokenizer = load_tokenizer(
        args.tokenizer_name_or_path
    )

    print("Creating a three-layer tiny LLaVA model.")

    model = create_tiny_model(
        tokenizer
    )

    save_tiny_model(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
    )

    parameter_count = count_parameters(
        model
    )

    print("Tiny LLaVA model creation completed.")
    print(f"Output directory: {output_dir}")
    print(f"Transformer layers: {NUM_HIDDEN_LAYERS}")
    print(f"Hidden size: {HIDDEN_SIZE}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Parameter count: {parameter_count:,}")


if __name__ == "__main__":
    main()