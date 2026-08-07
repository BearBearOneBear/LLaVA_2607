#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import re
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from llava.conversation import conv_templates
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM


NEUTRAL_PROMPTS = [
    "Explain why the sky appears blue.",
    "Write a short paragraph about public libraries.",
    "What is the difference between weather and climate?",
    "Summarize the process of photosynthesis in simple terms.",
    "Give three tips for organizing a study schedule.",
    "Describe how a bicycle works.",
    "Explain the purpose of a database index.",
    "What are the advantages of regular backups?",
    "Write a short story about a train arriving in a small town.",
    "Explain supply and demand to a beginner.",
    "What causes ocean tides?",
    "Describe the basic structure of an essay.",
    "Why do leaves change color in autumn?",
    "Explain what an operating system does.",
    "Give a concise overview of the water cycle.",
    "What is the role of sleep in learning?",
    "Describe how a refrigerator keeps food cold.",
    "Explain the difference between a hypothesis and a theory.",
    "What makes a password strong?",
    "Write a polite reminder about an upcoming meeting.",
    "Explain why exercise can improve cardiovascular fitness.",
    "Describe the main parts of a computer network.",
    "What is inflation?",
    "Explain how vaccines train the immune system at a high level.",
    "Write a brief description of a rainy morning.",
    "What is the difference between renewable and nonrenewable energy?",
    "Explain how a search engine finds web pages.",
    "Give a short explanation of opportunity cost.",
    "Describe the lifecycle of a butterfly.",
    "What is the purpose of peer review in science?",
    "Explain why metal feels colder than wood at the same temperature.",
    "Write a short thank-you note to a colleague.",
    "What is a compiler?",
    "Explain the difference between RAM and storage.",
    "Describe how a bill becomes a law in general terms.",
    "What is the purpose of a checksum?",
    "Explain why seasons occur.",
    "Give a concise description of plate tectonics.",
    "What is a balanced diet?",
    "Explain the idea of compound interest without doing calculations.",
]

DSL_PATTERN = re.compile(
    r"\b(?:POINTS|SEG|CIRCLE|CENTER|PERP|PARA|EQ|ANG|DD|ON|SECTOR)\s*:",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight language-preservation audit for base LLaVA vs Stage 2."
    )
    parser.add_argument("--model-kind", choices=["base", "stage2"], required=True)
    parser.add_argument("--base-model", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--stage2-dir", type=Path, default=Path("./checkpoints/geometry_stage2"))
    parser.add_argument("--output-dir", type=Path, default=Path("./stage2_evaluate/results/language"))
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--ppl-seq-len", type=int, default=512)
    parser.add_argument("--ppl-stride", type=int, default=256)
    parser.add_argument(
        "--ppl-max-chars",
        type=int,
        default=1_500_000,
        help="Limit WikiText text size for a fast but stable comparison; 0 means no limit.",
    )
    return parser.parse_args()


def model_reference(args: argparse.Namespace) -> str:
    if args.model_kind == "base":
        return args.base_model
    if not (args.stage2_dir / "config.json").exists():
        raise FileNotFoundError(f"Stage-2 config not found: {args.stage2_dir / 'config.json'}")
    return str(args.stage2_dir)


def load_text_model(args: argparse.Namespace):
    # Stage 2 is a full LlavaLlamaForCausalLM HF checkpoint. We intentionally do
    # not load the CLIP tower here; text-only forward/generation does not need it.
    ref = model_reference(args)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    model = LlavaLlamaForCausalLM.from_pretrained(
        ref,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def generate_text(prompt_text: str, tokenizer, model, conv_mode: str, max_new_tokens: int) -> str:
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(input_device(model))
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    # LlavaLlamaForCausalLM.generate converts input_ids to inputs_embeds before
    # delegating to HF generation, matching llava/eval/model_vqa.py. Decode the
    # returned IDs directly rather than slicing by the textual prompt length.
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def evaluate_dsl_leakage(args: argparse.Namespace, tokenizer, model) -> dict:
    outputs = []
    leaks = 0
    for prompt in tqdm(NEUTRAL_PROMPTS, desc=f"DSL leakage:{args.model_kind}"):
        text = generate_text(prompt, tokenizer, model, args.conv_mode, args.max_new_tokens)
        leaked = bool(DSL_PATTERN.search(text))
        leaks += int(leaked)
        outputs.append({"prompt": prompt, "response": text, "dsl_leak": leaked})
    return {
        "n": len(outputs),
        "dsl_leak_count": leaks,
        "dsl_leak_rate": leaks / len(outputs) if outputs else 0.0,
        "outputs": outputs,
    }


def load_wikitext(max_chars: int) -> str:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required for WikiText PPL. Install it or run with --skip-ppl."
        ) from exc

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(str(x) for x in dataset["text"] if str(x).strip())
    if max_chars > 0:
        text = text[:max_chars]
    return text


def evaluate_ppl(
    tokenizer,
    model,
    text: str,
    seq_len: int,
    stride: int,
) -> dict:
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    n_tokens = input_ids.size(1)
    if n_tokens < 2:
        raise RuntimeError("WikiText tokenization produced too few tokens.")

    device = input_device(model)
    nll_sum = 0.0
    counted_tokens = 0
    prev_end = 0

    for begin in tqdm(range(0, n_tokens, stride), desc="WikiText PPL"):
        end = min(begin + seq_len, n_tokens)
        trg_len = end - prev_end
        if trg_len <= 0:
            break
        ids = input_ids[:, begin:end].to(device)
        target = ids.clone()
        target[:, :-trg_len] = -100

        with torch.inference_mode():
            outputs = model(input_ids=ids, labels=target, use_cache=False)
        # HF causal-LM loss is computed on shifted labels. Count the exact number
        # of non-ignored labels that survive that shift.
        effective = int((target[:, 1:] != -100).sum().item())
        if effective:
            nll_sum += float(outputs.loss) * effective
            counted_tokens += effective

        prev_end = end
        if end == n_tokens:
            break

    mean_nll = nll_sum / counted_tokens
    return {
        "dataset": "wikitext-2-raw-v1:test",
        "token_count": counted_tokens,
        "sequence_length": seq_len,
        "stride": stride,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_text_model(args)

    result = {
        "model_kind": args.model_kind,
        "model_reference": model_reference(args),
        "dsl_leakage": evaluate_dsl_leakage(args, tokenizer, model),
        "wikitext_ppl": None,
    }

    if not args.skip_ppl:
        try:
            text = load_wikitext(args.ppl_max_chars)
            result["wikitext_ppl"] = evaluate_ppl(
                tokenizer,
                model,
                text,
                seq_len=args.ppl_seq_len,
                stride=args.ppl_stride,
            )
        except Exception as exc:
            result["wikitext_ppl_error"] = f"{type(exc).__name__}: {exc}"
            print(f"WARNING: WikiText PPL was skipped after an error: {exc}")

    output_path = args.output_dir / f"{args.model_kind}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Language audit saved to {output_path}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
