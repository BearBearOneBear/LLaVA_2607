#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


# =============================================================================
# Repository bootstrap
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# Constants
# =============================================================================

MODEL_ORDER = (
    "base",
    "stage2",
    "stage2_only",
)

HF_PARQUET = (
    "wikitext-2-raw-v1/"
    "test-00000-of-00001.parquet"
)

HF_RESOLVE = (
    "https://huggingface.co/datasets/"
    "Salesforce/wikitext/resolve/main/"
    + HF_PARQUET
)

GITHUB_MIRROR = (
    "https://raw.githubusercontent.com/"
    "pytorch/examples/main/"
    "word_language_model/data/"
    "wikitext-2/test.txt"
)

# A short response is likely an HTML/error document rather than WikiText.
MIN_CORPUS_CHARS = 200_000


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate text-only language-model perplexity for "
            "Base, Stage2, and Stage2-only checkpoints."
        )
    )

    # Models
    parser.add_argument(
        "--base-model",
        required=True,
        help=(
            "Base LLaVA model ID/path, e.g. "
            "liuhaotian/llava-v1.5-7b."
        ),
    )

    parser.add_argument(
        "--stage2-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--stage2-only-dir",
        type=Path,
        required=True,
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-file",
        default="language_result.json",
    )

    # Corpus
    parser.add_argument(
        "--text-file",
        type=Path,
        default=None,
        help=(
            "Optional local UTF-8 text corpus. "
            "When supplied, all network download routes are skipped."
        ),
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=1_000_000,
        help=(
            "Maximum corpus characters to evaluate. "
            "0 uses the full corpus."
        ),
    )

    # PPL
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=256,
    )

    args = parser.parse_args()

    if args.seq_len < 2:
        raise ValueError(
            "--seq-len must be >= 2."
        )

    if args.stride < 1:
        raise ValueError(
            "--stride must be >= 1."
        )

    if args.stride > args.seq_len:
        raise ValueError(
            "--stride must be <= --seq-len."
        )

    if args.max_chars < 0:
        raise ValueError(
            "--max-chars must be >= 0."
        )

    return args


# =============================================================================
# Corpus loading
# =============================================================================

def _from_datasets() -> str:
    """
    Load WikiText-2 through the Hugging Face datasets interface.
    """

    from datasets import load_dataset

    split = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
    )

    return "\n\n".join(
        str(value)
        for value in split["text"]
        if str(value).strip()
    )


def _from_hub_file() -> str:
    """
    Download the parquet directly through huggingface_hub.
    """

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(
        repo_id="Salesforce/wikitext",
        filename=HF_PARQUET,
        repo_type="dataset",
    )

    column = pq.read_table(
        path,
        columns=["text"],
    ).column("text")

    return "\n\n".join(
        str(value)
        for value in column.to_pylist()
        if str(value).strip()
    )


def _from_hub_url() -> str:
    """
    Download the parquet through the raw Hugging Face resolve URL.
    """

    import io
    import urllib.request

    import pyarrow.parquet as pq

    with urllib.request.urlopen(
        HF_RESOLVE,
        timeout=60,
    ) as response:
        payload = response.read()

    column = pq.read_table(
        io.BytesIO(payload),
        columns=["text"],
    ).column("text")

    return "\n\n".join(
        str(value)
        for value in column.to_pylist()
        if str(value).strip()
    )


def _from_github() -> str:
    """
    WikiText-2 copy vendored in pytorch/examples.

    This version is tokenized rather than raw and therefore may contain
    artifacts such as <unk> and @-@.

    Its absolute PPL is not directly comparable to the raw WikiText split.
    It is still usable for the current within-run comparison because every
    checkpoint receives exactly the same text and tokenizer.
    """

    import urllib.request

    with urllib.request.urlopen(
        GITHUB_MIRROR,
        timeout=60,
    ) as response:
        return response.read().decode(
            "utf-8"
        )


CORPUS_ROUTES = (
    (
        "hf_datasets",
        _from_datasets,
        "wikitext-2-raw-v1:test",
    ),
    (
        "hf_hub_file",
        _from_hub_file,
        "wikitext-2-raw-v1:test",
    ),
    (
        "hf_url",
        _from_hub_url,
        "wikitext-2-raw-v1:test",
    ),
    (
        "github_pytorch_examples",
        _from_github,
        "wikitext-2-v1:test (tokenised)",
    ),
)


def load_corpus(
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    """
    Return corpus text and a fingerprint describing exactly what is scored.
    """

    def finish(
        text: str,
        source: str,
    ) -> tuple[str, dict[str, Any]]:
        available_chars = len(
            text
        )

        if args.max_chars > 0:
            text = text[
                :args.max_chars
            ]

        if not text.strip():
            raise RuntimeError(
                "Language corpus is empty."
            )

        return text, {
            "source": source,
            "available_chars": available_chars,
            "used_chars": len(text),
            "used_words": len(text.split()),
            "sha256": hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest(),
        }

    # -------------------------------------------------------------------------
    # Local corpus
    # -------------------------------------------------------------------------
    if args.text_file is not None:
        if not args.text_file.exists():
            raise FileNotFoundError(
                f"Text file not found: {args.text_file}"
            )

        if not args.text_file.is_file():
            raise FileNotFoundError(
                f"Text path is not a file: {args.text_file}"
            )

        text = args.text_file.read_text(
            encoding="utf-8",
            errors="replace",
        )

        return finish(
            text,
            str(args.text_file),
        )

    # -------------------------------------------------------------------------
    # Network fallbacks
    # -------------------------------------------------------------------------
    failures: list[str] = []

    for name, loader, label in CORPUS_ROUTES:
        try:
            text = loader()

            if len(text) < MIN_CORPUS_CHARS:
                raise RuntimeError(
                    f"returned only {len(text):,} characters, "
                    f"below minimum {MIN_CORPUS_CHARS:,}; "
                    "the response may be an error document"
                )

            print(
                f"Corpus via {name}: "
                f"{len(text):,} chars "
                f"({label})"
            )

            return finish(
                text,
                f"{label} via {name}",
            )

        except Exception as exc:
            failures.append(
                f"  {name}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                f"  route {name} failed: "
                f"{type(exc).__name__}"
            )

    raise RuntimeError(
        "Every corpus route failed:\n"
        + "\n".join(failures)
        + "\n\nPass --text-file with a local English corpus."
    )


# =============================================================================
# Tokenizer / models
# =============================================================================

def load_base_tokenizer(
    base_model: str,
):
    """
    One Base tokenizer is shared by every checkpoint.

    PPL comparison becomes difficult to interpret if checkpoints tokenize
    the corpus differently, so tokenization is intentionally fixed here.
    """

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.unk_token
        )

    return tokenizer


def model_reference(
    kind: str,
    args: argparse.Namespace,
) -> str:
    if kind == "base":
        return args.base_model

    if kind == "stage2":
        directory = args.stage2_dir

    elif kind == "stage2_only":
        directory = args.stage2_only_dir

    else:
        raise ValueError(
            f"Unknown model kind: {kind!r}"
        )

    config = (
        directory
        / "config.json"
    )

    if not config.exists():
        raise FileNotFoundError(
            f"{kind} config not found: {config}"
        )

    return str(
        directory
    )


def load_model(
    kind: str,
    args: argparse.Namespace,
    tokenizer,
):
    """
    Load only the LLaVA causal-LM checkpoint.

    Text-only forward does not invoke the vision tower.
    """

    from llava.model.language_model.llava_llama import (
        LlavaLlamaForCausalLM,
    )

    reference = model_reference(
        kind,
        args,
    )

    model = (
        LlavaLlamaForCausalLM
        .from_pretrained(
            reference,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    )

    model.eval()

    embedding_rows = (
        model.get_input_embeddings()
        .weight.shape[0]
    )

    if embedding_rows < len(
        tokenizer
    ):
        raise RuntimeError(
            f"{kind} has {embedding_rows:,} embedding rows "
            f"against Base tokenizer size {len(tokenizer):,}. "
            "The shared tokenizer cannot safely be used."
        )

    return model, reference


# =============================================================================
# Tokenization
# =============================================================================

def tokenize_corpus(
    tokenizer,
    text: str,
) -> torch.Tensor:
    """
    Tokenize exactly once with the Base tokenizer.

    The same tensor is used for Base, Stage2, and Stage2-only.
    """

    input_ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
    ).input_ids

    if (
        input_ids.ndim != 2
        or input_ids.shape[0] != 1
    ):
        raise RuntimeError(
            "Unexpected tokenized corpus shape: "
            f"{tuple(input_ids.shape)}"
        )

    if input_ids.shape[1] < 2:
        raise RuntimeError(
            "Corpus tokenized to fewer than two tokens."
        )

    return input_ids


# =============================================================================
# Perplexity
# =============================================================================

def measure_perplexity(
    *,
    model,
    input_ids: torch.Tensor,
    seq_len: int,
    stride: int,
    model_kind: str,
) -> dict[str, Any]:
    """
    Sliding-window perplexity.

    Windows overlap for context, while each target token is scored only once.

    Window mean losses are weighted by the exact number of labels that survive
    the causal shift, so the final short window does not receive excess weight.
    """

    total_tokens = int(
        input_ids.shape[1]
    )

    device = (
        model.get_input_embeddings()
        .weight.device
    )

    nll_sum = 0.0
    scored_tokens = 0
    previous_end = 0
    windows = 0

    progress = tqdm(
        range(
            0,
            total_tokens,
            stride,
        ),
        desc=f"{model_kind} / perplexity",
        unit="window",
    )

    for begin in progress:
        end = min(
            begin + seq_len,
            total_tokens,
        )

        target_len = (
            end
            - previous_end
        )

        if target_len <= 0:
            break

        ids = input_ids[
            :,
            begin:end,
        ].to(
            device
        )

        target = ids.clone()

        if target_len < target.shape[1]:
            target[
                :,
                :-target_len,
            ] = -100

        # CausalLM loss shifts labels internally.
        effective = int(
            (
                target[
                    :,
                    1:
                ]
                != -100
            )
            .sum()
            .item()
        )

        if effective > 0:
            with torch.inference_mode():
                outputs = model(
                    input_ids=ids,
                    labels=target,
                    use_cache=False,
                    return_dict=True,
                )

            loss = float(
                outputs.loss
                .detach()
                .float()
                .cpu()
            )

            if not math.isfinite(
                loss
            ):
                raise RuntimeError(
                    f"Non-finite loss for "
                    f"{model_kind}: {loss}"
                )

            nll_sum += (
                loss
                * effective
            )

            scored_tokens += (
                effective
            )

        windows += 1
        previous_end = end

        current_nll = (
            nll_sum
            / scored_tokens
            if scored_tokens
            else float("nan")
        )

        progress.set_postfix(
            mean_nll=(
                f"{current_nll:.4f}"
                if math.isfinite(
                    current_nll
                )
                else "-"
            ),
            scored=scored_tokens,
        )

        if end == total_tokens:
            break

    if scored_tokens == 0:
        raise RuntimeError(
            f"No tokens were scored for {model_kind}."
        )

    mean_nll = (
        nll_sum
        / scored_tokens
    )

    perplexity = (
        math.exp(mean_nll)
        if mean_nll < 709
        else math.inf
    )

    return {
        "scored_tokens": scored_tokens,
        "total_tokens": total_tokens,
        "seq_len": seq_len,
        "stride": stride,
        "windows": windows,
        "mean_nll": mean_nll,
        "perplexity": perplexity,
    }


# =============================================================================
# Output
# =============================================================================

def write_result(
    *,
    output_file: Path,
    args: argparse.Namespace,
    corpus_info: dict[str, Any],
    tokenizer,
    results: dict[str, dict[str, Any]],
    complete: bool,
) -> None:
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "status": (
            "complete"
            if complete
            else "partial"
        ),

        "run": {
            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "base_tokenizer": args.base_model,
            "tokenizer_size": len(tokenizer),
            "seq_len": args.seq_len,
            "stride": args.stride,
            "max_chars": args.max_chars,
        },

        "corpus": corpus_info,

        "models": results,

        "metric": {
            "name": "text_only_perplexity",
            "mean_nll": (
                "token-weighted negative log-likelihood "
                "over uniquely scored target tokens"
            ),
            "perplexity": (
                "exp(mean_nll); lower is better"
            ),
        },

        "notes": [
            (
                "Base, Stage2, and Stage2-only use the same Base tokenizer "
                "and the exact same tokenized corpus."
            ),
            (
                "Sliding windows overlap for context, but each target token "
                "contributes to the loss at most once."
            ),
            (
                "This evaluator reports raw measurements only. "
                "Relative changes against Base are computed in compare.py."
            ),
            (
                "If the GitHub PyTorch WikiText fallback is used, the corpus "
                "is tokenised WikiText rather than the raw split, so absolute "
                "PPL should not be compared directly with runs using the raw "
                "WikiText corpus."
            ),
        ],
    }

    output_file.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Updated result: {output_file}"
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    output_file = (
        args.output_dir
        / args.output_file
    )

    print(
        f"Repository root : {REPO_ROOT}"
    )

    print(
        f"Stage2 dir      : {args.stage2_dir}"
    )

    print(
        f"Stage2-only dir : {args.stage2_only_dir}"
    )

    print(
        f"Output           : {output_file}"
    )

    # -------------------------------------------------------------------------
    # Corpus
    # -------------------------------------------------------------------------
    corpus, corpus_info = load_corpus(
        args
    )

    print(
        "\nCorpus: "
        f"{corpus_info['source']}"
    )

    print(
        f"Used chars: "
        f"{corpus_info['used_chars']:,}"
    )

    print(
        f"SHA256: "
        f"{corpus_info['sha256'][:16]}"
    )

    # -------------------------------------------------------------------------
    # One shared tokenizer and one shared token tensor
    # -------------------------------------------------------------------------
    tokenizer = load_base_tokenizer(
        args.base_model
    )

    input_ids = tokenize_corpus(
        tokenizer,
        corpus,
    )

    corpus_info[
        "token_count"
    ] = int(
        input_ids.shape[1]
    )

    print(
        f"Base-tokenizer tokens: "
        f"{input_ids.shape[1]:,}"
    )

    results: dict[
        str,
        dict[str, Any],
    ] = {}

    write_result(
        output_file=output_file,
        args=args,
        corpus_info=corpus_info,
        tokenizer=tokenizer,
        results=results,
        complete=False,
    )

    # -------------------------------------------------------------------------
    # One checkpoint at a time
    # -------------------------------------------------------------------------
    for kind in MODEL_ORDER:
        print(
            f"\n=== {kind} ==="
        )

        model = None

        try:
            model, reference = load_model(
                kind,
                args,
                tokenizer,
            )

            measurement = measure_perplexity(
                model=model,
                input_ids=input_ids,
                seq_len=args.seq_len,
                stride=args.stride,
                model_kind=kind,
            )

            measurement[
                "model_reference"
            ] = reference

            results[
                kind
            ] = measurement

            print(
                f"{kind}: "
                f"PPL {measurement['perplexity']:.4f} | "
                f"NLL {measurement['mean_nll']:.4f} | "
                f"{measurement['scored_tokens']:,} scored tokens"
            )

            write_result(
                output_file=output_file,
                args=args,
                corpus_info=corpus_info,
                tokenizer=tokenizer,
                results=results,
                complete=False,
            )

        finally:
            if model is not None:
                del model

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Final output
    # -------------------------------------------------------------------------
    write_result(
        output_file=output_file,
        args=args,
        corpus_info=corpus_info,
        tokenizer=tokenizer,
        results=results,
        complete=True,
    )

    print(
        "\n"
        + "=" * 64
    )

    print(
        f"{'model':<16}"
        f"{'perplexity':>16}"
        f"{'mean_nll':>16}"
    )

    print(
        "-" * 64
    )

    for kind in MODEL_ORDER:
        item = results[
            kind
        ]

        print(
            f"{kind:<16}"
            f"{item['perplexity']:>16.4f}"
            f"{item['mean_nll']:>16.4f}"
        )

    print(
        "=" * 64
    )

    print(
        f"Saved to {output_file}"
    )


if __name__ == "__main__":
    main()