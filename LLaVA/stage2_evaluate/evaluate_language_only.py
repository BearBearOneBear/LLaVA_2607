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

# Resolved from the script, not the working directory, so the command above
# works whether it is run from the repository root or from this folder.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# llava lives at the repository root; without this, running from inside
# stage2_evaluate/ would fail to import it.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402


DEFAULT_BASE_MODEL = "liuhaotian/llava-v1.5-7b"
DEFAULT_STAGE2_DIR = REPO_ROOT / "checkpoints" / "geometry_stage2"
# Alongside evaluation_summary.json, so the whole audit sits in one folder.
DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "language_result.json"

HF_PARQUET = "wikitext-2-raw-v1/test-00000-of-00001.parquet"
HF_RESOLVE = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/" + HF_PARQUET
)
GITHUB_MIRROR = (
    "https://raw.githubusercontent.com/pytorch/examples/main/"
    "word_language_model/data/wikitext-2/test.txt"
)

# A shorter download is almost certainly an error page, and perplexity computed
# over an error page still returns a plausible-looking number.
MIN_CORPUS_CHARS = 200_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])

    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--stage2-dir", type=Path, default=DEFAULT_STAGE2_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--text-file",
        type=Path,
        default=None,
        help="Plain-text corpus. Skips the download entirely when given.",
    )
    parser.add_argument("--max-chars", type=int, default=1_000_000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def _from_datasets() -> str:
    """Parquet mirror. The bare "wikitext" id is a script, unsupported since datasets 3.0."""
    from datasets import load_dataset

    split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(str(x) for x in split["text"] if str(x).strip())


def _from_hub_file() -> str:
    """Direct parquet download, bypassing load_dataset's resolution."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(
        repo_id="Salesforce/wikitext", filename=HF_PARQUET, repo_type="dataset"
    )
    column = pq.read_table(path, columns=["text"]).column("text")
    return "\n\n".join(str(x) for x in column.to_pylist() if str(x).strip())


def _from_hub_url() -> str:
    """Plain HTTPS to the resolve endpoint, for when huggingface_hub itself fails."""
    import io
    import urllib.request
    import pyarrow.parquet as pq

    with urllib.request.urlopen(HF_RESOLVE, timeout=60) as response:
        payload = response.read()

    column = pq.read_table(io.BytesIO(payload), columns=["text"]).column("text")
    return "\n\n".join(str(x) for x in column.to_pylist() if str(x).strip())


def _from_github() -> str:
    """WikiText-2 as vendored in pytorch/examples.

    Tokenised rather than raw, so it carries <unk> and @-@ artefacts and its
    absolute perplexity is not comparable to the raw split's. It is fixed English
    text that both models read identically, which is all this comparison needs.
    """
    import urllib.request

    with urllib.request.urlopen(GITHUB_MIRROR, timeout=60) as response:
        return response.read().decode("utf-8")


CORPUS_ROUTES = (
    ("hf_datasets", _from_datasets, "wikitext-2-raw-v1:test"),
    ("hf_hub_file", _from_hub_file, "wikitext-2-raw-v1:test"),
    ("hf_url", _from_hub_url, "wikitext-2-raw-v1:test"),
    ("github_pytorch_examples", _from_github, "wikitext-2-v1:test (tokenised)"),
)


def load_corpus(args: argparse.Namespace) -> tuple[str, dict]:
    """Corpus text plus a fingerprint identifying exactly what was read."""

    def finish(text: str, source: str) -> tuple[str, dict]:
        available = len(text)
        if args.max_chars > 0:
            text = text[: args.max_chars]

        return text, {
            "source": source,
            "available_chars": available,
            "used_chars": len(text),
            "used_words": len(text.split()),
            # Fingerprints the text actually scored, so a later run can be
            # checked for having read the same bytes without storing a copy.
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }

    if args.text_file:
        return finish(
            args.text_file.read_text(encoding="utf-8", errors="replace"),
            str(args.text_file),
        )

    failures = []
    for name, loader, label in CORPUS_ROUTES:
        try:
            text = loader()

            if len(text) < MIN_CORPUS_CHARS:
                raise RuntimeError(
                    f"returned {len(text)} chars, below the {MIN_CORPUS_CHARS} "
                    "minimum; this is usually an error page rather than the corpus"
                )

            print(f"Corpus via {name}: {len(text):,} chars ({label})")
            return finish(text, f"{label} via {name}")

        except Exception as exc:  # noqa: BLE001 - report every route, then advise
            failures.append(f"  {name}: {type(exc).__name__}: {exc}")
            print(f"  route {name} failed: {type(exc).__name__}")

    raise RuntimeError(
        "Every corpus route failed:\n"
        + "\n".join(failures)
        + "\n\nPass --text-file with any sizeable English text. Only the gap "
          "between the two models is read, so the corpus identity does not "
          "matter as long as both see the same one."
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def load_model(kind: str, args: argparse.Namespace):
    from transformers import AutoTokenizer
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

    # One tokenizer for both, or the token counts differ and the perplexities
    # stop being comparable.
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    if kind == "base":
        reference = args.base_model
    else:
        config = args.stage2_dir / "config.json"
        if not config.exists():
            raise FileNotFoundError(f"Stage-2 config not found: {config}")
        reference = str(args.stage2_dir)

    # Both are LlavaLlamaForCausalLM checkpoints; a text-only forward never
    # builds the vision tower.
    model = LlavaLlamaForCausalLM.from_pretrained(
        reference,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    rows = model.get_input_embeddings().weight.shape[0]
    if rows < len(tokenizer):
        raise RuntimeError(
            f"{kind} has {rows} embedding rows against a tokenizer of "
            f"{len(tokenizer)}; the perplexities would not be comparable."
        )

    return tokenizer, model, reference


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def measure_perplexity(tokenizer, model, text: str, seq_len: int, stride: int) -> dict:
    """Sliding-window perplexity, scoring each token exactly once.

    Windows overlap so tokens near a boundary still have context, but only those
    past the previous window's end contribute; everything earlier is masked out.
    Counting the labels that survive the causal shift, rather than assuming
    target_len of them, keeps the token-weighted mean exact.
    """
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    total_tokens = input_ids.size(1)

    if total_tokens < 2:
        raise RuntimeError("Corpus tokenized to fewer than two tokens.")

    device = model.get_input_embeddings().weight.device
    nll_sum = 0.0
    counted = 0
    previous_end = 0

    for begin in tqdm(range(0, total_tokens, stride), desc="perplexity"):
        end = min(begin + seq_len, total_tokens)
        target_len = end - previous_end
        if target_len <= 0:
            break

        ids = input_ids[:, begin:end].to(device)
        target = ids.clone()
        target[:, :-target_len] = -100

        with torch.inference_mode():
            outputs = model(input_ids=ids, labels=target, use_cache=False)

        effective = int((target[:, 1:] != -100).sum().item())
        if effective:
            nll_sum += float(outputs.loss) * effective
            counted += effective

        previous_end = end
        if end == total_tokens:
            break

    mean_nll = nll_sum / counted
    return {
        "scored_tokens": counted,
        "total_tokens": total_tokens,
        "seq_len": seq_len,
        "stride": stride,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
    }


# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Repository root : {REPO_ROOT}")
    print(f"Stage-2 dir     : {args.stage2_dir}")
    print(f"Output          : {args.output}\n")

    corpus, corpus_info = load_corpus(args)
    print(
        f"Corpus: {corpus_info['source']} | "
        f"{corpus_info['used_chars']:,} chars | "
        f"sha256 {corpus_info['sha256'][:16]}"
    )

    results: dict[str, dict] = {}

    for kind in ("base", "stage2"):
        print(f"\n=== {kind} ===")
        tokenizer, model, reference = load_model(kind, args)

        measurement = measure_perplexity(
            tokenizer, model, corpus, args.seq_len, args.stride
        )
        measurement["model_reference"] = reference
        results[kind] = measurement

        print(
            f"{kind}: perplexity {measurement['perplexity']:.4f} "
            f"over {measurement['scored_tokens']:,} tokens"
        )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    base_ppl = results["base"]["perplexity"]
    stage2_ppl = results["stage2"]["perplexity"]

    summary = {
        "run": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tokenizer": args.base_model,
            "seq_len": args.seq_len,
            "stride": args.stride,
            "max_chars": args.max_chars,
        },
        "corpus": corpus_info,
        "models": results,
        "delta": {
            "stage2_minus_base": stage2_ppl - base_ppl,
            "relative_increase": stage2_ppl / base_ppl - 1.0,
        },
        "notes": [
            "Absolute perplexity depends on the corpus and tokenizer; only the "
            "difference between the two models is interpretable. corpus.sha256 "
            "identifies the exact text scored.",
            "base is llava-v1.5-7b, the checkpoint Stage 2 was trained from, so "
            "the delta isolates what Stage 1+2 cost rather than including "
            "LLaVA's own instruction tuning.",
        ],
    }

    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 52)
    print(f"{'model':<10} {'perplexity':>16}")
    print("-" * 52)
    print(f"{'base':<10} {base_ppl:>16.4f}")
    print(f"{'stage2':<10} {stage2_ppl:>16.4f}")
    print("-" * 52)
    print(
        f"ΔPPL (stage2 - base): {stage2_ppl - base_ppl:+.4f} "
        f"({(stage2_ppl / base_ppl - 1) * 100:+.2f}%)"
    )
    print("=" * 52)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
