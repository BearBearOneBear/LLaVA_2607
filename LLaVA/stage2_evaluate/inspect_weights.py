#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None


LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
KNOWN_MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Streaming weight-delta audit for Stage 2. Compares LLM weights against "
            "the original LLaVA-1.5 checkpoint and the multimodal projector against Stage 1."
        )
    )
    parser.add_argument("--base-model", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--stage1-dir", type=Path, default=Path("./checkpoints/geometry_stage1"))
    parser.add_argument("--stage2-dir", type=Path, default=Path("./checkpoints/geometry_stage2"))
    parser.add_argument("--output-dir", type=Path, default=Path("./stage2_evaludate/results/weight_audit"))
    parser.add_argument("--sample-per-tensor", type=int, default=4096)
    parser.add_argument("--global-sample-limit", type=int, default=750000)
    parser.add_argument(
        "--svd-layers",
        default="0,8,16,24,31",
        help="Comma-separated LLM layers for optional truncated SVD diagnostics.",
    )
    parser.add_argument(
        "--svd-modules",
        default="self_attn.q_proj,self_attn.o_proj,mlp.down_proj",
    )
    parser.add_argument("--svd-topk", type=int, default=0, help="0 disables expensive LLM SVD.")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def resolve_checkpoint_dir(reference: str | Path) -> Path:
    path = Path(reference).expanduser()
    if path.exists():
        return path.resolve()
    if snapshot_download is None:
        raise RuntimeError(
            f"{reference} is not a local path and huggingface_hub is unavailable."
        )
    try:
        cached = snapshot_download(
            repo_id=str(reference),
            local_files_only=True,
            allow_patterns=["*.json", "*.bin", "*.safetensors"],
        )
    except Exception:
        cached = snapshot_download(
            repo_id=str(reference),
            allow_patterns=["*.json", "*.bin", "*.safetensors"],
        )
    return Path(cached)


class CheckpointReader:
    """Read one tensor at a time from HF safetensors or PyTorch shards."""

    def __init__(self, root: Path):
        self.root = root
        self.key_to_file: dict[str, Path] = {}
        self.format: str | None = None
        self._bin_cache_file: Path | None = None
        self._bin_cache: dict[str, torch.Tensor] | None = None
        self._build_index()

    def _build_index(self) -> None:
        safe_index = self.root / "model.safetensors.index.json"
        bin_index = self.root / "pytorch_model.bin.index.json"

        if safe_index.exists():
            if safe_open is None:
                raise RuntimeError("safetensors is required to read model.safetensors shards.")
            data = json.loads(safe_index.read_text(encoding="utf-8"))
            self.key_to_file = {
                key: self.root / filename for key, filename in data["weight_map"].items()
            }
            self.format = "safetensors"
            return

        if bin_index.exists():
            data = json.loads(bin_index.read_text(encoding="utf-8"))
            self.key_to_file = {
                key: self.root / filename for key, filename in data["weight_map"].items()
            }
            self.format = "bin"
            return

        safe_files = sorted(self.root.glob("*.safetensors"))
        if safe_files:
            if safe_open is None:
                raise RuntimeError("safetensors is required to read model.safetensors.")
            self.format = "safetensors"
            for path in safe_files:
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        self.key_to_file[key] = path
            return

        bin_files = sorted(self.root.glob("pytorch_model*.bin"))
        if bin_files:
            self.format = "bin"
            for path in bin_files:
                state = torch.load(path, map_location="cpu")
                for key in state:
                    self.key_to_file[key] = path
                del state
                gc.collect()
            return

        raise FileNotFoundError(f"No HF model weight files found under {self.root}")

    def keys(self) -> set[str]:
        return set(self.key_to_file)

    def get(self, key: str) -> torch.Tensor:
        path = self.key_to_file[key]
        if self.format == "safetensors":
            assert safe_open is not None
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                return handle.get_tensor(key)

        if self._bin_cache_file != path:
            self._bin_cache = torch.load(path, map_location="cpu")
            self._bin_cache_file = path
        assert self._bin_cache is not None
        return self._bin_cache[key]

    def clear_cache(self) -> None:
        self._bin_cache = None
        self._bin_cache_file = None
        gc.collect()


def canonical_key(key: str) -> str:
    for prefix in ("base_model.model.", "module."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def make_key_map(keys: set[str]) -> dict[str, str]:
    out = {}
    for key in keys:
        out[canonical_key(key)] = key
    return out


def layer_of(key: str) -> int | None:
    match = LAYER_RE.search(key)
    return int(match.group(1)) if match else None


def module_of(key: str) -> str:
    for name in KNOWN_MODULES:
        if name in key:
            return name
    if "input_layernorm" in key:
        return "input_layernorm"
    if "post_attention_layernorm" in key:
        return "post_attention_layernorm"
    if "embed_tokens" in key:
        return "embed_tokens"
    if "lm_head" in key:
        return "lm_head"
    if "norm" in key:
        return "final_norm"
    return "other"


def is_llm_tensor(key: str, tensor: torch.Tensor) -> bool:
    if not tensor.is_floating_point():
        return False
    if "vision_tower" in key or "mm_projector" in key or "vision_resampler" in key:
        return False
    return (
        "layers." in key
        or "embed_tokens" in key
        or "lm_head" in key
        or key.endswith("norm.weight")
    )


def deterministic_sample_abs(delta: torch.Tensor, n: int) -> torch.Tensor:
    flat = delta.detach().reshape(-1)
    if flat.numel() <= n:
        return flat.abs().float().cpu()
    # Evenly spaced deterministic sample; avoids allocating randperm for huge tensors.
    idx = torch.linspace(0, flat.numel() - 1, steps=n, dtype=torch.float64)
    idx = idx.round().long()
    return flat[idx].abs().float().cpu()


def tensor_stats(base: torch.Tensor, stage2: torch.Tensor, sample_n: int) -> tuple[dict[str, Any], torch.Tensor]:
    if base.shape != stage2.shape:
        raise ValueError(f"Shape mismatch: {tuple(base.shape)} vs {tuple(stage2.shape)}")

    b = base.float()
    s = stage2.float()
    d = s - b

    base_sq = torch.sum(b * b).item()
    stage2_sq = torch.sum(s * s).item()
    delta_sq = torch.sum(d * d).item()
    dot = torch.sum(b * s).item()
    delta_sum = torch.sum(d).item()
    delta_sum_sq = torch.sum(d * d).item()
    count = d.numel()

    base_norm = math.sqrt(base_sq)
    stage2_norm = math.sqrt(stage2_sq)
    delta_norm = math.sqrt(delta_sq)
    denom = base_norm * stage2_norm
    cosine = dot / denom if denom else float("nan")

    sampled = deterministic_sample_abs(d, sample_n)
    quantiles = torch.quantile(sampled, torch.tensor([0.50, 0.90, 0.95, 0.99])).tolist()

    stats = {
        "numel": count,
        "base_norm": base_norm,
        "stage2_norm": stage2_norm,
        "delta_norm": delta_norm,
        "relative_delta_norm": delta_norm / base_norm if base_norm else float("nan"),
        "weight_cosine": cosine,
        "delta_mean": delta_sum / count if count else 0.0,
        "delta_std": math.sqrt(max(delta_sum_sq / count - (delta_sum / count) ** 2, 0.0)) if count else 0.0,
        "delta_abs_p50_sample": quantiles[0],
        "delta_abs_p90_sample": quantiles[1],
        "delta_abs_p95_sample": quantiles[2],
        "delta_abs_p99_sample": quantiles[3],
        "delta_abs_max": torch.max(torch.abs(d)).item() if count else 0.0,
        "delta_sq_sum": delta_sq,
        "base_sq_sum": base_sq,
        "stage2_sq_sum": stage2_sq,
        "dot_sum": dot,
        "delta_sum": delta_sum,
        "delta_sum_sq": delta_sum_sq,
    }
    del b, s, d
    return stats, sampled


def aggregate_rows(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)

    output = []
    for group, items in sorted(groups.items(), key=lambda x: str(x[0])):
        count = sum(int(x["numel"]) for x in items)
        delta_sq = sum(float(x["delta_sq_sum"]) for x in items)
        base_sq = sum(float(x["base_sq_sum"]) for x in items)
        stage2_sq = sum(float(x["stage2_sq_sum"]) for x in items)
        dot = sum(float(x["dot_sum"]) for x in items)
        dsum = sum(float(x["delta_sum"]) for x in items)
        dsum_sq = sum(float(x["delta_sum_sq"]) for x in items)
        base_norm = math.sqrt(base_sq)
        stage2_norm = math.sqrt(stage2_sq)
        delta_norm = math.sqrt(delta_sq)
        output.append(
            {
                group_key: group,
                "tensor_count": len(items),
                "numel": count,
                "base_norm": base_norm,
                "stage2_norm": stage2_norm,
                "delta_norm": delta_norm,
                "relative_delta_norm": delta_norm / base_norm if base_norm else float("nan"),
                "weight_cosine": dot / (base_norm * stage2_norm) if base_norm and stage2_norm else float("nan"),
                "delta_mean": dsum / count if count else 0.0,
                "delta_std": math.sqrt(max(dsum_sq / count - (dsum / count) ** 2, 0.0)) if count else 0.0,
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def compare_projector(stage1_dir: Path, stage2_reader: CheckpointReader) -> list[dict[str, Any]]:
    stage1_path = stage1_dir / "mm_projector.bin"
    if not stage1_path.exists():
        raise FileNotFoundError(f"Stage-1 projector not found: {stage1_path}")
    state1 = torch.load(stage1_path, map_location="cpu")

    stage2_keys = stage2_reader.keys()
    rows = []
    for key1, tensor1 in sorted(state1.items()):
        if not tensor1.is_floating_point():
            continue
        suffix = key1.split("mm_projector", 1)[-1]
        candidates = [k for k in stage2_keys if "mm_projector" in k and k.endswith(suffix)]
        if len(candidates) != 1:
            print(f"WARNING: projector key {key1} matched {len(candidates)} Stage-2 keys; skipped.")
            continue
        key2 = candidates[0]
        tensor2 = stage2_reader.get(key2)
        stats, _ = tensor_stats(tensor1, tensor2, min(8192, tensor1.numel()))
        rows.append({"stage1_key": key1, "stage2_key": key2, **stats})
    del state1
    gc.collect()
    return rows


def optional_svd(
    base_reader: CheckpointReader,
    stage2_reader: CheckpointReader,
    key_pairs: dict[str, tuple[str, str]],
    selected_layers: set[int],
    selected_modules: set[str],
    topk: int,
) -> list[dict[str, Any]]:
    if topk <= 0:
        return []
    rows = []
    for ckey, (bkey, skey) in sorted(key_pairs.items()):
        layer = layer_of(ckey)
        module = module_of(ckey)
        if layer not in selected_layers or module not in selected_modules:
            continue
        b = base_reader.get(bkey)
        s = stage2_reader.get(skey)
        if b.ndim != 2 or min(b.shape) < 2:
            continue
        d = s.float() - b.float()
        q = min(max(topk + 4, topk), min(d.shape))
        print(f"Running truncated SVD: layer={layer} module={module} shape={tuple(d.shape)} q={q}")
        # torch.svd_lowrank is substantially cheaper than full SVD for these 7B matrices.
        _, singular_values, _ = torch.svd_lowrank(d, q=q, niter=2)
        sv = torch.sort(singular_values, descending=True).values[:topk]
        frob_sq = torch.sum(d * d).item()
        top_energy = torch.sum(sv * sv).item()
        sigma1 = sv[0].item() if len(sv) else 0.0
        rows.append(
            {
                "layer": layer,
                "module": module,
                "key": ckey,
                "shape": "x".join(map(str, d.shape)),
                "topk": len(sv),
                "singular_values": json.dumps([float(x) for x in sv.tolist()]),
                "topk_energy_fraction": top_energy / frob_sq if frob_sq else 0.0,
                "approx_stable_rank": frob_sq / (sigma1 * sigma1) if sigma1 else 0.0,
            }
        )
        del b, s, d, singular_values, sv
        gc.collect()
    return rows


def make_plots(output_dir: Path, layer_rows: list[dict[str, Any]], abs_samples: torch.Tensor) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping plots.")
        return

    numeric_layers = [r for r in layer_rows if str(r["layer"]).isdigit()]
    numeric_layers.sort(key=lambda r: int(r["layer"]))
    if numeric_layers:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(
            [int(r["layer"]) for r in numeric_layers],
            [float(r["relative_delta_norm"]) for r in numeric_layers],
            marker="o",
        )
        ax.set_xlabel("LLM layer")
        ax.set_ylabel("||ΔW|| / ||W_base||")
        ax.set_title("Stage-2 relative weight change by layer")
        fig.tight_layout()
        fig.savefig(output_dir / "layer_relative_delta_norm.png", dpi=160)
        plt.close(fig)

    if abs_samples.numel():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        positive = abs_samples[abs_samples > 0].numpy()
        if len(positive):
            ax.hist(positive, bins=100)
            ax.set_xscale("log")
        ax.set_xlabel("|Δw| (sampled)")
        ax.set_ylabel("count")
        ax.set_title("Sampled Stage-2 absolute weight changes")
        fig.tight_layout()
        fig.savefig(output_dir / "delta_abs_histogram.png", dpi=160)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = resolve_checkpoint_dir(args.base_model)
    stage2_dir = resolve_checkpoint_dir(args.stage2_dir)
    print(f"Base checkpoint directory: {base_dir}")
    print(f"Stage-2 checkpoint directory: {stage2_dir}")

    base_reader = CheckpointReader(base_dir)
    stage2_reader = CheckpointReader(stage2_dir)
    base_map = make_key_map(base_reader.keys())
    stage2_map = make_key_map(stage2_reader.keys())
    common = sorted(set(base_map) & set(stage2_map))

    tensor_rows: list[dict[str, Any]] = []
    sample_chunks: list[torch.Tensor] = []
    sample_total = 0
    key_pairs: dict[str, tuple[str, str]] = {}

    for ckey in common:
        bkey, skey = base_map[ckey], stage2_map[ckey]
        base_tensor = base_reader.get(bkey)
        if not is_llm_tensor(ckey, base_tensor):
            continue
        stage2_tensor = stage2_reader.get(skey)
        if base_tensor.shape != stage2_tensor.shape:
            print(f"WARNING: shape mismatch for {ckey}; skipped.")
            continue

        stats, sampled = tensor_stats(base_tensor, stage2_tensor, args.sample_per_tensor)
        layer = layer_of(ckey)
        row = {
            "key": ckey,
            "layer": layer if layer is not None else "other",
            "module": module_of(ckey),
            "shape": "x".join(map(str, base_tensor.shape)),
            **stats,
        }
        tensor_rows.append(row)
        key_pairs[ckey] = (bkey, skey)

        if sample_total < args.global_sample_limit:
            remaining = args.global_sample_limit - sample_total
            part = sampled[:remaining]
            sample_chunks.append(part)
            sample_total += part.numel()

        del base_tensor, stage2_tensor, sampled
        gc.collect()

    if not tensor_rows:
        raise RuntimeError(
            "No common LLM tensors were found. Check whether the Stage-2 directory contains the full HF model."
        )

    layer_rows = aggregate_rows(tensor_rows, "layer")
    module_rows = aggregate_rows(tensor_rows, "module")
    projector_rows = compare_projector(args.stage1_dir, stage2_reader)

    selected_layers = {int(x) for x in args.svd_layers.split(",") if x.strip()}
    selected_modules = {x.strip() for x in args.svd_modules.split(",") if x.strip()}
    svd_rows = optional_svd(
        base_reader,
        stage2_reader,
        key_pairs,
        selected_layers,
        selected_modules,
        args.svd_topk,
    )

    write_csv(args.output_dir / "llm_tensor_delta.csv", tensor_rows)
    write_csv(args.output_dir / "llm_layer_delta.csv", layer_rows)
    write_csv(args.output_dir / "llm_module_delta.csv", module_rows)
    write_csv(args.output_dir / "projector_stage1_to_stage2_delta.csv", projector_rows)
    if svd_rows:
        write_csv(args.output_dir / "llm_selected_svd.csv", svd_rows)

    samples = torch.cat(sample_chunks) if sample_chunks else torch.empty(0)
    q = (
        torch.quantile(samples, torch.tensor([0.50, 0.90, 0.95, 0.99])).tolist()
        if samples.numel()
        else [None] * 4
    )

    summary = {
        "base_model": args.base_model,
        "base_checkpoint_dir": str(base_dir),
        "stage1_projector": str(args.stage1_dir / "mm_projector.bin"),
        "stage2_checkpoint_dir": str(stage2_dir),
        "llm_tensor_count": len(tensor_rows),
        "llm_layer_count": len({r["layer"] for r in tensor_rows if r["layer"] != "other"}),
        "projector_tensor_count": len(projector_rows),
        "sampled_abs_delta_count": int(samples.numel()),
        "sampled_abs_delta_quantiles": {
            "p50": q[0],
            "p90": q[1],
            "p95": q[2],
            "p99": q[3],
        },
        "svd_note": (
            "LLM SVD is disabled by default. When enabled, torch.svd_lowrank is used on selected "
            "layers/modules only. topk_energy_fraction and approx_stable_rank are diagnostics, not full-rank estimates."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not args.no_plots:
        make_plots(args.output_dir, layer_rows, samples)

    print(f"Weight audit completed: {args.output_dir}")


if __name__ == "__main__":
    main()
