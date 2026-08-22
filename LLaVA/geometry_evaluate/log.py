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


MODEL_COMPARISONS = (
    "stage2",
    "stage2_only",
)

LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize full training loss history and layer-wise weight changes."
    )

    # Loss logs / training outputs
    parser.add_argument("--stage1-log", type=Path, required=True)
    parser.add_argument("--stage2-log", type=Path, required=True)
    parser.add_argument("--stage2-only-log", type=Path, required=True)

    # Models
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--stage2-dir", type=Path, required=True)
    parser.add_argument("--stage2-only-dir", type=Path, required=True)

    # Output
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-file", default="log_results.md")

    parser.add_argument(
        "--local-files-only",
        action="store_true",
    )

    return parser.parse_args()


# =============================================================================
# Loss history
# =============================================================================

def read_trainer_state(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    history = payload.get("log_history")

    if not isinstance(history, list):
        raise ValueError(
            f"No valid log_history in {path}"
        )

    return [
        item
        for item in history
        if isinstance(item, dict)
    ]


def max_logged_step(history: list[dict[str, Any]]) -> int:
    steps = []

    for item in history:
        try:
            if item.get("step") is not None:
                steps.append(int(item["step"]))
        except (TypeError, ValueError):
            pass

    return max(steps) if steps else -1


def find_complete_trainer_state(path: Path) -> Path:
    """
    If checkpoint-*/trainer_state.json files all exist, choose the one
    containing the highest global step.

    This avoids concatenating resumed histories and duplicating early steps.
    """

    if path.is_file():
        if path.name != "trainer_state.json":
            raise ValueError(
                f"Expected trainer_state.json, got {path}"
            )
        return path

    candidates = list(
        path.rglob("trainer_state.json")
    )

    if not candidates:
        raise FileNotFoundError(
            f"No trainer_state.json found under {path}"
        )

    scored = []

    for candidate in candidates:
        try:
            history = read_trainer_state(candidate)
            scored.append(
                (
                    max_logged_step(history),
                    len(history),
                    candidate,
                )
            )
        except Exception:
            continue

    if not scored:
        raise RuntimeError(
            f"No readable trainer_state.json under {path}"
        )

    scored.sort(
        key=lambda x: (x[0], x[1]),
        reverse=True,
    )

    return scored[0][2]


def extract_loss_history(
    name: str,
    path: Path,
) -> dict[str, Any]:
    state_path = find_complete_trainer_state(path)
    history = read_trainer_state(state_path)

    records = []

    for item in history:
        common = {
            "step": item.get("step"),
            "epoch": item.get("epoch"),
            "learning_rate": item.get("learning_rate"),
        }

        if item.get("loss") is not None:
            records.append(
                {
                    **common,
                    "type": "train",
                    "loss": item["loss"],
                }
            )

        if item.get("eval_loss") is not None:
            records.append(
                {
                    **common,
                    "type": "eval",
                    "loss": item["eval_loss"],
                }
            )

    return {
        "name": name,
        "source": str(state_path),
        "records": records,
    }


def write_loss_csv(
    path: Path,
    result: dict[str, Any],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "step",
                "epoch",
                "type",
                "loss",
                "learning_rate",
            ),
        )

        writer.writeheader()

        # Every recorded loss step is written.
        for record in result["records"]:
            writer.writerow(record)


# =============================================================================
# Checkpoint access
# =============================================================================

def resolve_checkpoint(
    reference: str | Path,
    local_files_only: bool,
) -> Path:
    path = Path(str(reference))

    if path.exists():
        return path.resolve()

    from huggingface_hub import snapshot_download

    resolved = snapshot_download(
        repo_id=str(reference),
        allow_patterns=[
            "*.json",
            "model*.safetensors",
            "pytorch_model*.bin",
        ],
        local_files_only=local_files_only,
    )

    return Path(resolved)


class CheckpointStore:
    """
    Tensor-by-tensor checkpoint reader.

    Only one .bin shard is kept in RAM at a time.
    Safetensors tensors are read individually.
    """

    def __init__(self, root: Path):
        self.root = root
        self.weight_map: dict[str, Path] = {}

        self._cached_bin_path: Path | None = None
        self._cached_bin: dict[str, torch.Tensor] | None = None

        self._build_index()

    def _build_index(self) -> None:
        for index_name in (
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ):
            index_path = self.root / index_name

            if index_path.exists():
                payload = json.loads(
                    index_path.read_text(encoding="utf-8")
                )

                for key, filename in payload["weight_map"].items():
                    self.weight_map[key] = (
                        self.root / filename
                    )

                return

        safetensors = sorted(
            self.root.glob("model*.safetensors")
        )

        if safetensors:
            from safetensors import safe_open

            for file in safetensors:
                with safe_open(
                    str(file),
                    framework="pt",
                    device="cpu",
                ) as handle:
                    for key in handle.keys():
                        self.weight_map[key] = file

            return

        bins = sorted(
            self.root.glob("pytorch_model*.bin")
        )

        if not bins:
            raise FileNotFoundError(
                f"No checkpoint weights found in {self.root}"
            )

        for file in bins:
            state = self._load_bin(file)

            for key in state:
                self.weight_map[key] = file

        self.clear()

    def _load_bin(
        self,
        path: Path,
    ) -> dict[str, torch.Tensor]:
        if (
            self._cached_bin_path == path
            and self._cached_bin is not None
        ):
            return self._cached_bin

        self.clear()

        try:
            state = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except TypeError:
            state = torch.load(
                path,
                map_location="cpu",
            )

        if (
            isinstance(state, dict)
            and "state_dict" in state
        ):
            state = state["state_dict"]

        self._cached_bin_path = path
        self._cached_bin = state

        return state

    def get(
        self,
        key: str,
    ) -> torch.Tensor:
        file = self.weight_map[key]

        if file.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(
                str(file),
                framework="pt",
                device="cpu",
            ) as handle:
                return handle.get_tensor(key)

        return self._load_bin(file)[key]

    def keys(self) -> set[str]:
        return set(self.weight_map)

    def clear(self) -> None:
        self._cached_bin_path = None
        self._cached_bin = None
        gc.collect()


# =============================================================================
# Weight change
# =============================================================================

def get_layer(name: str) -> int | None:
    match = LAYER_PATTERN.search(name)

    return int(match.group(1)) if match else None


def empty_norms() -> dict[str, float]:
    return {
        "base_sq": 0.0,
        "delta_sq": 0.0,
        "params": 0,
    }


def add_tensor_change(
    stats: dict[str, float],
    base: torch.Tensor,
    target: torch.Tensor,
) -> None:
    base = base.detach().float()
    target = target.detach().float()

    delta = target - base

    stats["base_sq"] += float(
        torch.sum(
            base.square(),
            dtype=torch.float64,
        )
    )

    stats["delta_sq"] += float(
        torch.sum(
            delta.square(),
            dtype=torch.float64,
        )
    )

    stats["params"] += base.numel()


def finalize_change(
    stats: dict[str, float],
) -> dict[str, Any]:
    base_l2 = math.sqrt(
        stats["base_sq"]
    )

    delta_l2 = math.sqrt(
        stats["delta_sq"]
    )

    relative = (
        delta_l2 / base_l2
        if base_l2 > 0
        else None
    )

    return {
        "params": int(stats["params"]),
        "base_l2": base_l2,
        "delta_l2": delta_l2,
        "relative_change": relative,
        "relative_change_percent": (
            relative * 100
            if relative is not None
            else None
        ),
    }


def compare_weights(
    name: str,
    base_dir: Path,
    target_dir: Path,
) -> dict[str, Any]:
    print(f"\nWeight comparison: Base -> {name}")

    base = CheckpointStore(base_dir)
    target = CheckpointStore(target_dir)

    common_keys = sorted(
        base.keys() & target.keys()
    )

    overall = empty_norms()

    layer_stats = defaultdict(
        empty_norms
    )

    compared_tensors = 0

    for index, key in enumerate(
        common_keys,
        start=1,
    ):
        if index % 100 == 0:
            print(
                f"  {index:,}/{len(common_keys):,} tensors"
            )

        layer = get_layer(key)

        # We only need LLM transformer layers for the layer-wise result.
        # Overall model change still uses every common floating tensor.
        base_tensor = base.get(key)
        target_tensor = target.get(key)

        if base_tensor.shape != target_tensor.shape:
            continue

        if not (
            base_tensor.is_floating_point()
            and target_tensor.is_floating_point()
        ):
            continue

        add_tensor_change(
            overall,
            base_tensor,
            target_tensor,
        )

        if layer is not None:
            add_tensor_change(
                layer_stats[layer],
                base_tensor,
                target_tensor,
            )

        compared_tensors += 1

        del base_tensor
        del target_tensor

    base.clear()
    target.clear()

    return {
        "comparison": f"base_to_{name}",
        "compared_tensors": compared_tensors,
        "overall": finalize_change(overall),
        "layers": {
            str(layer): finalize_change(stats)
            for layer, stats in sorted(
                layer_stats.items()
            )
        },
    }


# =============================================================================
# Formatting
# =============================================================================

def fmt_loss(value: Any) -> str:
    if value is None:
        return "-"

    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_percent(value: float | None) -> str:
    if value is None:
        return "-"

    return f"{value:.4f}%"


# =============================================================================
# Markdown
# =============================================================================

def write_markdown(
    *,
    output_file: Path,
    losses: dict[str, Any],
    weights: dict[str, Any],
) -> None:
    lines = [
        "# Training Log / Weight Change",
        "",
        "## Weight change definition",
        "",
        (
            r"$\mathrm{Relative\ Change} = "
            r"\frac{\|W_{target}-W_{base}\|_2}"
            r"{\|W_{base}\|_2}\times100$"
        ),
        "",
        (
            "Weight comparisons are Base→Stage2 and "
            "Base→Stage2-only."
        ),
        "",
    ]

    # =========================================================================
    # Full loss histories
    # =========================================================================
    for run in (
        "stage1",
        "stage2",
        "stage2_only",
    ):
        result = losses[run]

        lines.extend(
            [
                f"## {run} — Full Loss History",
                "",
                f"Source: `{result['source']}`",
                "",
                "| Step | Epoch | Type | Loss | Learning Rate |",
                "|---:|---:|---|---:|---:|",
            ]
        )

        # No last-N filtering.
        for record in result["records"]:
            lines.append(
                f"| {record.get('step', '-')} "
                f"| {fmt_loss(record.get('epoch'))} "
                f"| {record['type']} "
                f"| {fmt_loss(record['loss'])} "
                f"| {fmt_loss(record.get('learning_rate'))} |"
            )

        lines.append("")

    # =========================================================================
    # Whole-model change
    # =========================================================================
    lines.extend(
        [
            "## Whole-model Weight Change",
            "",
            "| Model | Relative Change | ΔW L2 | Base W L2 |",
            "|---|---:|---:|---:|",
        ]
    )

    for name in MODEL_COMPARISONS:
        result = weights[name]["overall"]

        lines.append(
            f"| {name} "
            f"| {fmt_percent(result['relative_change_percent'])} "
            f"| {result['delta_l2']:.6g} "
            f"| {result['base_l2']:.6g} |"
        )

    lines.append("")

    # =========================================================================
    # Layer-wise comparison
    # =========================================================================
    lines.extend(
        [
            "## LLM Layer-wise Weight Change",
            "",
            "| Layer | Stage2 | Stage2-only |",
            "|---:|---:|---:|",
        ]
    )

    layer_ids = sorted(
        {
            int(layer)
            for name in MODEL_COMPARISONS
            for layer in weights[name]["layers"]
        }
    )

    for layer in layer_ids:
        stage2 = weights[
            "stage2"
        ][
            "layers"
        ].get(str(layer))

        stage2_only = weights[
            "stage2_only"
        ][
            "layers"
        ].get(str(layer))

        lines.append(
            f"| L{layer} "
            f"| {fmt_percent(stage2['relative_change_percent'] if stage2 else None)} "
            f"| {fmt_percent(stage2_only['relative_change_percent'] if stage2_only else None)} |"
        )

    lines.append("")

    output_file.write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        args.output_dir
        / args.output_file
    )

    # =========================================================================
    # Loss
    # =========================================================================
    losses = {
        "stage1": extract_loss_history(
            "stage1",
            args.stage1_log,
        ),
        "stage2": extract_loss_history(
            "stage2",
            args.stage2_log,
        ),
        "stage2_only": extract_loss_history(
            "stage2_only",
            args.stage2_only_log,
        ),
    }

    for run, result in losses.items():
        write_loss_csv(
            args.output_dir
            / f"loss_{run}.csv",
            result,
        )

        print(
            f"{run}: "
            f"{len(result['records']):,} recorded loss points"
        )

    # =========================================================================
    # Weight
    # =========================================================================
    print("\nResolving Base checkpoint...")

    base_dir = resolve_checkpoint(
        args.base_model,
        args.local_files_only,
    )

    weights = {
        "stage2": compare_weights(
            "stage2",
            base_dir,
            args.stage2_dir,
        ),
        "stage2_only": compare_weights(
            "stage2_only",
            base_dir,
            args.stage2_only_dir,
        ),
    }

    # =========================================================================
    # Machine-readable output
    # =========================================================================
    (
        args.output_dir
        / "log_results.json"
    ).write_text(
        json.dumps(
            {
                "loss": losses,
                "weights": weights,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # =========================================================================
    # Human-readable output
    # =========================================================================
    write_markdown(
        output_file=output_file,
        losses=losses,
        weights=weights,
    )

    print("\nLog analysis completed.")
    print(f"Summary: {output_file}")


if __name__ == "__main__":
    main()