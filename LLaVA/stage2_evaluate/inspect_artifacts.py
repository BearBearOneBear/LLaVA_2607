#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect saved Stage-1/Stage-2 training artifacts and trainer state.")
    parser.add_argument("--stage1-dir", type=Path, default=Path("./checkpoints/geometry_stage1"))
    parser.add_argument("--stage2-dir", type=Path, default=Path("./checkpoints/geometry_stage2"))
    parser.add_argument("--logs-root", type=Path, default=Path("./logs/geometry_pipeline"))
    parser.add_argument("--output", type=Path, default=Path("./stage2_evaludate/results/integrity.json"))
    return parser.parse_args()


def file_summary(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def find_model_weights(root: Path) -> list[str]:
    patterns = ["model*.safetensors", "pytorch_model*.bin"]
    files = []
    for pattern in patterns:
        files.extend(str(p) for p in sorted(root.glob(pattern)))
    return files


def latest_pipeline_log(logs_root: Path) -> Path | None:
    candidates = sorted(logs_root.glob("*/05_stage2_training.log")) if logs_root.exists() else []
    return candidates[-1] if candidates else None


def main() -> None:
    args = parse_args()
    state_path = args.stage2_dir / "trainer_state.json"
    trainer_state = None
    if state_path.exists():
        trainer_state = json.loads(state_path.read_text(encoding="utf-8"))

    train_history = []
    eval_history = []
    if trainer_state:
        for item in trainer_state.get("log_history", []):
            if "loss" in item:
                train_history.append(
                    {
                        "step": item.get("step"),
                        "epoch": item.get("epoch"),
                        "loss": item.get("loss"),
                        "learning_rate": item.get("learning_rate"),
                    }
                )
            if "eval_loss" in item:
                eval_history.append(
                    {
                        "step": item.get("step"),
                        "epoch": item.get("epoch"),
                        "eval_loss": item.get("eval_loss"),
                    }
                )

    pipeline_log = latest_pipeline_log(args.logs_root)
    result = {
        "stage1": {
            "directory": str(args.stage1_dir),
            "config": file_summary(args.stage1_dir / "config.json"),
            "projector": file_summary(args.stage1_dir / "mm_projector.bin"),
        },
        "stage2": {
            "directory": str(args.stage2_dir),
            "config": file_summary(args.stage2_dir / "config.json"),
            "trainer_state": file_summary(state_path),
            "model_weight_files": find_model_weights(args.stage2_dir),
            "checkpoint_directories": [str(p) for p in sorted(args.stage2_dir.glob("checkpoint-*"))],
        },
        "trainer_state_summary": {
            "available": trainer_state is not None,
            "best_model_checkpoint": trainer_state.get("best_model_checkpoint") if trainer_state else None,
            "best_metric": trainer_state.get("best_metric") if trainer_state else None,
            "global_step": trainer_state.get("global_step") if trainer_state else None,
            "epoch": trainer_state.get("epoch") if trainer_state else None,
            "train_loss_history": train_history,
            "eval_loss_history": eval_history,
        },
        "pipeline_log": file_summary(pipeline_log) if pipeline_log else {"path": None, "exists": False, "size_bytes": None},
        "notes": [
            "Stage 1 intentionally has no validation/best-checkpoint selection; the final mm_projector.bin is the Stage-1 artifact.",
            "Stage 2 calls trainer.save_state(), so trainer_state.json should contain log_history and best-model metadata if the output directory was preserved.",
            "run_stage1_stage2_pipeline.sh also tees Stage-2 stdout/stderr to logs/geometry_pipeline/<run_id>/05_stage2_training.log.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["trainer_state_summary"], ensure_ascii=False, indent=2))
    print(f"Integrity report saved to {args.output}")


if __name__ == "__main__":
    main()
