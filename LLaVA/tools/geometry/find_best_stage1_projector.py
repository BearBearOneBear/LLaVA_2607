from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Stage 1 output path"""

    parser = argparse.ArgumentParser(
        description="Find the best Stage 1 projector checkpoint."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Stage 1 training output directory.",
    )

    return parser.parse_args()


def find_projector_path(
    output_dir: Path,
    step: int,
    final_step: int,
) -> Path | None:
    """projector"""

    candidates = [
        output_dir / f"checkpoint-{step}" / "mm_projector.bin",
        output_dir / "mm_projector" / f"checkpoint-{step}.bin",
    ]

    if step == final_step:
        candidates.append(output_dir / "mm_projector.bin")

    for path in candidates:
        if path.is_file():
            return path.resolve()

    return None


def main() -> None:
    args = parse_args()

    trainer_state_path = args.output_dir / "trainer_state.json"

    with trainer_state_path.open("r", encoding="utf-8") as file:
        trainer_state: dict[str, Any] = json.load(file)

    final_step = int(trainer_state["global_step"])

    evaluation_results: list[dict[str, Any]] = []

    for log in trainer_state["log_history"]:
        if "eval_loss" not in log or "step" not in log:
            continue

        step = int(log["step"])
        eval_loss = float(log["eval_loss"])

        projector_path = find_projector_path(
            output_dir=args.output_dir,
            step=step,
            final_step=final_step,
        )

        evaluation_results.append(
            {
                "step": step,
                "eval_loss": eval_loss,
                "projector_path": projector_path,
            }
        )

    available_results = [
        result
        for result in evaluation_results
        if result["projector_path"] is not None
    ]

    best_result = min(
        available_results,
        key=lambda result: result["eval_loss"],
    )

    result_path = args.output_dir / "best_stage1_projector.json"

    result_data = {
        "step": best_result["step"],
        "eval_loss": best_result["eval_loss"],
        "projector_path": str(best_result["projector_path"]),
    }

    with result_path.open("w", encoding="utf-8") as file:
        json.dump(
            result_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("Best Stage 1 projector found.")
    print(f"Step: {best_result['step']}")
    print(f"Evaluation loss: {best_result['eval_loss']:.6f}")
    print(f"Projector path: {best_result['projector_path']}")
    print(f"Result saved to: {result_path}")


if __name__ == "__main__":
    main()