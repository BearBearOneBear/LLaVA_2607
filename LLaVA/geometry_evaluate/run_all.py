#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


EXPERIMENT_ORDER = (
    "accuracy",
    "anchor",
    "represent",
    "visual_dependency",
    "language",
    "log",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all geometry evaluation modules sequentially."
    )

    # Evaluation code
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
    )

    # Evaluation datasets
    parser.add_argument(
        "--stage1-data-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stage2-data-dir",
        type=Path,
        required=True,
    )

    # Models
    parser.add_argument(
        "--base-model",
        required=True,
    )
    parser.add_argument(
        "--stage1-dir",
        type=Path,
        required=True,
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

    # Training logs / output directories containing trainer_state.json
    parser.add_argument(
        "--stage1-log",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stage2-log",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stage2-only-log",
        type=Path,
        required=True,
    )

    # Result root
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
    )

    # Optional local language corpus
    parser.add_argument(
        "--language-text-file",
        type=Path,
        default=None,
    )

    # Shared smoke-test option
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
    )

    # log.py base-model download behavior
    parser.add_argument(
        "--local-files-only",
        action="store_true",
    )

    return parser.parse_args()


def build_commands(
    args: argparse.Namespace,
) -> dict[str, list[str]]:
    python = sys.executable
    eval_dir = args.eval_dir.resolve()
    result_dir = args.result_dir.resolve()

    common_model_args = [
        "--base-model",
        str(args.base_model),

        "--stage1-dir",
        str(args.stage1_dir.resolve()),

        "--stage2-dir",
        str(args.stage2_dir.resolve()),

        "--stage2-only-dir",
        str(args.stage2_only_dir.resolve()),
    ]

    common_data_args = [
        "--stage1-data-dir",
        str(args.stage1_data_dir.resolve()),

        "--stage2-data-dir",
        str(args.stage2_data_dir.resolve()),
    ]

    commands: dict[str, list[str]] = {}

    # =========================================================================
    # 1. Accuracy
    # =========================================================================
    commands["accuracy"] = [
        python,
        str(eval_dir / "accuracy.py"),

        *common_data_args,
        *common_model_args,

        "--output-dir",
        str(result_dir / "accuracy"),

        "--max-samples",
        str(args.max_samples),
    ]

    # =========================================================================
    # 2. Anchor
    # =========================================================================
    commands["anchor"] = [
        python,
        str(eval_dir / "anchor.py"),

        "--stage2-data-dir",
        str(args.stage2_data_dir.resolve()),

        *common_model_args,

        "--output-dir",
        str(result_dir / "anchor"),

        "--max-samples",
        str(args.max_samples),
    ]

    # =========================================================================
    # 3. Representation
    # =========================================================================
    commands["represent"] = [
        python,
        str(eval_dir / "represent.py"),

        *common_data_args,
        *common_model_args,

        "--output-dir",
        str(result_dir / "represent"),

        "--max-samples",
        str(args.max_samples),
    ]

    # =========================================================================
    # 4. Visual dependency
    # =========================================================================
    commands["visual_dependency"] = [
        python,
        str(eval_dir / "visual_dependency.py"),

        "--stage2-data-dir",
        str(args.stage2_data_dir.resolve()),

        *common_model_args,

        "--output-dir",
        str(result_dir / "visual_dependency"),

        "--max-samples",
        str(args.max_samples),
    ]

    # =========================================================================
    # 5. Language
    #
    # language.py deliberately does not require Stage1.
    # =========================================================================
    commands["language"] = [
        python,
        str(eval_dir / "language.py"),

        "--base-model",
        str(args.base_model),

        "--stage2-dir",
        str(args.stage2_dir.resolve()),

        "--stage2-only-dir",
        str(args.stage2_only_dir.resolve()),

        "--output-dir",
        str(result_dir / "language"),
    ]

    if args.language_text_file is not None:
        commands["language"].extend(
            [
                "--text-file",
                str(args.language_text_file.resolve()),
            ]
        )

    # =========================================================================
    # 6. Training log / weight change
    # =========================================================================
    commands["log"] = [
        python,
        str(eval_dir / "log.py"),

        "--stage1-log",
        str(args.stage1_log.resolve()),

        "--stage2-log",
        str(args.stage2_log.resolve()),

        "--stage2-only-log",
        str(args.stage2_only_log.resolve()),

        "--base-model",
        str(args.base_model),

        "--stage2-dir",
        str(args.stage2_dir.resolve()),

        "--stage2-only-dir",
        str(args.stage2_only_dir.resolve()),

        "--output-dir",
        str(result_dir / "log"),
    ]

    if args.local_files_only:
        commands["log"].append(
            "--local-files-only"
        )

    return commands


def run_experiment(
    *,
    name: str,
    command: list[str],
    eval_dir: Path,
    log_file: Path,
) -> dict[str, Any]:
    """
    Run one evaluator as an isolated subprocess.

    A non-zero exit code is recorded but never raises into the global runner.
    The next experiment therefore always starts.
    """

    start_time = datetime.now()
    start_clock = time.monotonic()

    print()
    print("=" * 80)
    print(f"[START] {name}")
    print(f"Time : {start_time.isoformat(timespec='seconds')}")
    print(f"Log  : {log_file}")
    print("=" * 80)
    print(" ".join(command))
    print()

    log_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return_code: int | None = None
    launch_error: str | None = None

    try:
        with log_file.open(
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(
                f"Experiment: {name}\n"
            )
            handle.write(
                f"Started: {start_time.isoformat(timespec='seconds')}\n"
            )
            handle.write(
                "Command:\n"
                + " ".join(command)
                + "\n\n"
            )
            handle.flush()

            process = subprocess.Popen(
                command,
                cwd=str(eval_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert process.stdout is not None

            for line in process.stdout:
                print(
                    f"[{name}] {line}",
                    end="",
                    flush=True,
                )

                handle.write(line)
                handle.flush()

            return_code = process.wait()

    except KeyboardInterrupt:
        print(
            f"\n[INTERRUPTED] {name}"
        )
        raise

    except Exception as exc:
        launch_error = (
            f"{type(exc).__name__}: {exc}"
        )

        print(
            f"[RUNNER ERROR] {name}: {launch_error}"
        )

    elapsed = (
        time.monotonic()
        - start_clock
    )

    end_time = datetime.now()

    success = (
        launch_error is None
        and return_code == 0
    )

    status = (
        "SUCCESS"
        if success
        else "FAILED"
    )

    print()
    print(
        f"[{status}] {name} "
        f"(return_code={return_code}, "
        f"elapsed={elapsed:.1f}s)"
    )

    # Important:
    # We return the failure rather than raise it.
    # The caller will proceed to the next experiment.
    return {
        "experiment": name,
        "status": status,
        "return_code": return_code,
        "launch_error": launch_error,
        "started": start_time.isoformat(
            timespec="seconds"
        ),
        "finished": end_time.isoformat(
            timespec="seconds"
        ),
        "elapsed_seconds": elapsed,
        "log_file": str(log_file),
        "command": command,
    }


def write_summary(
    result_dir: Path,
    runs: list[dict[str, Any]],
) -> None:
    # =========================================================================
    # JSON
    # =========================================================================
    json_file = (
        result_dir
        / "run_summary.json"
    )

    json_file.write_text(
        json.dumps(
            {
                "generated": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "runs": runs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # =========================================================================
    # Markdown
    # =========================================================================
    lines = [
        "# Geometry Evaluation Run Summary",
        "",
        "| Experiment | Status | Return Code | Elapsed | Log |",
        "|---|---|---:|---:|---|",
    ]

    for run in runs:
        elapsed = (
            f"{run['elapsed_seconds'] / 60:.1f} min"
        )

        return_code = (
            run["return_code"]
            if run["return_code"] is not None
            else "-"
        )

        lines.append(
            f"| {run['experiment']} "
            f"| **{run['status']}** "
            f"| {return_code} "
            f"| {elapsed} "
            f"| `{run['log_file']}` |"
        )

    success_n = sum(
        run["status"] == "SUCCESS"
        for run in runs
    )

    failure_n = (
        len(runs)
        - success_n
    )

    lines.extend(
        [
            "",
            f"- Successful: **{success_n}/{len(runs)}**",
            f"- Failed: **{failure_n}/{len(runs)}**",
            "",
            (
                "A failed evaluator does not stop the remaining evaluators. "
                "See the corresponding runner log for its traceback."
            ),
            "",
        ]
    )

    (
        result_dir
        / "run_summary.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    args.eval_dir = (
        args.eval_dir.resolve()
    )

    args.result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    commands = build_commands(
        args
    )

    # =========================================================================
    # Preflight: report missing evaluator scripts.
    #
    # Do not abort the entire run. Missing scripts are treated exactly like
    # a failed experiment.
    # =========================================================================
    runner_log_dir = (
        args.result_dir
        / "runner_logs"
    )

    runner_log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs: list[
        dict[str, Any]
    ] = []

    for name in EXPERIMENT_ORDER:
        script = (
            args.eval_dir
            / f"{name}.py"
        )

        # Actual filename is visual_dependency.py.
        if name == "visual_dependency":
            script = (
                args.eval_dir
                / "visual_dependency.py"
            )

        if not script.exists():
            print(
                f"[FAILED] {name}: "
                f"script not found: {script}"
            )

            runs.append(
                {
                    "experiment": name,
                    "status": "FAILED",
                    "return_code": None,
                    "launch_error": (
                        f"script not found: {script}"
                    ),
                    "started": None,
                    "finished": None,
                    "elapsed_seconds": 0.0,
                    "log_file": str(
                        runner_log_dir
                        / f"{name}.log"
                    ),
                    "command": commands[name],
                }
            )

            write_summary(
                args.result_dir,
                runs,
            )

            continue

        result = run_experiment(
            name=name,
            command=commands[name],
            eval_dir=args.eval_dir,
            log_file=(
                runner_log_dir
                / f"{name}.log"
            ),
        )

        runs.append(result)

        # Save the summary after every evaluator.
        # Even if the server/job dies later, completed statuses remain.
        write_summary(
            args.result_dir,
            runs,
        )

    success_n = sum(
        run["status"] == "SUCCESS"
        for run in runs
    )

    failure_n = (
        len(runs)
        - success_n
    )

    print()
    print("=" * 80)
    print("ALL EVALUATORS FINISHED")
    print("=" * 80)
    print(
        f"Successful: {success_n}/{len(runs)}"
    )
    print(
        f"Failed    : {failure_n}/{len(runs)}"
    )
    print(
        f"Results   : {args.result_dir}"
    )

    # All experiments have already been attempted.
    # Return non-zero only at the very end if anything failed.
    if failure_n:
        sys.exit(1)


if __name__ == "__main__":
    main()