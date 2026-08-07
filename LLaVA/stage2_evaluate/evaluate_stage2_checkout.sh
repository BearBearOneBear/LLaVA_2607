#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

MODE_ORDER = ("normal", "shuffled", "blank", "none")
MODEL_ORDER = ("base", "stage1", "stage2")
STAGE3_DATASETS = ("stage3_base", "stage3_values", "stage3_unseen", "stage3_wide")

LOAD_ERRORS: list[str] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine Stage-2 evaluator outputs, including partial/failing runs, into one self-contained summary."
    )
    parser.add_argument("--root", type=Path, default=Path("./stage2_evaluate/results"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOAD_ERRORS.append(f"Failed to read JSON {path}: {type(exc).__name__}: {exc}")
        return None
    return value if isinstance(value, dict) else None


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return None
    low = value.lower()
    if low in {"none", "null", "nan", "inf", "+inf", "-inf"}:
        return None if low in {"none", "null", "nan"} else value
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [
                {key: parse_scalar(value or "") for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
    except Exception as exc:
        LOAD_ERRORS.append(f"Failed to read CSV {path}: {type(exc).__name__}: {exc}")
        return []


def load_status(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                rows.append(
                    {
                        "step": row.get("step"),
                        "status": row.get("status"),
                        "exit_code": row.get("exit_code"),
                        "log": row.get("log"),
                    }
                )
    except Exception as exc:
        LOAD_ERRORS.append(f"Failed to read status file {path}: {type(exc).__name__}: {exc}")
    return rows


def get(d: dict | None, *keys, default=None):
    cur: Any = d
    for key in keys:
        if cur is None or not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def fmt(x: Any) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        if not math.isfinite(x):
            return "N/A"
        if abs(x) >= 1000:
            return f"{x:.2f}"
        return f"{x:.4f}"
    return str(x)


def fmt_path(x: Any) -> str:
    return "N/A" if not x else f"`{x}`"


def metric_path(root: Path, model: str, mode: str, dataset: str) -> Path:
    return root / model / mode / "metrics" / f"{dataset}.json"


def load_metric(root: Path, model: str, mode: str, dataset: str) -> dict[str, Any] | None:
    return load_json(metric_path(root, model, mode, dataset))


def delta_from_normal(values: dict[str, Any], mode: str) -> float | None:
    normal = values.get("normal")
    other = values.get(mode)
    if not isinstance(normal, (int, float)) or not isinstance(other, (int, float)):
        return None
    return float(normal) - float(other)


def compact_anchor(m: dict | None) -> dict[str, Any]:
    return {
        "n": get(m, "anchor", "n"),
        "parse_success_rate": get(m, "anchor", "parse_success_rate"),
        "semantic_exact_match": get(m, "anchor", "semantic_exact_match"),
        "fact_micro_f1": get(m, "anchor", "fact_micro", "f1"),
        "point_micro_f1": get(m, "anchor", "point_micro", "f1"),
    }


def collect_stratified(m: dict | None) -> dict[str, Any]:
    strat = get(m, "stratified", default={}) or {}
    out: dict[str, Any] = {}
    for axis, buckets in strat.items():
        if not isinstance(buckets, dict):
            continue
        out[axis] = {}
        for bucket, bm in buckets.items():
            out[axis][bucket] = {
                "n": get(bm, "n"),
                "parse_success_rate": get(bm, "parse_success_rate"),
                "fact_micro_f1": get(bm, "fact_micro", "f1"),
                "point_micro_f1": get(bm, "point_micro", "f1"),
            }
    return out


def finite_numbers(items: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for item in items:
        value = item.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out.append(float(value))
    return out


def history_summary(integrity: dict | None) -> dict[str, Any]:
    train = get(integrity, "trainer_state_summary", "train_loss_history", default=[]) or []
    evals = get(integrity, "trainer_state_summary", "eval_loss_history", default=[]) or []
    train_losses = finite_numbers(train, "loss")
    eval_losses = finite_numbers(evals, "eval_loss")
    return {
        "train_points": len(train),
        "eval_points": len(evals),
        "first_train_loss": train_losses[0] if train_losses else None,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "min_train_loss": min(train_losses) if train_losses else None,
        "max_train_loss": max(train_losses) if train_losses else None,
        "best_eval_loss": min(eval_losses) if eval_losses else None,
        "last_train_entries": train[-20:],
        "eval_history": evals,
    }


def sort_layer_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]):
        layer = row.get("layer")
        if isinstance(layer, int):
            return (0, layer)
        if isinstance(layer, str) and layer.isdigit():
            return (0, int(layer))
        return (1, str(layer))

    return sorted(rows, key=key)


def tail_nonempty(path: str | None, max_lines: int = 8) -> list[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists() or not p.is_file():
        return []
    try:
        lines = [x.rstrip() for x in p.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]
        return lines[-max_lines:]
    except Exception:
        return []


def representation_tables(rep: dict | None) -> dict[str, Any]:
    if not rep:
        return {
            "metadata": None,
            "best_layer": None,
            "experiment_a_rows": [],
            "concept_rows": [],
            "experiment_b_rows": [],
            "teacher_residual_rows": [],
            "student_delta_rows": [],
        }

    metadata = rep.get("metadata") if isinstance(rep.get("metadata"), dict) else {}
    best_layer = get(rep, "quick_decision", "best_stage2_layer_by_centered_visual_separation")
    layers = [str(x) for x in metadata.get("layers", [])]
    if not layers:
        layers = sorted(
            get(rep, "experiment_a", "models", "stage2", "layers", default={}).keys(),
            key=lambda x: int(x) if str(x).isdigit() else str(x),
        )

    a_rows: list[dict[str, Any]] = []
    for model_name in ("llava-v1.5-7b", "stage2"):
        model_data = get(rep, "experiment_a", "models", model_name, default={}) or {}
        for layer in layers:
            d = get(model_data, "layers", str(layer), default={}) or {}
            raw = get(d, "normal_minus_blank", "raw", default={}) or {}
            centered = get(d, "normal_minus_blank", "centered", default={}) or {}
            shuffled_centered = get(d, "shuffled_minus_blank", "centered", default={}) or {}
            a_rows.append(
                {
                    "model": model_name,
                    "layer": layer,
                    "raw_s_same": raw.get("s_same"),
                    "raw_s_diff": raw.get("s_diff"),
                    "raw_separation": raw.get("separation"),
                    "raw_random_cosine": raw.get("random_cosine_baseline"),
                    "centered_s_same": centered.get("s_same"),
                    "centered_s_diff": centered.get("s_diff"),
                    "centered_separation": centered.get("separation"),
                    "centered_random_cosine": centered.get("random_cosine_baseline"),
                    "shuffled_centered_separation": shuffled_centered.get("separation"),
                    "delta_visual_norm_mean": get(d, "normal_minus_blank", "norm", "mean"),
                    "h_normal_random_cosine": get(d, "state_random_cosine", "h_normal"),
                    "h_blank_random_cosine": get(d, "state_random_cosine", "h_blank"),
                    "h_shuffled_random_cosine": get(d, "state_random_cosine", "h_shuffled"),
                }
            )

    concept_rows: list[dict[str, Any]] = []
    best_layer_key = str(get(best_layer, "layer")) if best_layer else None
    if best_layer_key:
        for model_name in ("llava-v1.5-7b", "stage2"):
            d = get(rep, "experiment_a", "models", model_name, "layers", best_layer_key, default={}) or {}
            raw_concepts = get(d, "normal_minus_blank", "raw", "concept_s_same", default={}) or {}
            centered_concepts = get(d, "normal_minus_blank", "centered", "concept_s_same", default={}) or {}
            for concept in sorted(set(raw_concepts) | set(centered_concepts)):
                concept_rows.append(
                    {
                        "model": model_name,
                        "layer": best_layer_key,
                        "concept": concept,
                        "raw_s_same": raw_concepts.get(concept),
                        "centered_s_same": centered_concepts.get(concept),
                    }
                )

    b_rows: list[dict[str, Any]] = []
    alignments = get(rep, "experiment_b", "alignments", default={}) or {}
    for student_name, student_data in alignments.items():
        if not isinstance(student_data, dict):
            continue
        for layer, layer_data in student_data.items():
            if not isinstance(layer_data, dict):
                continue
            for teacher_format in ("T1", "T2"):
                tf = layer_data.get(teacher_format) or {}
                for state_name in ("h_pre", "delta"):
                    metrics = tf.get(state_name) or {}
                    b_rows.append(
                        {
                            "student": student_name,
                            "layer": layer,
                            "teacher": teacher_format,
                            "state": state_name,
                            "matched": metrics.get("matched_cosine"),
                            "mismatched": metrics.get("mismatched_cosine"),
                            "margin": metrics.get("margin"),
                        }
                    )

    teacher_rows: list[dict[str, Any]] = []
    teacher_cmp = get(rep, "experiment_b", "teacher_delta_visual_comparison", default={}) or {}
    for layer, d in teacher_cmp.items():
        teacher_rows.append(
            {
                "layer": layer,
                "T1_norm_mean": get(d, "T1_delta_visual_norm", "mean"),
                "T2_norm_mean": get(d, "T2_delta_visual_norm", "mean"),
                "T2_over_T1": d.get("T2_over_T1_mean_norm_ratio") if isinstance(d, dict) else None,
                "T1_T2_cosine": d.get("T1_vs_T2_delta_visual_cosine") if isinstance(d, dict) else None,
            }
        )

    student_rows: list[dict[str, Any]] = []
    student_norm = get(rep, "experiment_b", "student_delta_text_norm", default={}) or {}
    for student_name, layer_data in student_norm.items():
        if not isinstance(layer_data, dict):
            continue
        for layer, stats in layer_data.items():
            student_rows.append(
                {
                    "student": student_name,
                    "layer": layer,
                    "mean": get(stats, "mean"),
                    "std": get(stats, "std"),
                    "p10": get(stats, "p10"),
                    "p50": get(stats, "p50"),
                    "p90": get(stats, "p90"),
                }
            )

    return {
        "metadata": metadata,
        "best_layer": best_layer,
        "experiment_a_rows": a_rows,
        "concept_rows": concept_rows,
        "experiment_b_rows": b_rows,
        "teacher_residual_rows": teacher_rows,
        "student_delta_rows": student_rows,
    }


def main() -> None:
    args = parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)

    execution_status = load_status(root / "step_status.tsv")
    integrity = load_json(root / "integrity.json")
    weight = load_json(root / "weight_audit" / "summary.json")
    weight_layers = sort_layer_rows(load_csv(root / "weight_audit" / "llm_layer_delta.csv"))
    weight_modules = load_csv(root / "weight_audit" / "llm_module_delta.csv")
    projector_rows = load_csv(root / "weight_audit" / "projector_stage1_to_stage2_delta.csv")
    representation = load_json(root / "representation" / "representation.json")
    rep_compact = representation_tables(representation)

    language = {
        model: load_json(root / "language" / f"{model}.json")
        for model in ("base", "stage2")
    }

    normal_metrics = {
        model: {
            dataset: load_metric(root, model, "normal", dataset)
            for dataset in ("stage1", "stage2")
        }
        for model in MODEL_ORDER
    }
    behavior = {
        "stage1_test_concept_accuracy": {
            model: get(normal_metrics[model]["stage1"], "stage1_behavior", "concept_accuracy")
            for model in MODEL_ORDER
        },
        "stage2_local_concept_accuracy": {
            model: get(normal_metrics[model]["stage2"], "local", "concept_accuracy")
            for model in MODEL_ORDER
        },
        "stage2_anchor_fact_micro_f1": {
            model: get(normal_metrics[model]["stage2"], "anchor", "fact_micro", "f1")
            for model in MODEL_ORDER
        },
        "stage2_anchor_parse_success": {
            model: get(normal_metrics[model]["stage2"], "anchor", "parse_success_rate")
            for model in MODEL_ORDER
        },
        "stage2_pair_signature_inconsistency": {
            model: get(normal_metrics[model]["stage2"], "pair_consistency", "inconsistency_rate")
            for model in MODEL_ORDER
        },
    }

    stage2_mode_metrics = {
        mode: load_metric(root, "stage2", mode, "stage2") for mode in MODE_ORDER
    }
    local_by_mode = {
        mode: get(m, "local", "concept_accuracy") for mode, m in stage2_mode_metrics.items()
    }
    anchor_by_mode = {
        mode: get(m, "anchor", "fact_micro", "f1") for mode, m in stage2_mode_metrics.items()
    }
    parse_by_mode = {
        mode: get(m, "anchor", "parse_success_rate") for mode, m in stage2_mode_metrics.items()
    }
    image_ablation = {
        "stage2_local_concept_accuracy": local_by_mode,
        "stage2_anchor_fact_micro_f1": anchor_by_mode,
        "stage2_anchor_parse_success": parse_by_mode,
        "normal_minus_mode": {
            "local_concept_accuracy": {
                mode: delta_from_normal(local_by_mode, mode) for mode in MODE_ORDER[1:]
            },
            "anchor_fact_micro_f1": {
                mode: delta_from_normal(anchor_by_mode, mode) for mode in MODE_ORDER[1:]
            },
        },
        "visual_gain_normal_minus_none": {
            "local_concept_accuracy": delta_from_normal(local_by_mode, "none"),
            "anchor_fact_micro_f1": delta_from_normal(anchor_by_mode, "none"),
        },
    }

    stage3_transfer: dict[str, Any] = {}
    for dataset in STAGE3_DATASETS:
        stage3_transfer[dataset] = {}
        for mode in MODE_ORDER:
            m = load_metric(root, "stage2", mode, dataset)
            stage3_transfer[dataset][mode] = compact_anchor(m)
    normal_base_f1 = get(stage3_transfer, "stage3_base", "normal", "fact_micro_f1")
    for dataset in STAGE3_DATASETS:
        current = get(stage3_transfer, dataset, "normal", "fact_micro_f1")
        stage3_transfer[dataset]["normal_delta_vs_stage3_base"] = (
            current - normal_base_f1
            if isinstance(current, (int, float)) and isinstance(normal_base_f1, (int, float))
            else None
        )
        normal_f1 = get(stage3_transfer, dataset, "normal", "fact_micro_f1")
        none_f1 = get(stage3_transfer, dataset, "none", "fact_micro_f1")
        stage3_transfer[dataset]["visual_gain_normal_minus_none"] = (
            normal_f1 - none_f1
            if isinstance(normal_f1, (int, float)) and isinstance(none_f1, (int, float))
            else None
        )
    stage3_stratified = {
        dataset: collect_stratified(load_metric(root, "stage2", "normal", dataset))
        for dataset in STAGE3_DATASETS
    }

    ppl_base = get(language["base"], "wikitext_ppl", "perplexity")
    ppl_stage2 = get(language["stage2"], "wikitext_ppl", "perplexity")
    training = history_summary(integrity)

    summary = {
        "execution_status": execution_status,
        "load_errors": LOAD_ERRORS,
        "integrity": integrity,
        "training_summary": training,
        "weight_audit": weight,
        "weight_audit_detail": {
            "llm_layers": weight_layers,
            "llm_modules": weight_modules,
            "projector_stage1_to_stage2": projector_rows,
        },
        "behavior": behavior,
        "image_ablation": image_ablation,
        "representation_audit": representation,
        "representation_compact": rep_compact,
        "language": {
            "base_ppl": ppl_base,
            "stage2_ppl": ppl_stage2,
            "ppl_delta_stage2_minus_base": (
                ppl_stage2 - ppl_base
                if isinstance(ppl_base, (int, float)) and isinstance(ppl_stage2, (int, float))
                else None
            ),
            "base_dsl_leak_rate": get(language["base"], "dsl_leakage", "dsl_leak_rate"),
            "stage2_dsl_leak_rate": get(language["stage2"], "dsl_leakage", "dsl_leak_rate"),
        },
        "stage3_transfer": stage3_transfer,
        "stage3_stratified": stage3_stratified,
    }

    out_json = root / "evaluation_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = [
        "# Stage-2 Checkpoint Evaluation Summary",
        "",
        "> This report is intentionally self-contained. Failed evaluators are shown as FAILED and their unavailable metrics appear as N/A.",
        "",
        "## 0. Execution status",
        "",
        "| step | status | exit code |",
        "|---|---|---:|",
    ]
    if execution_status:
        for row in execution_status:
            lines.append(f"| {row.get('step')} | {row.get('status')} | {row.get('exit_code')} |")
    else:
        lines.append("| N/A | No step_status.tsv found | N/A |")

    failed = [x for x in execution_status if x.get("status") == "FAILED"]
    if failed:
        lines += ["", "### Failed-step diagnostics", ""]
        for row in failed:
            lines.append(f"**{row.get('step')}** — exit code {row.get('exit_code')}")
            tail = tail_nonempty(row.get("log"))
            if tail:
                lines += ["", "```text", *tail, "```", ""]
            else:
                lines.append("")

    if LOAD_ERRORS:
        lines += ["", "### Summary-load warnings", ""]
        for error in LOAD_ERRORS:
            lines.append(f"- {error}")

    trainer = get(integrity, "trainer_state_summary", default={}) or {}
    stage1_projector = get(integrity, "stage1", "projector", default={}) or {}
    stage2_model_files = get(integrity, "stage2", "model_weight_files", default=[]) or []
    lines += [
        "",
        "## 1. Training and checkpoint state",
        "",
        f"- Stage1 projector path: {fmt_path(stage1_projector.get('path'))}",
        f"- Stage1 projector exists: {fmt(stage1_projector.get('exists'))}",
        f"- Stage2 full-model weight files found: {len(stage2_model_files)}",
        f"- `trainer_state.json` available: {fmt(trainer.get('available'))}",
        f"- Best checkpoint: {fmt_path(trainer.get('best_model_checkpoint'))}",
        f"- Best metric: {fmt(trainer.get('best_metric'))}",
        f"- Final global step: {fmt(trainer.get('global_step'))}",
        f"- Final epoch: {fmt(trainer.get('epoch'))}",
        f"- Recorded train-loss points: {training['train_points']}",
        f"- First / final train loss: {fmt(training['first_train_loss'])} / {fmt(training['final_train_loss'])}",
        f"- Min / max train loss: {fmt(training['min_train_loss'])} / {fmt(training['max_train_loss'])}",
        f"- Best recorded eval loss: {fmt(training['best_eval_loss'])}",
        "",
    ]

    if training["eval_history"]:
        lines += ["### Eval-loss history", "", "| step | epoch | eval loss |", "|---:|---:|---:|"]
        for item in training["eval_history"]:
            lines.append(f"| {fmt(item.get('step'))} | {fmt(item.get('epoch'))} | {fmt(item.get('eval_loss'))} |")
        lines.append("")

    if training["last_train_entries"]:
        lines += ["### Recent train-loss history (last up to 20 points)", "", "| step | epoch | loss | learning rate |", "|---:|---:|---:|---:|"]
        for item in training["last_train_entries"]:
            lines.append(
                f"| {fmt(item.get('step'))} | {fmt(item.get('epoch'))} | {fmt(item.get('loss'))} | {fmt(item.get('learning_rate'))} |"
            )
        lines.append("")

    lines += [
        "## 2. Weight changes",
        "",
        "LLM rows compare original LLaVA-1.5 to Stage2. Projector rows compare Stage1 projector to Stage2 projector.",
        "",
        f"- Base checkpoint: {fmt_path(get(weight, 'base_checkpoint_dir'))}",
        f"- Stage2 checkpoint: {fmt_path(get(weight, 'stage2_checkpoint_dir'))}",
        f"- LLM tensors compared: {fmt(get(weight, 'llm_tensor_count'))}",
        f"- LLM layers found: {fmt(get(weight, 'llm_layer_count'))}",
        f"- Projector tensors compared: {fmt(get(weight, 'projector_tensor_count'))}",
        f"- Sampled |Δw| p50 / p90 / p95 / p99: {fmt(get(weight, 'sampled_abs_delta_quantiles', 'p50'))} / {fmt(get(weight, 'sampled_abs_delta_quantiles', 'p90'))} / {fmt(get(weight, 'sampled_abs_delta_quantiles', 'p95'))} / {fmt(get(weight, 'sampled_abs_delta_quantiles', 'p99'))}",
        "",
    ]

    if weight_layers:
        lines += ["### LLM layer-wise delta", "", "| layer | relative Δ norm | weight cosine | Δ norm | Δ mean | Δ std |", "|---:|---:|---:|---:|---:|---:|"]
        for row in weight_layers:
            lines.append(
                f"| {fmt(row.get('layer'))} | {fmt(row.get('relative_delta_norm'))} | {fmt(row.get('weight_cosine'))} | "
                f"{fmt(row.get('delta_norm'))} | {fmt(row.get('delta_mean'))} | {fmt(row.get('delta_std'))} |"
            )
        lines.append("")
    else:
        lines += ["LLM layer-wise delta: N/A", ""]

    if weight_modules:
        lines += ["### LLM module-wise delta", "", "| module | relative Δ norm | weight cosine | Δ norm |", "|---|---:|---:|---:|"]
        for row in weight_modules:
            lines.append(
                f"| {fmt(row.get('module'))} | {fmt(row.get('relative_delta_norm'))} | {fmt(row.get('weight_cosine'))} | {fmt(row.get('delta_norm'))} |"
            )
        lines.append("")

    if projector_rows:
        lines += ["### Projector Stage1 → Stage2", "", "| Stage1 tensor | relative Δ norm | weight cosine | Δ norm |", "|---|---:|---:|---:|"]
        for row in projector_rows:
            lines.append(
                f"| {fmt(row.get('stage1_key'))} | {fmt(row.get('relative_delta_norm'))} | {fmt(row.get('weight_cosine'))} | {fmt(row.get('delta_norm'))} |"
            )
        lines.append("")

    lines += [
        "## 3. Behavior — normal images",
        "",
        "| metric | base | stage1 | stage2 |",
        "|---|---:|---:|---:|",
        "| Stage1 concept accuracy | {} | {} | {} |".format(*[
            fmt(behavior["stage1_test_concept_accuracy"][m]) for m in MODEL_ORDER
        ]),
        "| Stage2 local concept accuracy | {} | {} | {} |".format(*[
            fmt(behavior["stage2_local_concept_accuracy"][m]) for m in MODEL_ORDER
        ]),
        "| Stage2 anchor fact micro-F1 | {} | {} | {} |".format(*[
            fmt(behavior["stage2_anchor_fact_micro_f1"][m]) for m in MODEL_ORDER
        ]),
        "| Stage2 anchor parse success | {} | {} | {} |".format(*[
            fmt(behavior["stage2_anchor_parse_success"][m]) for m in MODEL_ORDER
        ]),
        "| Stage2 pair inconsistency | {} | {} | {} |".format(*[
            fmt(behavior["stage2_pair_signature_inconsistency"][m]) for m in MODEL_ORDER
        ]),
        "",
        "### Stage2 image ablation",
        "",
        "`none` removes the image token/tensor; `blank` keeps the visual pathway with a white image; `shuffled` supplies a mismatched image.",
        "",
        "| image mode | local concept acc | anchor fact F1 | anchor parse | normal - mode F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODE_ORDER:
        drop = None if mode == "normal" else image_ablation["normal_minus_mode"]["anchor_fact_micro_f1"].get(mode)
        lines.append(
            f"| {mode} | {fmt(local_by_mode.get(mode))} | {fmt(anchor_by_mode.get(mode))} | {fmt(parse_by_mode.get(mode))} | {fmt(drop)} |"
        )
    lines += [
        "",
        f"- Visual gain (normal - none), local concept accuracy: {fmt(image_ablation['visual_gain_normal_minus_none']['local_concept_accuracy'])}",
        f"- Visual gain (normal - none), anchor fact F1: {fmt(image_ablation['visual_gain_normal_minus_none']['anchor_fact_micro_f1'])}",
        "",
        "## 4. Representation anchor sanity check",
        "",
    ]

    if representation:
        md = rep_compact["metadata"] or {}
        best = rep_compact["best_layer"] or {}
        lines += [
            f"- Stage2-test samples in Experiment A: {fmt(md.get('stage2_samples'))}",
            f"- Stage3-base samples in Experiment B: {fmt(md.get('stage3_base_samples'))}",
            f"- Layers: {fmt(md.get('layers'))}",
            f"- Best Stage2 layer by centered visual separation: {fmt(best.get('layer'))} (separation {fmt(best.get('separation'))})",
            "",
            "### A. Stage2 visual anchor",
            "",
            "Δvisual = h_normal - h_blank. Centered values additionally subtract the dataset mean residual.",
            "",
            "| model | layer | raw S_same | raw S_diff | raw sep | raw random | centered S_same | centered S_diff | centered sep | centered random | shuffled centered sep | mean ||Δvisual|| | h_normal random |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rep_compact["experiment_a_rows"]:
            lines.append(
                f"| {row['model']} | {row['layer']} | {fmt(row['raw_s_same'])} | {fmt(row['raw_s_diff'])} | {fmt(row['raw_separation'])} | "
                f"{fmt(row['raw_random_cosine'])} | {fmt(row['centered_s_same'])} | {fmt(row['centered_s_diff'])} | {fmt(row['centered_separation'])} | "
                f"{fmt(row['centered_random_cosine'])} | {fmt(row['shuffled_centered_separation'])} | {fmt(row['delta_visual_norm_mean'])} | {fmt(row['h_normal_random_cosine'])} |"
            )
        lines.append("")

        if rep_compact["concept_rows"]:
            lines += [
                f"### Concept-wise S_same at selected layer {fmt(best.get('layer'))}",
                "",
                "| model | concept | raw S_same | centered S_same |",
                "|---|---|---:|---:|",
            ]
            for row in rep_compact["concept_rows"]:
                lines.append(
                    f"| {row['model']} | {row['concept']} | {fmt(row['raw_s_same'])} | {fmt(row['centered_s_same'])} |"
                )
            lines.append("")

        lines += [
            "### B. Student compatibility",
            "",
            "Matched compares the same geometry; mismatched deterministically shuffles the teacher side while keeping the student fixed. Margin = matched - mismatched.",
            "",
            "| student | layer | teacher | state | matched cosine | mismatched cosine | margin |",
            "|---|---:|---|---|---:|---:|---:|",
        ]
        for row in rep_compact["experiment_b_rows"]:
            lines.append(
                f"| {row['student']} | {row['layer']} | {row['teacher']} | {row['state']} | {fmt(row['matched'])} | {fmt(row['mismatched'])} | {fmt(row['margin'])} |"
            )
        lines.append("")

        if rep_compact["teacher_residual_rows"]:
            lines += [
                "### Teacher T1 vs T2 visual-residual size",
                "",
                "| layer | mean ||Δvisual T1|| | mean ||Δvisual T2|| | T2/T1 norm | cosine(T1,T2) |",
                "|---:|---:|---:|---:|---:|",
            ]
            for row in rep_compact["teacher_residual_rows"]:
                lines.append(
                    f"| {row['layer']} | {fmt(row['T1_norm_mean'])} | {fmt(row['T2_norm_mean'])} | {fmt(row['T2_over_T1'])} | {fmt(row['T1_T2_cosine'])} |"
                )
            lines.append("")

        if rep_compact["student_delta_rows"]:
            lines += [
                "### Student Δtext size",
                "",
                "| student | layer | mean | std | p10 | p50 | p90 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for row in rep_compact["student_delta_rows"]:
                lines.append(
                    f"| {row['student']} | {row['layer']} | {fmt(row['mean'])} | {fmt(row['std'])} | {fmt(row['p10'])} | {fmt(row['p50'])} | {fmt(row['p90'])} |"
                )
            lines.append("")
    else:
        lines += ["Representation result unavailable (failed, skipped, or no output file).", ""]

    lines += [
        "## 5. Language preservation",
        "",
        f"- Base WikiText-2 PPL: {fmt(summary['language']['base_ppl'])}",
        f"- Stage2 WikiText-2 PPL: {fmt(summary['language']['stage2_ppl'])}",
        f"- ΔPPL (Stage2 - Base): {fmt(summary['language']['ppl_delta_stage2_minus_base'])}",
        f"- Base DSL leak rate: {fmt(summary['language']['base_dsl_leak_rate'])}",
        f"- Stage2 DSL leak rate: {fmt(summary['language']['stage2_dsl_leak_rate'])}",
        "",
        "## 6. Stage3 transfer — normal images",
        "",
        "| condition | parse | fact F1 | point F1 | Δ fact F1 vs base | visual gain vs none |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in STAGE3_DATASETS:
        v = stage3_transfer[dataset]["normal"]
        lines.append(
            f"| {dataset} | {fmt(v['parse_success_rate'])} | {fmt(v['fact_micro_f1'])} | {fmt(v['point_micro_f1'])} | "
            f"{fmt(stage3_transfer[dataset]['normal_delta_vs_stage3_base'])} | {fmt(stage3_transfer[dataset]['visual_gain_normal_minus_none'])} |"
        )

    lines += [
        "",
        "### Stage3 image-mode ablation — fact micro-F1",
        "",
        "| condition | normal | shuffled | blank | none |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in STAGE3_DATASETS:
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                dataset,
                *[fmt(stage3_transfer[dataset][mode]["fact_micro_f1"]) for mode in MODE_ORDER],
            )
        )

    lines += ["", "### Stage3 stratified metrics — normal images", ""]
    axis_titles = {
        "point_count_bin": "Point count",
        "ignored_symbol_count": "Ignored symbol / original-arrow presence",
        "unseen_symbols": "Actual unseen-symbol presence",
        "unseen_symbol_type": "Unseen symbol type (non-exclusive)",
        "max_collinear": "Collinearity",
        "values_in_diagram": "Values in diagram",
    }
    for dataset in STAGE3_DATASETS:
        lines += [f"#### {dataset}", ""]
        strat = stage3_stratified.get(dataset, {})
        if not strat:
            lines += ["No stratified metadata/result found.", ""]
            continue
        for axis in (
            "point_count_bin",
            "ignored_symbol_count",
            "unseen_symbols",
            "max_collinear",
            "values_in_diagram",
            "unseen_symbol_type",
        ):
            buckets = strat.get(axis)
            if not buckets:
                continue
            lines += [
                f"**{axis_titles.get(axis, axis)}**",
                "",
                "| bucket | n | parse | fact F1 | point F1 |",
                "|---|---:|---:|---:|---:|",
            ]
            for bucket, bm in buckets.items():
                lines.append(
                    f"| {bucket} | {fmt(bm.get('n'))} | {fmt(bm.get('parse_success_rate'))} | {fmt(bm.get('fact_micro_f1'))} | {fmt(bm.get('point_micro_f1'))} |"
                )
            lines.append("")

    lines += [
        "## 7. What to download",
        "",
        "This Markdown file is the primary human-readable artifact. It includes training/checkpoint state, weight changes, behavior, representation-anchor checks, language preservation, Stage3 transfer, and failed-step diagnostics.",
        "",
        "For machine-readable full detail, `evaluation_summary.json` contains the same sections plus the complete representation JSON and detailed weight tables.",
        "",
    ]

    out_md = root / "evaluation_summary.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary saved to {out_json}")
    print(f"Readable report saved to {out_md}")


if __name__ == "__main__":
    main()
