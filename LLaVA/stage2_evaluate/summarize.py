#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MODE_ORDER = ("normal", "shuffled", "blank", "none")
MODEL_ORDER = ("base", "stage1", "stage2")
STAGE3_DATASETS = ("stage3_base", "stage3_values", "stage3_unseen", "stage3_wide")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine Stage-2 evaluator outputs into one summary.")
    parser.add_argument("--root", type=Path, default=Path("./stage2_evaluate/results"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get(d: dict | None, *keys, default=None):
    cur = d
    for key in keys:
        if cur is None or not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def fmt(x: Any) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def metric_path(root: Path, model: str, mode: str, dataset: str) -> Path:
    return root / model / mode / "metrics" / f"{dataset}.json"


def load_metric(root: Path, model: str, mode: str, dataset: str) -> dict[str, Any] | None:
    return load_json(metric_path(root, model, mode, dataset))


def delta_from_normal(values: dict[str, Any], mode: str) -> float | None:
    normal = values.get("normal")
    other = values.get(mode)
    if normal is None or other is None:
        return None
    return normal - other


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


def representation_primary_rows(rep: dict | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name in ("llava-v1.5-7b", "stage2"):
        model = get(rep, "experiment_a", "models", model_name, default={}) or {}
        for layer, data in sorted((model.get("layers") or {}).items(), key=lambda kv: int(kv[0])):
            normal_raw = get(data, "normal_minus_blank", "raw", default={}) or {}
            normal_centered = get(data, "normal_minus_blank", "centered", default={}) or {}
            shuffled_centered = get(data, "shuffled_minus_blank", "centered", default={}) or {}
            rows.append({
                "model": model_name,
                "layer": int(layer),
                "raw_s_same": normal_raw.get("s_same"),
                "raw_s_diff": normal_raw.get("s_diff"),
                "raw_separation": normal_raw.get("separation"),
                "raw_random_cosine": normal_raw.get("random_cosine_baseline"),
                "centered_separation": normal_centered.get("separation"),
                "centered_random_cosine": normal_centered.get("random_cosine_baseline"),
                "shuffled_centered_separation": shuffled_centered.get("separation"),
                "delta_norm_mean": get(data, "normal_minus_blank", "norm", "mean"),
            })
    return rows


def representation_alignment_rows(rep: dict | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    alignments = get(rep, "experiment_b", "alignments", default={}) or {}
    for student_name, by_layer in alignments.items():
        for layer, by_teacher in sorted(by_layer.items(), key=lambda kv: int(kv[0])):
            for teacher_format in ("T1", "T2"):
                for state_name in ("h_pre", "delta"):
                    m = get(by_teacher, teacher_format, state_name, default={}) or {}
                    rows.append({
                        "student": student_name,
                        "layer": int(layer),
                        "teacher": teacher_format,
                        "state": state_name,
                        "matched": m.get("matched_cosine"),
                        "mismatched": m.get("mismatched_cosine"),
                        "margin": m.get("margin"),
                    })
    return rows


def main() -> None:
    args = parse_args()
    root = args.root

    integrity = load_json(root / "integrity.json")
    weight = load_json(root / "weight_audit" / "summary.json")
    representation = load_json(root / "representation" / "representation.json")
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
            if current is not None and normal_base_f1 is not None
            else None
        )
        normal_f1 = get(stage3_transfer, dataset, "normal", "fact_micro_f1")
        none_f1 = get(stage3_transfer, dataset, "none", "fact_micro_f1")
        stage3_transfer[dataset]["visual_gain_normal_minus_none"] = (
            normal_f1 - none_f1
            if normal_f1 is not None and none_f1 is not None
            else None
        )
    stage3_stratified = {
        dataset: collect_stratified(load_metric(root, "stage2", "normal", dataset))
        for dataset in STAGE3_DATASETS
    }

    ppl_base = get(language["base"], "wikitext_ppl", "perplexity")
    ppl_stage2 = get(language["stage2"], "wikitext_ppl", "perplexity")
    summary = {
        "integrity": integrity,
        "weight_audit": weight,
        "behavior": behavior,
        "image_ablation": image_ablation,
        "representation_audit": representation,
        "language": {
            "base_ppl": ppl_base,
            "stage2_ppl": ppl_stage2,
            "ppl_delta_stage2_minus_base": (
                ppl_stage2 - ppl_base if ppl_base is not None and ppl_stage2 is not None else None
            ),
            "base_dsl_leak_rate": get(language["base"], "dsl_leakage", "dsl_leak_rate"),
            "stage2_dsl_leak_rate": get(language["stage2"], "dsl_leakage", "dsl_leak_rate"),
        },
        "stage3_transfer": stage3_transfer,
        "stage3_stratified": stage3_stratified,
    }

    out_json = root / "evaluation_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Stage-2 Checkpoint Evaluation Summary",
        "",
        "## Behavior — normal images",
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
        "",
        "## Image ablation — Stage2 model on Stage2 test",
        "",
        "`none` removes the image token and image tensor entirely; `blank` keeps the visual pathway but supplies a white image; `shuffled` supplies a mismatched image from the same dataset.",
        "",
        "| image mode | local concept acc | anchor fact F1 | anchor parse | normal - mode F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODE_ORDER:
        drop = None if mode == "normal" else image_ablation["normal_minus_mode"]["anchor_fact_micro_f1"].get(mode)
        lines.append(
            f"| {mode} | {fmt(local_by_mode.get(mode))} | {fmt(anchor_by_mode.get(mode))} | "
            f"{fmt(parse_by_mode.get(mode))} | {fmt(drop)} |"
        )
    lines += [
        "",
        f"- Visual gain (normal - none), local concept accuracy: {fmt(image_ablation['visual_gain_normal_minus_none']['local_concept_accuracy'])}",
        f"- Visual gain (normal - none), anchor fact F1: {fmt(image_ablation['visual_gain_normal_minus_none']['anchor_fact_micro_f1'])}",
        "",
        "## Representation — A. Stage2 visual anchor",
        "",
    ]

    if representation is None:
        lines += ["Representation audit result was not found.", ""]
    else:
        best = get(representation, "quick_decision", "best_stage2_layer_by_centered_visual_separation")
        if best:
            lines += [
                f"- Best Stage2 layer by centered visual-residual separation: **{fmt(best.get('layer'))}** (separation {fmt(best.get('separation'))})",
                "",
            ]
        lines += [
            "`Δvisual = h_normal - h_blank`. Centered metrics subtract the dataset mean residual before cosine normalization. The shuffled column uses `h_shuffled - h_blank` but keeps the source concept labels.",
            "",
            "| model | layer | raw S_same | raw S_diff | raw separation | raw random cosine | centered separation | shuffled centered separation | mean ||Δvisual|| |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in representation_primary_rows(representation):
            lines.append(
                f"| {row['model']} | {row['layer']} | {fmt(row['raw_s_same'])} | {fmt(row['raw_s_diff'])} | "
                f"{fmt(row['raw_separation'])} | {fmt(row['raw_random_cosine'])} | {fmt(row['centered_separation'])} | "
                f"{fmt(row['shuffled_centered_separation'])} | {fmt(row['delta_norm_mean'])} |"
            )

        lines += [
            "",
            "Concept-specific S_same values are retained in `representation/representation.json` rather than expanded into this main report.",
            "",
            "## Representation — B. Student compatibility",
            "",
            "T1 = Stage2(image + anchor prompt). T2 = Stage2(image + structure_description + anchor prompt). Both students receive the same `structure_description + anchor prompt`. `delta` compares teacher Δvisual with student Δtext. Mismatched cosine deranges only the teacher rows.",
            "",
            "| student | layer | teacher | state | matched cosine | mismatched cosine | margin |",
            "|---|---:|---|---|---:|---:|---:|",
        ]
        for row in representation_alignment_rows(representation):
            lines.append(
                f"| {row['student']} | {row['layer']} | {row['teacher']} | {row['state']} | "
                f"{fmt(row['matched'])} | {fmt(row['mismatched'])} | {fmt(row['margin'])} |"
            )

        lines += [
            "",
            "### Teacher residual size — T1 vs T2",
            "",
            "| layer | mean ||Δvisual T1|| | mean ||Δvisual T2|| | T2/T1 norm ratio | cosine(ΔT1, ΔT2) |",
            "|---:|---:|---:|---:|---:|",
        ]
        comparison = get(representation, "experiment_b", "teacher_delta_visual_comparison", default={}) or {}
        for layer, row in sorted(comparison.items(), key=lambda kv: int(kv[0])):
            lines.append(
                f"| {layer} | {fmt(get(row, 'T1_delta_visual_norm', 'mean'))} | "
                f"{fmt(get(row, 'T2_delta_visual_norm', 'mean'))} | "
                f"{fmt(row.get('T2_over_T1_mean_norm_ratio'))} | "
                f"{fmt(row.get('T1_vs_T2_delta_visual_cosine'))} |"
            )

        lines += [
            "",
            "### Student Δtext size",
            "",
            "| student | layer | mean ||Δtext|| | std ||Δtext|| |",
            "|---|---:|---:|---:|",
        ]
        delta_text_norm = get(representation, "experiment_b", "student_delta_text_norm", default={}) or {}
        for student, by_layer in delta_text_norm.items():
            for layer, row in sorted(by_layer.items(), key=lambda kv: int(kv[0])):
                lines.append(
                    f"| {student} | {layer} | {fmt(row.get('mean'))} | {fmt(row.get('std'))} |"
                )

    lines += [
        "",
        "## Language",
        "",
        f"- Base WikiText-2 PPL: {fmt(summary['language']['base_ppl'])}",
        f"- Stage2 WikiText-2 PPL: {fmt(summary['language']['stage2_ppl'])}",
        f"- ΔPPL (Stage2 - Base): {fmt(summary['language']['ppl_delta_stage2_minus_base'])}",
        f"- Base DSL leak rate: {fmt(summary['language']['base_dsl_leak_rate'])}",
        f"- Stage2 DSL leak rate: {fmt(summary['language']['stage2_dsl_leak_rate'])}",
        "",
        "## Stage3 transfer — normal images",
        "",
        "| condition | parse | fact F1 | point F1 | Δ fact F1 vs base | visual gain vs none |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in STAGE3_DATASETS:
        v = stage3_transfer[dataset]["normal"]
        lines.append(
            f"| {dataset} | {fmt(v['parse_success_rate'])} | {fmt(v['fact_micro_f1'])} | "
            f"{fmt(v['point_micro_f1'])} | {fmt(stage3_transfer[dataset]['normal_delta_vs_stage3_base'])} | "
            f"{fmt(stage3_transfer[dataset]['visual_gain_normal_minus_none'])} |"
        )

    lines += [
        "",
        "## Stage3 image-mode ablation — fact micro-F1",
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

    lines += ["", "## Stage3 stratified metrics — normal images", ""]
    axis_titles = {
        "point_count_bin": "Point count",
        "ignored_symbol_count": "Ignored symbol / original-arrow presence",
        "unseen_symbols": "Actual unseen-symbol presence",
        "unseen_symbol_type": "Unseen symbol type (non-exclusive)",
        "max_collinear": "Collinearity",
        "values_in_diagram": "Values in diagram",
    }
    for dataset in STAGE3_DATASETS:
        lines += [f"### {dataset}", ""]
        strat = stage3_stratified.get(dataset, {})
        if not strat:
            lines += ["No stratified metadata found.", ""]
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
                    f"| {bucket} | {fmt(bm.get('n'))} | {fmt(bm.get('parse_success_rate'))} | "
                    f"{fmt(bm.get('fact_micro_f1'))} | {fmt(bm.get('point_micro_f1'))} |"
                )
            lines.append("")

    out_md = root / "evaluation_summary.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary saved to {out_json}")
    print(f"Readable report saved to {out_md}")


if __name__ == "__main__":
    main()
