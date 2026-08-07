#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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
        if cur is None or key not in cur:
            return default
        cur = cur[key]
    return cur


def fmt(x: Any) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def main() -> None:
    args = parse_args()
    root = args.root

    metrics = {}
    for model in ("base", "stage1", "stage2"):
        metrics[model] = {}
        for dataset in ("stage1", "stage2", "stage3_base", "stage3_values", "stage3_unseen", "stage3_wide"):
            metrics[model][dataset] = load_json(root / model / "metrics" / f"{dataset}.json")

    language = {
        model: load_json(root / "language" / f"{model}.json")
        for model in ("base", "stage2")
    }
    integrity = load_json(root / "integrity.json")
    weight = load_json(root / "weight_audit" / "summary.json")

    stage3_scores = {}
    for dataset in ("stage3_base", "stage3_values", "stage3_unseen", "stage3_wide"):
        m = metrics["stage2"].get(dataset)
        stage3_scores[dataset] = {
            "parse_success_rate": get(m, "anchor", "parse_success_rate"),
            "semantic_exact_match": get(m, "anchor", "semantic_exact_match"),
            "fact_micro_f1": get(m, "anchor", "fact_micro", "f1"),
            "point_micro_f1": get(m, "anchor", "point_micro", "f1"),
        }

    base_f1 = stage3_scores.get("stage3_base", {}).get("fact_micro_f1")
    for dataset, values in stage3_scores.items():
        f1 = values.get("fact_micro_f1")
        values["fact_f1_delta_vs_stage3_base"] = (
            f1 - base_f1 if f1 is not None and base_f1 is not None else None
        )

    ppl_base = get(language["base"], "wikitext_ppl", "perplexity")
    ppl_stage2 = get(language["stage2"], "wikitext_ppl", "perplexity")

    summary = {
        "integrity": integrity,
        "weight_audit": weight,
        "behavior": {
            "stage1_test_concept_accuracy": {
                model: get(metrics[model]["stage1"], "stage1_behavior", "concept_accuracy")
                for model in ("base", "stage1", "stage2")
            },
            "stage2_local_concept_accuracy": {
                model: get(metrics[model]["stage2"], "local", "concept_accuracy")
                for model in ("base", "stage1", "stage2")
            },
            "stage2_anchor_fact_micro_f1": {
                model: get(metrics[model]["stage2"], "anchor", "fact_micro", "f1")
                for model in ("base", "stage1", "stage2")
            },
            "stage2_anchor_parse_success": {
                model: get(metrics[model]["stage2"], "anchor", "parse_success_rate")
                for model in ("base", "stage1", "stage2")
            },
            "stage2_pair_signature_inconsistency": {
                model: get(metrics[model]["stage2"], "pair_consistency", "inconsistency_rate")
                for model in ("base", "stage1", "stage2")
            },
        },
        "language": {
            "base_ppl": ppl_base,
            "stage2_ppl": ppl_stage2,
            "ppl_delta_stage2_minus_base": (
                ppl_stage2 - ppl_base if ppl_base is not None and ppl_stage2 is not None else None
            ),
            "base_dsl_leak_rate": get(language["base"], "dsl_leakage", "dsl_leak_rate"),
            "stage2_dsl_leak_rate": get(language["stage2"], "dsl_leakage", "dsl_leak_rate"),
        },
        "stage3_transfer": stage3_scores,
        "representation_audit": "DEFERRED",
    }

    out_json = root / "evaluation_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Stage-2 Checkpoint Evaluation Summary",
        "",
        "## Behavior",
        "",
        "| metric | base | stage1 | stage2 |",
        "|---|---:|---:|---:|",
        "| Stage1 concept accuracy | {} | {} | {} |".format(*[
            fmt(summary["behavior"]["stage1_test_concept_accuracy"][m]) for m in ("base", "stage1", "stage2")
        ]),
        "| Stage2 local concept accuracy | {} | {} | {} |".format(*[
            fmt(summary["behavior"]["stage2_local_concept_accuracy"][m]) for m in ("base", "stage1", "stage2")
        ]),
        "| Stage2 anchor fact micro-F1 | {} | {} | {} |".format(*[
            fmt(summary["behavior"]["stage2_anchor_fact_micro_f1"][m]) for m in ("base", "stage1", "stage2")
        ]),
        "| Stage2 anchor parse success | {} | {} | {} |".format(*[
            fmt(summary["behavior"]["stage2_anchor_parse_success"][m]) for m in ("base", "stage1", "stage2")
        ]),
        "",
        "## Language",
        "",
        f"- Base WikiText-2 PPL: {fmt(summary['language']['base_ppl'])}",
        f"- Stage2 WikiText-2 PPL: {fmt(summary['language']['stage2_ppl'])}",
        f"- ΔPPL (Stage2 - Base): {fmt(summary['language']['ppl_delta_stage2_minus_base'])}",
        f"- Base DSL leak rate: {fmt(summary['language']['base_dsl_leak_rate'])}",
        f"- Stage2 DSL leak rate: {fmt(summary['language']['stage2_dsl_leak_rate'])}",
        "",
        "## Stage3 transfer",
        "",
        "| condition | parse | fact F1 | point F1 | Δ fact F1 vs base |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in ("stage3_base", "stage3_values", "stage3_unseen", "stage3_wide"):
        v = stage3_scores[dataset]
        lines.append(
            f"| {dataset} | {fmt(v['parse_success_rate'])} | {fmt(v['fact_micro_f1'])} | "
            f"{fmt(v['point_micro_f1'])} | {fmt(v['fact_f1_delta_vs_stage3_base'])} |"
        )

    lines += [
        "",
        "## Representation",
        "",
        "Deferred. No representation-probe result is included in this evaluator version.",
        "",
    ]

    out_md = root / "evaluation_summary.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary saved to {out_json}")
    print(f"Readable report saved to {out_md}")


if __name__ == "__main__":
    main()
