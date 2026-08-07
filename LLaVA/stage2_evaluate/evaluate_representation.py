#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import random
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


STAGE2_GLOB = "stage2_geometry_test_test_*.parquet"
STAGE3_BASE_GLOB = "stage3_geometry_test_base_test_*.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Forward-only representation sanity check for Stage-2 KD design. "
            "No probe is trained and no checkpoint is modified."
        )
    )
    parser.add_argument("--base-model", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--vicuna-model", default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--stage2-dir", type=Path, default=Path("./checkpoints/geometry_stage2"))
    parser.add_argument("--data-dir", type=Path, default=Path("./stage2_test_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./stage2_evaluate/results/representation"))
    parser.add_argument("--layers", nargs="+", type=int, default=[8, 16, 24, 32])
    parser.add_argument(
        "--stage2-max-samples",
        type=int,
        default=0,
        help="Maximum unique Stage2 anchor images for Experiment A; 0 means all.",
    )
    parser.add_argument(
        "--stage3-samples",
        type=int,
        default=200,
        help="Number of Stage3-base samples for Experiment B.",
    )
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def load_parquet_pattern(data_dir: Path, pattern: str) -> list[dict]:
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files matched: {data_dir / pattern}")
    rows: list[dict] = []
    for path in files:
        table = pq.read_table(path)
        shard = table.to_pylist()
        rows.extend(shard)
        print(f"Loaded {len(shard):,} rows from {path}")
    print(f"Matched {len(files)} shard(s), {len(rows):,} total rows for {pattern}")
    return rows


def decode_image(value: Any) -> Image.Image:
    if isinstance(value, bytes):
        return Image.open(BytesIO(value)).convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, str):
        return Image.open(value).convert("RGB")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    raise TypeError(f"Unsupported image field type: {type(value)!r}")


def image_key(row: dict, fallback: int) -> str:
    return str(row.get("image_sha256") or row.get("id") or row.get("problem_id") or f"row-{fallback}")


def select_stage2_anchor_rows(rows: list[dict], max_samples: int, seed: int) -> list[dict]:
    anchor_rows = [r for r in rows if str(r.get("task_kind") or "").lower() == "anchor"]
    if not anchor_rows:
        raise RuntimeError("Stage2 test contains no task_kind=anchor rows.")

    unique: dict[str, dict] = {}
    for idx, row in enumerate(anchor_rows):
        unique.setdefault(image_key(row, idx), row)
    selected = list(unique.values())

    if max_samples > 0 and len(selected) > max_samples:
        rng = random.Random(seed)
        selected = rng.sample(selected, max_samples)
    selected.sort(key=lambda r: (str(r.get("concept") or ""), str(r.get("image_sha256") or r.get("id") or "")))
    print(f"Experiment A: {len(selected):,} unique Stage2 anchor images")
    return selected


def select_stage3_base_rows(rows: list[dict], n_samples: int, seed: int) -> list[dict]:
    usable = []
    seen: set[str] = set()
    for idx, row in enumerate(rows):
        description = str(row.get("structure_description") or "").strip()
        prompt = str(row.get("prompt") or "").strip()
        if not description or not prompt:
            continue
        key = image_key(row, idx)
        if key in seen:
            continue
        seen.add(key)
        usable.append(row)
    if not usable:
        raise RuntimeError("Stage3 base has no rows with both structure_description and prompt.")
    if n_samples > 0 and len(usable) > n_samples:
        rng = random.Random(seed)
        usable = rng.sample(usable, n_samples)
    usable.sort(key=lambda r: str(r.get("image_sha256") or r.get("problem_id") or r.get("id") or ""))
    print(f"Experiment B: {len(usable):,} Stage3-base rows with structure_description")
    return usable


def build_stage2_shuffle(rows: list[dict]) -> dict[str, dict]:
    """Deterministically map every Stage2 image to an image from another concept.

    Stage2 has multiple concept groups. Cycling to the next concept avoids the
    O(N^2) retry logic used by a naive shuffled-image search and guarantees that
    the shuffled control is semantically mismatched whenever >=2 concepts exist.
    """
    by_concept: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_concept[str(row.get("concept") or "")].append(row)
    concepts = sorted(by_concept)
    mapping: dict[str, dict] = {}
    if len(concepts) <= 1:
        rotated = rows[1:] + rows[:1]
        for idx, (src, dst) in enumerate(zip(rows, rotated)):
            mapping[image_key(src, idx)] = dst
        return mapping

    for ci, concept in enumerate(concepts):
        src_group = by_concept[concept]
        dst_group = by_concept[concepts[(ci + 1) % len(concepts)]]
        for j, src in enumerate(src_group):
            mapping[image_key(src, j)] = dst_group[j % len(dst_group)]
    return mapping


def model_input_device(model: torch.nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def model_dtype(model: torch.nn.Module) -> torch.dtype:
    try:
        return next(p for p in model.parameters() if p.is_floating_point()).dtype
    except StopIteration:
        return torch.float16


def clean_anchor_prompt(row: dict) -> str:
    prompt = str(row.get("prompt") or "").strip()
    prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, "").strip()
    if not prompt:
        raise ValueError("Empty anchor prompt in dataset row.")
    return prompt


def teacher_text(row: dict, teacher_format: str) -> str:
    prompt = clean_anchor_prompt(row)
    if teacher_format == "T1":
        return prompt
    if teacher_format == "T2":
        description = str(row.get("structure_description") or "").strip()
        if not description:
            raise ValueError("T2 requires structure_description.")
        return f"{description}\n\n{prompt}"
    raise ValueError(f"Unknown teacher format: {teacher_format}")


def render_conversation(text: str, conv_mode: str, use_image: bool, model_config=None) -> str:
    text = str(text).replace(DEFAULT_IMAGE_TOKEN, "").strip()
    if use_image:
        if getattr(model_config, "mm_use_im_start_end", False):
            text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + text
        else:
            text = DEFAULT_IMAGE_TOKEN + "\n" + text
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def validate_layers(hidden_states: tuple[torch.Tensor, ...], layers: list[int]) -> None:
    # hidden_states[0] is the embedding output. hidden_states[k] is the state
    # after Transformer block k, so user-facing layers 8/16/24/32 map directly.
    max_layer = len(hidden_states) - 1
    bad = [layer for layer in layers if layer < 1 or layer > max_layer]
    if bad:
        raise ValueError(f"Requested layer(s) {bad} but model exposes transformer layers 1..{max_layer}.")


def extract_llava_states(
    row: dict,
    text: str,
    tokenizer,
    model,
    image_processor,
    layers: list[int],
    conv_mode: str,
    image_mode: str,
    image_override: Any | None = None,
) -> dict[int, torch.Tensor]:
    if image_mode not in {"normal", "blank", "none"}:
        raise ValueError(image_mode)
    use_image = image_mode != "none"
    rendered = render_conversation(text, conv_mode, use_image=use_image, model_config=model.config)
    device = model_input_device(model)

    if use_image:
        input_ids = tokenizer_image_token(
            rendered, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)
        value = row.get("image") if image_override is None else image_override
        image = decode_image(value)
        if image_mode == "blank":
            image = Image.new("RGB", image.size, (255, 255, 255))
        image_tensor = process_images([image], image_processor, model.config)[0]
        image_tensor = image_tensor.unsqueeze(0).to(device=device, dtype=model_dtype(model))
        forward_kwargs = {
            "input_ids": input_ids,
            "images": image_tensor,
            "image_sizes": [image.size],
        }
    else:
        input_ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
        forward_kwargs = {"input_ids": input_ids}

    with torch.inference_mode():
        outputs = model(
            **forward_kwargs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    hidden_states = outputs.hidden_states
    validate_layers(hidden_states, layers)
    return {
        layer: hidden_states[layer][0, -1, :].detach().float().cpu()
        for layer in layers
    }


def extract_text_states(
    text: str,
    tokenizer,
    model,
    layers: list[int],
    conv_mode: str,
) -> dict[int, torch.Tensor]:
    rendered = render_conversation(text, conv_mode, use_image=False)
    device = model_input_device(model)
    input_ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    hidden_states = outputs.hidden_states
    validate_layers(hidden_states, layers)
    return {
        layer: hidden_states[layer][0, -1, :].detach().float().cpu()
        for layer in layers
    }


def load_llava(kind: str, base_model: str, stage2_dir: Path):
    disable_torch_init()
    if kind == "base":
        model_path = base_model
        model_base = None
        model_name = "llava-v1.5-7b-base"
    elif kind == "stage2":
        if not (stage2_dir / "config.json").exists():
            raise FileNotFoundError(f"Stage2 config was not found: {stage2_dir / 'config.json'}")
        model_path = str(stage2_dir)
        model_base = None
        model_name = "llava-v1.5-7b-stage2"
    else:
        raise ValueError(kind)

    print(f"Loading LLaVA model: {kind} ({model_path})")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        load_8bit=False,
        load_4bit=False,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model, image_processor, context_len


def load_vicuna(model_name: str):
    print(f"Loading Vicuna text model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def release_model(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def stack_by_layer(storage: dict[int, list[torch.Tensor]], layers: list[int]) -> dict[int, torch.Tensor]:
    return {layer: torch.stack(storage[layer], dim=0) for layer in layers}


def norm_stats(vectors: torch.Tensor) -> dict[str, float]:
    norms = torch.linalg.vector_norm(vectors, dim=1).float()
    if norms.numel() == 0:
        return {"mean": math.nan, "std": math.nan, "p10": math.nan, "p50": math.nan, "p90": math.nan}
    q = torch.quantile(norms, torch.tensor([0.1, 0.5, 0.9]))
    return {
        "mean": float(norms.mean()),
        "std": float(norms.std(unbiased=False)),
        "p10": float(q[0]),
        "p50": float(q[1]),
        "p90": float(q[2]),
    }


def group_pair_cosine_metrics(vectors: torch.Tensor, labels: list[str], centered: bool) -> dict[str, Any]:
    if vectors.ndim != 2 or vectors.shape[0] != len(labels):
        raise ValueError("Vector/label shape mismatch.")
    x = vectors.float()
    if centered:
        x = x - x.mean(dim=0, keepdim=True)
    x = F.normalize(x, p=2, dim=1, eps=1e-12)
    n = x.shape[0]
    if n < 2:
        return {
            "n": n,
            "s_same": None,
            "s_diff": None,
            "separation": None,
            "random_cosine_baseline": None,
            "concept_s_same": {},
        }

    # Exact all-pairs cosine mean without materializing an NxN matrix:
    # sum_{i<j} vi.vj = (||sum_i vi||^2 - N) / 2 for unit vectors.
    total_pairs = n * (n - 1) // 2
    total_pair_sum = (float(torch.sum(torch.sum(x, dim=0) ** 2)) - n) / 2.0

    indices_by_label: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        indices_by_label[label].append(idx)

    same_pair_sum = 0.0
    same_pair_count = 0
    concept_s_same: dict[str, float | None] = {}
    for label, indices in sorted(indices_by_label.items()):
        if len(indices) < 2:
            concept_s_same[label] = None
            continue
        g = x[indices]
        count = len(indices) * (len(indices) - 1) // 2
        pair_sum = (float(torch.sum(torch.sum(g, dim=0) ** 2)) - len(indices)) / 2.0
        concept_s_same[label] = pair_sum / count
        same_pair_sum += pair_sum
        same_pair_count += count

    diff_pair_count = total_pairs - same_pair_count
    diff_pair_sum = total_pair_sum - same_pair_sum
    s_same = same_pair_sum / same_pair_count if same_pair_count else None
    s_diff = diff_pair_sum / diff_pair_count if diff_pair_count else None
    return {
        "n": n,
        "centered": centered,
        "s_same": s_same,
        "s_diff": s_diff,
        "separation": (s_same - s_diff) if s_same is not None and s_diff is not None else None,
        "random_cosine_baseline": total_pair_sum / total_pairs if total_pairs else None,
        "concept_s_same": concept_s_same,
    }


def cosine_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=1, eps=1e-12)


def derangement(n: int, seed: int) -> list[int]:
    if n < 2:
        return list(range(n))
    rng = random.Random(seed)
    base = list(range(n))
    for _ in range(1000):
        perm = base.copy()
        rng.shuffle(perm)
        if all(i != perm[i] for i in base):
            return perm
    # Guaranteed no fixed point fallback.
    return base[1:] + base[:1]


def alignment_metrics(teacher: torch.Tensor, student: torch.Tensor, teacher_perm: list[int]) -> dict[str, float]:
    if teacher.shape != student.shape:
        raise ValueError(f"Teacher/student shape mismatch: {teacher.shape} vs {student.shape}")
    matched = cosine_rows(teacher, student)
    mismatched = cosine_rows(teacher[teacher_perm], student)
    return {
        "matched_cosine": float(matched.mean()),
        "mismatched_cosine": float(mismatched.mean()),
        "margin": float(matched.mean() - mismatched.mean()),
        "matched_std": float(matched.std(unbiased=False)),
        "mismatched_std": float(mismatched.std(unbiased=False)),
    }


def collect_experiment_a_for_model(
    model_kind: str,
    rows: list[dict],
    shuffled_map: dict[str, dict],
    tokenizer,
    model,
    image_processor,
    layers: list[int],
    conv_mode: str,
) -> dict[str, Any]:
    labels = [str(row.get("concept") or "") for row in rows]
    normal: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    blank: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    shuffled: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

    for idx, row in enumerate(tqdm(rows, desc=f"A/{model_kind}")):
        prompt = clean_anchor_prompt(row)  # T1 = image + anchor prompt
        n_state = extract_llava_states(
            row, prompt, tokenizer, model, image_processor, layers, conv_mode, "normal"
        )
        b_state = extract_llava_states(
            row, prompt, tokenizer, model, image_processor, layers, conv_mode, "blank"
        )
        key = image_key(row, idx)
        shuffled_row = shuffled_map[key]
        s_state = extract_llava_states(
            row,
            prompt,
            tokenizer,
            model,
            image_processor,
            layers,
            conv_mode,
            "normal",
            image_override=shuffled_row.get("image"),
        )
        for layer in layers:
            normal[layer].append(n_state[layer])
            blank[layer].append(b_state[layer])
            shuffled[layer].append(s_state[layer])

    normal_s = stack_by_layer(normal, layers)
    blank_s = stack_by_layer(blank, layers)
    shuffled_s = stack_by_layer(shuffled, layers)

    result: dict[str, Any] = {"n": len(rows), "teacher_format": "T1", "layers": {}}
    for layer in layers:
        delta = normal_s[layer] - blank_s[layer]
        shuffled_delta = shuffled_s[layer] - blank_s[layer]
        result["layers"][str(layer)] = {
            "normal_minus_blank": {
                "raw": group_pair_cosine_metrics(delta, labels, centered=False),
                "centered": group_pair_cosine_metrics(delta, labels, centered=True),
                "norm": norm_stats(delta),
            },
            "shuffled_minus_blank": {
                "raw": group_pair_cosine_metrics(shuffled_delta, labels, centered=False),
                "centered": group_pair_cosine_metrics(shuffled_delta, labels, centered=True),
                "norm": norm_stats(shuffled_delta),
            },
            "state_random_cosine": {
                "h_normal": group_pair_cosine_metrics(normal_s[layer], labels, centered=False)["random_cosine_baseline"],
                "h_blank": group_pair_cosine_metrics(blank_s[layer], labels, centered=False)["random_cosine_baseline"],
                "h_shuffled": group_pair_cosine_metrics(shuffled_s[layer], labels, centered=False)["random_cosine_baseline"],
            },
        }
    return result


def collect_stage2_b(
    rows: list[dict],
    tokenizer,
    model,
    image_processor,
    layers: list[int],
    conv_mode: str,
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, dict[int, torch.Tensor]]]:
    teacher_lists = {
        "T1_h_pre": {l: [] for l in layers},
        "T1_delta_visual": {l: [] for l in layers},
        "T2_h_pre": {l: [] for l in layers},
        "T2_delta_visual": {l: [] for l in layers},
    }
    student_lists = {
        "h_pre": {l: [] for l in layers},
        "delta_text": {l: [] for l in layers},
    }

    for row in tqdm(rows, desc="B/stage2 teacher+student"):
        t1_text = teacher_text(row, "T1")
        t2_text = teacher_text(row, "T2")
        t1_normal = extract_llava_states(
            row, t1_text, tokenizer, model, image_processor, layers, conv_mode, "normal"
        )
        t1_blank = extract_llava_states(
            row, t1_text, tokenizer, model, image_processor, layers, conv_mode, "blank"
        )
        t2_normal = extract_llava_states(
            row, t2_text, tokenizer, model, image_processor, layers, conv_mode, "normal"
        )
        t2_blank = extract_llava_states(
            row, t2_text, tokenizer, model, image_processor, layers, conv_mode, "blank"
        )

        # Student input is identical for both candidates: structure_description + anchor prompt.
        desc_state = extract_llava_states(
            row, t2_text, tokenizer, model, image_processor, layers, conv_mode, "none"
        )
        empty_state = extract_llava_states(
            row, t1_text, tokenizer, model, image_processor, layers, conv_mode, "none"
        )
        for layer in layers:
            teacher_lists["T1_h_pre"][layer].append(t1_normal[layer])
            teacher_lists["T1_delta_visual"][layer].append(t1_normal[layer] - t1_blank[layer])
            teacher_lists["T2_h_pre"][layer].append(t2_normal[layer])
            teacher_lists["T2_delta_visual"][layer].append(t2_normal[layer] - t2_blank[layer])
            student_lists["h_pre"][layer].append(desc_state[layer])
            student_lists["delta_text"][layer].append(desc_state[layer] - empty_state[layer])

    teacher = {name: stack_by_layer(store, layers) for name, store in teacher_lists.items()}
    student = {name: stack_by_layer(store, layers) for name, store in student_lists.items()}
    return teacher, student


def collect_vicuna_b(
    rows: list[dict],
    tokenizer,
    model,
    layers: list[int],
    conv_mode: str,
) -> dict[str, dict[int, torch.Tensor]]:
    h_pre = {l: [] for l in layers}
    delta_text = {l: [] for l in layers}
    for row in tqdm(rows, desc="B/vicuna student"):
        prompt = clean_anchor_prompt(row)
        description = str(row.get("structure_description") or "").strip()
        desc_text = f"{description}\n\n{prompt}"
        desc_state = extract_text_states(desc_text, tokenizer, model, layers, conv_mode)
        empty_state = extract_text_states(prompt, tokenizer, model, layers, conv_mode)
        for layer in layers:
            h_pre[layer].append(desc_state[layer])
            delta_text[layer].append(desc_state[layer] - empty_state[layer])
    return {
        "h_pre": stack_by_layer(h_pre, layers),
        "delta_text": stack_by_layer(delta_text, layers),
    }


def score_experiment_b(
    teacher: dict[str, dict[int, torch.Tensor]],
    students: dict[str, dict[str, dict[int, torch.Tensor]]],
    layers: list[int],
    seed: int,
) -> dict[str, Any]:
    n = next(iter(teacher["T1_h_pre"].values())).shape[0]
    perm = derangement(n, seed)
    alignments: dict[str, Any] = {}
    for student_name, student in students.items():
        alignments[student_name] = {}
        for layer in layers:
            layer_key = str(layer)
            alignments[student_name][layer_key] = {}
            for teacher_format in ("T1", "T2"):
                alignments[student_name][layer_key][teacher_format] = {
                    "h_pre": alignment_metrics(
                        teacher[f"{teacher_format}_h_pre"][layer], student["h_pre"][layer], perm
                    ),
                    "delta": alignment_metrics(
                        teacher[f"{teacher_format}_delta_visual"][layer], student["delta_text"][layer], perm
                    ),
                }

    residual_comparison: dict[str, Any] = {}
    student_delta_norms: dict[str, Any] = {name: {} for name in students}
    for layer in layers:
        t1 = teacher["T1_delta_visual"][layer]
        t2 = teacher["T2_delta_visual"][layer]
        t1_norm = norm_stats(t1)
        t2_norm = norm_stats(t2)
        t1_t2_cos = cosine_rows(t1, t2)
        ratio = (
            t2_norm["mean"] / t1_norm["mean"]
            if t1_norm["mean"] not in (0.0, None) and math.isfinite(t1_norm["mean"])
            else None
        )
        residual_comparison[str(layer)] = {
            "T1_delta_visual_norm": t1_norm,
            "T2_delta_visual_norm": t2_norm,
            "T2_over_T1_mean_norm_ratio": ratio,
            "T1_vs_T2_delta_visual_cosine": float(t1_t2_cos.mean()),
        }
        for student_name, student in students.items():
            student_delta_norms[student_name][str(layer)] = norm_stats(student["delta_text"][layer])

    return {
        "n": n,
        "mismatched_definition": "Teacher rows are deterministically deranged; student rows stay fixed.",
        "alignments": alignments,
        "teacher_delta_visual_comparison": residual_comparison,
        "student_delta_text_norm": student_delta_norms,
    }


def best_layer_by_stage2_centered_separation(experiment_a: dict[str, Any]) -> dict[str, Any] | None:
    stage2 = experiment_a.get("models", {}).get("stage2", {})
    best = None
    for layer, data in stage2.get("layers", {}).items():
        sep = (
            data.get("normal_minus_blank", {})
            .get("centered", {})
            .get("separation")
        )
        if sep is None:
            continue
        if best is None or sep > best["separation"]:
            best = {"layer": int(layer), "separation": sep}
    return best


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layers = sorted(set(args.layers))
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    stage2_rows = select_stage2_anchor_rows(
        load_parquet_pattern(args.data_dir, STAGE2_GLOB),
        args.stage2_max_samples,
        args.seed,
    )
    stage3_rows = select_stage3_base_rows(
        load_parquet_pattern(args.data_dir, STAGE3_BASE_GLOB),
        args.stage3_samples,
        args.seed,
    )
    shuffle_map = build_stage2_shuffle(stage2_rows)

    experiment_a = {
        "definition": {
            "teacher_format": "T1 = image + anchor prompt",
            "delta_visual": "h_normal - h_blank",
            "centered_delta_visual": "delta_visual - dataset_mean(delta_visual)",
            "shuffled_control": "h_shuffled - h_blank, scored against the source concept",
            "layer_indexing": "hidden_states[k] = output after Transformer block k; embedding output is index 0",
        },
        "models": {},
    }

    # A/base: original LLaVA representation.
    base_tok, base_model, base_proc, _ = load_llava("base", args.base_model, args.stage2_dir)
    experiment_a["models"]["llava-v1.5-7b"] = collect_experiment_a_for_model(
        "llava-v1.5-7b",
        stage2_rows,
        shuffle_map,
        base_tok,
        base_model,
        base_proc,
        layers,
        args.conv_mode,
    )
    del base_model, base_tok, base_proc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # A/stage2 and B/teacher + Stage2-text student reuse one Stage2 load.
    s2_tok, s2_model, s2_proc, _ = load_llava("stage2", args.base_model, args.stage2_dir)
    experiment_a["models"]["stage2"] = collect_experiment_a_for_model(
        "stage2",
        stage2_rows,
        shuffle_map,
        s2_tok,
        s2_model,
        s2_proc,
        layers,
        args.conv_mode,
    )
    teacher_b, stage2_student_b = collect_stage2_b(
        stage3_rows, s2_tok, s2_model, s2_proc, layers, args.conv_mode
    )
    del s2_model, s2_tok, s2_proc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # B/Vicuna student.
    vicuna_tok, vicuna_model = load_vicuna(args.vicuna_model)
    vicuna_student_b = collect_vicuna_b(
        stage3_rows, vicuna_tok, vicuna_model, layers, args.conv_mode
    )
    del vicuna_model, vicuna_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    experiment_b = score_experiment_b(
        teacher_b,
        {
            "vicuna-7b-v1.5": vicuna_student_b,
            "stage2-text": stage2_student_b,
        },
        layers,
        args.seed + 1000,
    )
    experiment_b["definition"] = {
        "teacher_T1": "Stage2 + image + anchor prompt",
        "teacher_T2": "Stage2 + image + structure_description + anchor prompt",
        "teacher_delta_visual": "h_normal - h_blank for the same teacher text",
        "student_input": "structure_description + anchor prompt; identical text for Vicuna and Stage2-text",
        "student_delta_text": "h(description + prompt) - h(prompt only)",
        "full_state_alignment": "teacher h_pre vs student h_pre",
        "residual_alignment": "teacher delta_visual vs student delta_text",
    }

    result = {
        "metadata": {
            "base_model": args.base_model,
            "vicuna_model": args.vicuna_model,
            "stage2_dir": str(args.stage2_dir),
            "layers": layers,
            "stage2_samples": len(stage2_rows),
            "stage3_base_samples": len(stage3_rows),
            "seed": args.seed,
            "forward_only": True,
            "checkpoint_updates": False,
        },
        "experiment_a": experiment_a,
        "experiment_b": experiment_b,
        "quick_decision": {
            "best_stage2_layer_by_centered_visual_separation": best_layer_by_stage2_centered_separation(experiment_a),
        },
    }

    out = args.output_dir / "representation.json"
    out.write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Representation audit saved to {out}")
    best = result["quick_decision"]["best_stage2_layer_by_centered_visual_separation"]
    if best:
        print(
            "Best Stage2 layer by centered (normal-blank) separation: "
            f"layer {best['layer']} / {best['separation']:.4f}"
        )


if __name__ == "__main__":
    main()
