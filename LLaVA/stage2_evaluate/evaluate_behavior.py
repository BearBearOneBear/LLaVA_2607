#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
<<<<<<< HEAD
import random
=======
>>>>>>> 499470cfa0cb6010a9ddbc450ed1509fba3563c8
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


DATASET_GLOBS = {
    "stage1": "stage1_geometry_test_test_*.parquet",
    "stage2": "stage2_geometry_test_test_*.parquet",
    "stage3_base": "stage3_geometry_test_base_test_*.parquet",
    "stage3_values": "stage3_geometry_test_values_test_*.parquet",
    "stage3_unseen": "stage3_geometry_test_unseen_test_*.parquet",
    "stage3_wide": "stage3_geometry_test_wide_test_*.parquet",
}

# Explicit aliases are used only when the model answers in natural language.
# The evaluator still reads the actual concept names from the parquet files.
CONCEPT_ALIASES = {
    "triangle_altitude": ["triangle altitude", "altitude"],
    "triangle_median": ["triangle median", "median"],
<<<<<<< HEAD
    # The Stage 2 template reads "FI bisects angle DFJ", which contains neither
    # "angle bisector" nor "triangle angle bisector". Without the verb form this
    # concept is never detected and caps local accuracy at 26/27.
    "triangle_angle_bisector": ["triangle angle bisector", "angle bisector", "bisects angle"],
=======
    "triangle_angle_bisector": ["triangle angle bisector", "angle bisector"],
>>>>>>> 499470cfa0cb6010a9ddbc450ed1509fba3563c8
    "triangle_perpendicular_bisector": ["triangle perpendicular bisector", "perpendicular bisector"],
    "triangle_centroid": ["triangle centroid", "centroid"],
    "triangle_circumcenter": ["triangle circumcenter", "circumcenter"],
    "triangle_incenter": ["triangle incenter", "incenter"],
    "isosceles_triangle": ["isosceles triangle", "isosceles"],
    "right_triangle": ["right triangle"],
    "equilateral_triangle": ["equilateral triangle", "equilateral"],
    "corresponding_angles": ["corresponding angles", "corresponding angle"],
    "alternate_interior_angles": ["alternate interior angles", "alternate interior angle"],
    "same_side_interior_angles": ["same side interior angles", "same-side interior angles", "same side interior angle"],
    "vertical_angles": ["vertical angles", "vertical angle", "vertically opposite angles"],
    "parallelogram": ["parallelogram"],
    "rectangle": ["rectangle"],
    "rhombus": ["rhombus"],
    "square": ["square"],
    "trapezoid": ["trapezoid", "trapezium"],
    "circle_radius": ["circle radius", "radius"],
    "circle_diameter": ["circle diameter", "diameter"],
    "circle_chord": ["circle chord", "chord"],
    "circle_sector": ["circle sector", "sector"],
    "circle_tangent": ["circle tangent", "tangent"],
    "central_angle": ["central angle"],
    "inscribed_angle": ["inscribed angle"],
    "cyclic_quadrilateral": ["cyclic quadrilateral", "inscribed quadrilateral"],
}


@dataclass
class ParsedAnchor:
    parsed: bool
    tags: dict[str, list[str]]
    facts_by_tag: dict[str, set[str]]
    all_facts: set[str]
    points: set[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Forward-only behavior and Stage-3 transfer evaluator for the geometry "
            "Stage-2 checkpoint. One process loads exactly one model."
        )
    )
    parser.add_argument("--model-kind", choices=["base", "stage1", "stage2"], required=True)
    parser.add_argument("--base-model", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--stage1-dir", type=Path, default=Path("./checkpoints/geometry_stage1"))
    parser.add_argument("--stage2-dir", type=Path, default=Path("./checkpoints/geometry_stage2"))
    parser.add_argument("--data-dir", type=Path, default=Path("./stage2_test_data"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_GLOBS),
        default=["stage1", "stage2"],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./stage2_evaluate/results"))
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--max-new-tokens-local", type=int, default=128)
    parser.add_argument("--max-new-tokens-anchor", type=int, default=768)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples per dataset.")
    parser.add_argument("--resume", action="store_true")
<<<<<<< HEAD
    parser.add_argument(
        "--image-mode",
        choices=["normal", "shuffled", "blank", "none"],
        default="normal",
        help=(
            "Image ablation applied only at evaluation time: normal=original image; "
            "shuffled=another image from the same dataset; blank=white image with the same size; "
            "none=text-only prompt with no image token or image tensor."
        ),
    )
=======
>>>>>>> 499470cfa0cb6010a9ddbc450ed1509fba3563c8
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pair-signature-threshold",
        type=float,
        default=0.80,
        help="Ground-truth anchor-tag frequency required for a concept signature.",
    )
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items() if k != "bytes"}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def load_parquet_records(data_dir: Path, dataset_name: str) -> list[dict]:
    pattern = DATASET_GLOBS[dataset_name]
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files matched {data_dir / pattern}")

    records: list[dict] = []
    for path in files:
<<<<<<< HEAD
        # A shard that fails to open is raised rather than skipped. Committing
        # parquet through a text-mode filter truncates it to a couple of bytes,
        # and a skipped shard would quietly shrink the evaluation set instead of
        # reporting a broken input.
        try:
            table = pq.read_table(path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read {path}: {type(exc).__name__}: {exc}. "
                "If the file is only a few bytes, it was likely corrupted in "
                "transit; add '*.parquet binary' to .gitattributes and recommit."
            ) from exc

        shard_records = table.to_pylist()
        records.extend(shard_records)
        print(f"Loaded {len(shard_records):,} rows from {path}")

    if not records:
        raise RuntimeError(
            f"Dataset {dataset_name} matched {len(files)} file(s) but contains no rows."
        )

=======
        table = pq.read_table(path)
        shard_records = table.to_pylist()
        records.extend(shard_records)
        print(f"Loaded {len(shard_records):,} rows from {path}")
>>>>>>> 499470cfa0cb6010a9ddbc450ed1509fba3563c8
    print(f"Dataset {dataset_name}: {len(records):,} total rows from {len(files)} shard(s).")
    return records


def get_sample_id(row: dict, fallback_idx: int) -> str:
    value = row.get("id")
    if value is None:
        value = row.get("problem_id")
    return str(value if value is not None else fallback_idx)


def decode_image(image_value: Any) -> Image.Image:
    if isinstance(image_value, bytes):
        return Image.open(BytesIO(image_value)).convert("RGB")
    if isinstance(image_value, dict):
        if image_value.get("bytes") is not None:
            return Image.open(BytesIO(image_value["bytes"])).convert("RGB")
        if image_value.get("path"):
            return Image.open(image_value["path"]).convert("RGB")
    if isinstance(image_value, str):
        return Image.open(image_value).convert("RGB")
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")
    raise TypeError(f"Unsupported image field type: {type(image_value)!r}")


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} was not found: {path}")


def load_eval_model(args: argparse.Namespace):
    disable_torch_init()

    if args.model_kind == "base":
        model_path = args.base_model
        model_base = None
        model_name = "llava-v1.5-7b-base"
    elif args.model_kind == "stage1":
        require_path(args.stage1_dir / "mm_projector.bin", "Stage-1 projector")
        require_path(args.stage1_dir / "config.json", "Stage-1 config")
        model_path = str(args.stage1_dir)
        model_base = args.base_model
        # builder.py needs the word "llava" in model_name to enter the multimodal path.
        model_name = "llava-v1.5-7b-stage1"
    else:
        require_path(args.stage2_dir / "config.json", "Stage-2 config")
        model_path = str(args.stage2_dir)
        model_base = None
        # Do not use get_model_name_from_path("geometry_stage2") here: it would not
        # contain "llava" and the upstream builder would choose the text-only branch.
        model_name = "llava-v1.5-7b-stage2"

    print(f"Loading model kind={args.model_kind}")
    print(f"model_path={model_path}")
    print(f"model_base={model_base}")

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


def generate_one(
    row: dict,
    tokenizer,
    model,
    image_processor,
    conv_mode: str,
    max_new_tokens: int,
<<<<<<< HEAD
    image_mode: str = "normal",
    image_override: Any | None = None,
=======
>>>>>>> 499470cfa0cb6010a9ddbc450ed1509fba3563c8
) -> str:
    user_prompt = str(row.get("prompt") or "").strip()
    if not user_prompt:
        raise ValueError(f"Row {row.get('id')} has an empty prompt.")

    qs = user_prompt.replace(DEFAULT_IMAGE_TOKEN, "").strip()
<<<<<<< HEAD
    use_image = image_mode != "none"
    if use_image:
        if getattr(model.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
=======
    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
>>>>>>> 499470cfa0cb6010a9ddbc450ed1509fba3563c8

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    device = model_input_device(model)
<<<<<<< HEAD
    if use_image:
        input_ids = tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(device)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generate_kwargs = dict(
        do_sample=False,
        temperature=0.0,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )

    if use_image:
        image_value = row["image"] if image_override is None else image_override
        image = decode_image(image_value)
        if image_mode == "blank":
            image = Image.new("RGB", image.size, (255, 255, 255))
        image_tensor = process_images([image], image_processor, model.config)[0]
        image_tensor = image_tensor.unsqueeze(0).to(device=device, dtype=model_dtype(model))
        generate_kwargs["images"] = image_tensor
        generate_kwargs["image_sizes"] = [image.size]

    with torch.inference_mode():
        output_ids = model.generate(input_ids, **generate_kwargs)
=======
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(device)

    image = decode_image(row["image"])
    image_tensor = process_images([image], image_processor, model.config)[0]
    image_tensor = image_tensor.unsqueeze(0).to(device=device, dtype=model_dtype(model))

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=False,
            temperature=0.0,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
>>>>>>> 499470cfa0cb6010a9ddbc450ed1509fba3563c8

    text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return text


<<<<<<< HEAD
def build_shuffled_image_map(rows: list[dict], seed: int) -> dict[str, tuple[Any, str]]:
    """Map each source image hash to a different image from the same dataset.

    Local/anchor rows sharing one image get the same shuffled replacement, so pair
    consistency stays measurable. The mapping is deterministic for a fixed seed and
    rotates a shuffled list of unique images, so nothing maps to itself.

    A plain rotation lands on an image of the same concept about one time in
    twenty-seven, and those cases score as if the model had read the diagram
    correctly, inflating the ablation. Rotations are therefore retried within each
    concept group until the replacement's concept differs, which is what makes the
    shuffled condition a floor rather than a slightly softened normal condition.
    """
    representatives: dict[str, Any] = {}
    concepts: dict[str, str] = {}
    for idx, row in enumerate(rows):
        key = str(row.get("image_sha256") or f"__row_{idx}")
        representatives.setdefault(key, row.get("image"))
        concepts.setdefault(key, str(row.get("concept") or ""))

    keys = list(representatives)
    if len(keys) < 2:
        return {k: (representatives[k], k) for k in keys}

    rng = random.Random(seed)
    rng.shuffle(keys)

    mapping: dict[str, str] = {}
    for offset in range(1, len(keys)):
        remaining = [k for k in keys if k not in mapping]
        if not remaining:
            break
        for src in remaining:
            dst = keys[(keys.index(src) + offset) % len(keys)]
            if dst == src:
                continue
            if concepts.get(dst) and concepts[dst] == concepts.get(src):
                continue
            mapping[src] = dst

    # Datasets with a single concept, or a handful of images, may leave a few
    # sources unmatched. Falling back to the plain rotation keeps every row
    # scored rather than dropping it.
    for index, src in enumerate(keys):
        if src not in mapping:
            mapping[src] = keys[(index + 1) % len(keys)]

    return {src: (representatives[dst], dst) for src, dst in mapping.items()}


=======
>>>>>>> 499470cfa0cb6010a9ddbc450ed1509fba3563c8
def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def normalized_space(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def normalize_concept_text(text: str) -> str:
    text = text.lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def aliases_for_concept(concept: str) -> list[str]:
    aliases = set(CONCEPT_ALIASES.get(concept, []))
    aliases.add(concept.replace("_", " "))
    return sorted({normalize_concept_text(a) for a in aliases if a}, key=len, reverse=True)


def predict_concept(text: str, concepts: Iterable[str]) -> str | None:
<<<<<<< HEAD
    """Concept named by a sentence, or None when none is recognisable.

    A trailing plural is tolerated. Stage 1's point template is plural whenever
    the figure holds more than one point ("M, J, G, and L are points"), which is
    90% of that class, and an exact-form match would score it as unrecognised
    rather than wrong. Aliases are matched against the record's own concept list,
    so the loosened form cannot pull a Stage 2 concept toward a Stage 1 one.
    """
=======
>>>>>>> 499470cfa0cb6010a9ddbc450ed1509fba3563c8
    normalized = normalize_concept_text(text)
    matches: list[tuple[int, str]] = []
    for concept in concepts:
        for alias in aliases_for_concept(concept):
<<<<<<< HEAD
            if re.search(rf"(?:^|\s){re.escape(alias)}s?(?:$|\s)", normalized):
=======
            if re.search(rf"(?:^|\s){re.escape(alias)}(?:$|\s)", normalized):
>>>>>>> 499470cfa0cb6010a9ddbc450ed1509fba3563c8
                matches.append((len(alias), concept))
                break
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][1]


def expected_points(row: dict) -> set[str]:
    value = row.get("point_labels")
    if isinstance(value, str):
        # Supports either "A B C" or JSON-ish strings.
        labels = re.findall(r"\b[A-Z]\b", value.upper())
        return set(labels)
    if isinstance(value, (list, tuple, set)):
        return {str(x).strip().upper() for x in value if str(x).strip()}
    return set()


<<<<<<< HEAD
# Longest run of concatenated point labels in the Stage 2 templates. Quadrilateral
# concepts name four vertices at once ("ARJL is a parallelogram"), and roughly two
# thirds of all runs are three or four letters long, so a two-letter bound drops
# most of them.
MAX_LABEL_RUN = 4


def extract_point_labels(text: str) -> set[str]:
    """Point labels named in a sentence.

    No stop-word filter is applied. Two-letter English words are also valid label
    pairs -- "AI is a diameter of the circle" is a real template -- so filtering
    them would silently discard genuine labels. The surrounding prose is
    lowercase in every template, which leaves an all-caps run unambiguous.
    """
    # Remove an initial English article so "A right triangle ..." does not become point A.
    cleaned = re.sub(r"^\s*A\s+(?=[a-z])", "", str(text or ""))
    tokens = re.findall(rf"\b[A-Z]{{1,{MAX_LABEL_RUN}}}\b", cleaned)
    result: set[str] = set()
    for token in tokens:
        result.update(token)
=======
def extract_point_labels(text: str) -> set[str]:
    # Remove an initial English article so "A right triangle ..." does not become point A.
    cleaned = re.sub(r"^\s*A\s+(?=[a-z])", "", str(text or ""))
    tokens = re.findall(r"\b[A-Z]{1,2}\b", cleaned)
    ignored = {"AI", "AN", "OR", "TO", "IS", "IN", "OF", "ON", "BY", "AT", "AS"}
    result: set[str] = set()
    for token in tokens:
        if token in ignored:
            continue
        if len(token) == 1:
            result.add(token)
        else:
            result.update(token)
>>>>>>> 499470cfa0cb6010a9ddbc450ed1509fba3563c8
    return result


def canonical_segment(token: str) -> str:
    token = re.sub(r"[^A-Z]", "", token.upper())
    if len(token) == 2:
        return "".join(sorted(token))
    return token


def split_function_calls(payload: str) -> list[tuple[str, list[str]]]:
    calls = []
    for name, args_text in re.findall(r"([A-Z][A-Z0-9_]*)\s*\(([^()]*)\)", payload.upper()):
        args = [a.strip() for a in args_text.split(",") if a.strip()]
        calls.append((name, args))
    return calls


def parse_anchor(text: str) -> ParsedAnchor:
    raw = str(text or "").replace("```", "")
    # A generated model occasionally uses semicolons instead of line breaks.
    raw = raw.replace("; ", "\n")

    tags: dict[str, list[str]] = defaultdict(list)
    for line in raw.splitlines():
        match = re.match(r"^\s*([A-Z][A-Z0-9_]*)\s*:\s*(.*?)\s*$", line)
        if match:
            tags[match.group(1)].append(match.group(2))

    if not tags:
        return ParsedAnchor(False, {}, {}, set(), set())

    facts_by_tag: dict[str, set[str]] = defaultdict(set)
    points: set[str] = set()

    for tag, payloads in tags.items():
        for payload in payloads:
            p = normalized_space(payload.upper())
            if not p:
                continue

            if tag == "POINTS":
                pts = set(re.findall(r"\b[A-Z]\b", p))
                points.update(pts)
                facts_by_tag[tag].update({f"POINT:{x}" for x in pts})
                continue

            if tag == "SEG":
                segs = re.findall(r"\b[A-Z]{2}\b", p)
                facts_by_tag[tag].update({f"SEG:{canonical_segment(s)}" for s in segs})
                continue

            # Marker IDs are arbitrary names for equality/parallel groups. Compare the
            # groups of segments, not the numeric marker IDs themselves.
            if tag in {"PARA", "EQ"}:
                marker_calls = re.findall(r"(?:MARK|TICK)\s*\(\s*([A-Z]{2})\s*,\s*([0-9]+)\s*\)", p)
                if marker_calls:
                    groups: dict[str, set[str]] = defaultdict(set)
                    for seg, group_id in marker_calls:
                        groups[group_id].add(canonical_segment(seg))
                    for segs in groups.values():
                        facts_by_tag[tag].add(f"{tag}:GROUP:" + "|".join(sorted(segs)))
                    continue

            calls = split_function_calls(p)
            if calls:
                for name, args in calls:
                    norm_args = list(args)
                    if name == "PERP" and len(norm_args) >= 2:
                        pair = sorted(canonical_segment(x) for x in norm_args[:2])
                        facts_by_tag[tag].add(f"{tag}:{name}({','.join(pair)})")
                    elif name == "ON" and len(norm_args) >= 2:
                        facts_by_tag[tag].add(
                            f"{tag}:{name}({norm_args[0]},{canonical_segment(norm_args[1])})"
                        )
                    elif name == "CIRCLE":
                        facts_by_tag[tag].add(f"{tag}:{name}({','.join(sorted(norm_args))})")
                    elif name in {"CENTER", "NUM", "ARC", "ARC2", "SECTOR", "SECTOR2", "EQPTS"}:
                        facts_by_tag[tag].add(f"{tag}:{name}({','.join(norm_args)})")
                    else:
                        facts_by_tag[tag].add(f"{tag}:{name}({','.join(norm_args)})")
                continue

            # Fallback for canonical tokens without parentheses.
            for token in p.split():
                facts_by_tag[tag].add(f"{tag}:RAW:{token}")

    all_facts = set().union(*facts_by_tag.values()) if facts_by_tag else set()
    return ParsedAnchor(True, dict(tags), dict(facts_by_tag), all_facts, points)


def prf(pred: set[str], gold: set[str]) -> dict[str, float | int]:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else (1.0 if not gold else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def aggregate_prf(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn = counts.get("tp", 0), counts.get("fp", 0), counts.get("fn", 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def score_concept_records(records: list[dict]) -> dict[str, Any]:
    concepts = sorted({str(r.get("gold_concept")) for r in records if r.get("gold_concept")})
    correct = 0
    parsed = 0
    point_counts = Counter()
    confusion = Counter()
    per_concept = defaultdict(lambda: Counter(total=0, correct=0))

    for row in records:
        gold = row.get("gold_concept")
        if not gold:
            continue
        pred = predict_concept(row.get("prediction", ""), concepts)
        row["predicted_concept"] = pred
        per_concept[gold]["total"] += 1
        if pred is not None:
            parsed += 1
        if pred == gold:
            correct += 1
            per_concept[gold]["correct"] += 1
        confusion[(gold, pred or "<unparsed>")] += 1

<<<<<<< HEAD
        # The gold is the labels the reference sentence names, not every label in
        # the figure. A local answer describes one relation and mentions only the
        # points it involves: "C is the centroid of triangle JWM" names four of
        # the figure's six points, and the angle concepts ("∠1 and ∠2 are
        # vertical angles") name none at all. Scoring against the full label set
        # makes recall unreachable for those concepts by construction.
        gold_pts = extract_point_labels(row.get("gold_answer", ""))
=======
        gold_pts = expected_points({"point_labels": row.get("gold_point_labels")})
>>>>>>> 499470cfa0cb6010a9ddbc450ed1509fba3563c8
        pred_pts = extract_point_labels(row.get("prediction", ""))
        p = prf(pred_pts, gold_pts)
        point_counts["tp"] += p["tp"]
        point_counts["fp"] += p["fp"]
        point_counts["fn"] += p["fn"]
        if pred == gold:
            point_counts["concept_correct"] += 1
            if pred_pts == gold_pts:
                point_counts["conditional_exact"] += 1

    n = sum(v["total"] for v in per_concept.values())
    return {
        "n": n,
        "concept_accuracy": correct / n if n else 0.0,
        "concept_parse_rate": parsed / n if n else 0.0,
        "point_label_micro": aggregate_prf(point_counts),
        "point_label_exact_given_concept_correct": (
            point_counts["conditional_exact"] / point_counts["concept_correct"]
            if point_counts["concept_correct"]
            else None
        ),
        "per_concept_accuracy": {
            c: v["correct"] / v["total"] if v["total"] else 0.0
            for c, v in sorted(per_concept.items())
        },
        "confusion": [
            {"gold": g, "pred": p, "count": count}
            for (g, p), count in confusion.most_common()
        ],
    }


def score_anchor_records(records: list[dict]) -> dict[str, Any]:
    parse_ok = 0
    semantic_exact = 0
    surface_exact = 0
    micro_counts = Counter()
    point_counts = Counter()
    per_tag_counts: dict[str, Counter] = defaultdict(Counter)
    tag_gold_support = Counter()

    for row in records:
        gold_text = row.get("gold_canonical_answer") or row.get("gold_answer") or ""
        pred_text = row.get("prediction", "")
        gold = parse_anchor(gold_text)
        pred = parse_anchor(pred_text)
        row["pred_anchor_parsed"] = pred.parsed
        row["pred_anchor_tags"] = sorted(k for k, v in pred.facts_by_tag.items() if v)

        if pred.parsed:
            parse_ok += 1
        if pred.all_facts == gold.all_facts and pred.parsed and gold.parsed:
            semantic_exact += 1
        if normalized_space(pred_text).upper() == normalized_space(gold_text).upper():
            surface_exact += 1

        m = prf(pred.all_facts, gold.all_facts)
        micro_counts.update({k: int(m[k]) for k in ("tp", "fp", "fn")})

        pm = prf(pred.points, gold.points)
        point_counts.update({k: int(pm[k]) for k in ("tp", "fp", "fn")})

        for tag in set(gold.facts_by_tag) | set(pred.facts_by_tag):
            g = gold.facts_by_tag.get(tag, set())
            p = pred.facts_by_tag.get(tag, set())
            if g:
                tag_gold_support[tag] += len(g)
            tm = prf(p, g)
            per_tag_counts[tag].update({k: int(tm[k]) for k in ("tp", "fp", "fn")})

    n = len(records)
    per_tag = {tag: aggregate_prf(cnt) for tag, cnt in sorted(per_tag_counts.items())}
    supported_f1 = [per_tag[tag]["f1"] for tag in per_tag if tag_gold_support[tag] > 0]
    return {
        "n": n,
        "parse_success_rate": parse_ok / n if n else 0.0,
        "semantic_exact_match": semantic_exact / n if n else 0.0,
        "surface_exact_match": surface_exact / n if n else 0.0,
        "fact_micro": aggregate_prf(micro_counts),
        "point_micro": aggregate_prf(point_counts),
        "fact_macro_f1_over_supported_tags": sum(supported_f1) / len(supported_f1) if supported_f1 else 0.0,
        "per_tag": per_tag,
    }


def compute_pair_signature_inconsistency(
    records: list[dict],
    threshold: float,
) -> dict[str, Any]:
    by_image: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in records:
        image_key = str(row.get("image_sha256") or "")
        task_kind = str(row.get("task_kind") or "")
        if image_key and task_kind in {"local", "anchor"}:
            by_image[image_key][task_kind] = row

    # Learn concept -> stable anchor-tag signature from the ground truth of paired test examples.
    concept_tag_counts: dict[str, Counter] = defaultdict(Counter)
    concept_pair_counts = Counter()
    for pair in by_image.values():
        if "local" not in pair or "anchor" not in pair:
            continue
        concept = pair["local"].get("gold_concept")
        gold_anchor = parse_anchor(
            pair["anchor"].get("gold_canonical_answer") or pair["anchor"].get("gold_answer") or ""
        )
        if not concept or not gold_anchor.parsed:
            continue
        active_tags = {
            tag for tag, facts in gold_anchor.facts_by_tag.items()
            if facts and tag not in {"POINTS", "SEG"}
        }
        concept_pair_counts[concept] += 1
        concept_tag_counts[concept].update(active_tags)

    signatures: dict[str, set[str]] = {}
    for concept, counts in concept_tag_counts.items():
        denom = concept_pair_counts[concept]
        signatures[concept] = {
            tag for tag, count in counts.items() if denom and count / denom >= threshold
        }

    comparable = 0
    consistent = 0
    per_concept = defaultdict(lambda: Counter(comparable=0, consistent=0))
    for pair in by_image.values():
        if "local" not in pair or "anchor" not in pair:
            continue
        local = pair["local"]
        anchor = pair["anchor"]
        pred_concept = local.get("predicted_concept")
        pred_anchor = parse_anchor(anchor.get("prediction", ""))
        required = signatures.get(pred_concept or "", set())
        if not pred_concept or not pred_anchor.parsed or not required:
            continue
        comparable += 1
        per_concept[pred_concept]["comparable"] += 1
        active_pred_tags = {tag for tag, facts in pred_anchor.facts_by_tag.items() if facts}
        if required.issubset(active_pred_tags):
            consistent += 1
            per_concept[pred_concept]["consistent"] += 1

    total_pairs = sum(1 for p in by_image.values() if "local" in p and "anchor" in p)
    return {
        "definition": (
            "Data-driven coarse consistency: predicted local concept implies the stable set of "
            "ground-truth anchor tags observed for that concept; those tags must be present in "
            "the paired predicted anchor. This is not a full logical contradiction checker."
        ),
        "signature_frequency_threshold": threshold,
        "total_local_anchor_pairs": total_pairs,
        "comparable_pairs": comparable,
        "comparable_coverage": comparable / total_pairs if total_pairs else 0.0,
        "consistency_rate": consistent / comparable if comparable else None,
        "inconsistency_rate": 1.0 - consistent / comparable if comparable else None,
        "signatures": {k: sorted(v) for k, v in sorted(signatures.items())},
        "per_concept": {
            c: {
                "comparable": v["comparable"],
                "consistency_rate": v["consistent"] / v["comparable"] if v["comparable"] else None,
            }
            for c, v in sorted(per_concept.items())
        },
    }


def point_count_bin(value: Any) -> str:
    try:
        n = int(value)
    except Exception:
        return "unknown"
    if n <= 4:
        return "2-4"
    if n <= 6:
        return "5-6"
    if n <= 8:
        return "7-8"
    if n <= 10:
        return "9-10"
    return "11-12+"


<<<<<<< HEAD
def bool_bucket(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return "true"
    if text in {"0", "false", "no", "n", "off", "", "none", "null"}:
        return "false"
    return "unknown"


def nonzero_bucket(value: Any) -> str:
    try:
        return "present" if float(value) > 0 else "absent"
    except Exception:
        if value is None:
            return "unknown"
        return "present" if str(value).strip() else "absent"


def max_collinear_bucket(value: Any) -> str:
    try:
        n = int(value)
    except Exception:
        return "unknown"
    # Any three or more points on one line constitute a non-trivial collinear set.
    return "has_3plus_collinear" if n >= 3 else "no_3plus_collinear"


def parse_symbol_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, dict):
        return [str(k).strip() for k, v in value.items() if v and str(k).strip()]
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "[]", "{}"}:
        return []
    try:
        parsed = json.loads(text)
        if parsed is not value:
            return parse_symbol_list(parsed)
    except Exception:
        pass
    # Handles comma-separated strings and Python-list-like strings.
    parts = re.split(r"[,;|]", text.strip("[](){}"))
    return [p.strip().strip("'\"") for p in parts if p.strip().strip("'\"")]


def grouped_anchor_metrics(records: list[dict], key_fn) -> dict[str, Any]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        groups[str(key_fn(row))].append(row)
    return {name: score_anchor_records(rows) for name, rows in sorted(groups.items()) if rows}


def stage3_stratified_metrics(records: list[dict]) -> dict[str, Any]:
    """Break Stage-3 anchor performance down along the controlled metadata axes."""
    out: dict[str, Any] = {
        "point_count_bin": grouped_anchor_metrics(
            records, lambda r: point_count_bin(r.get("gold_point_count"))
        ),
        "ignored_symbol_count": grouped_anchor_metrics(
            records, lambda r: nonzero_bucket(r.get("ignored_symbol_count"))
        ),
        "unseen_symbols": grouped_anchor_metrics(
            records, lambda r: "present" if parse_symbol_list(r.get("unseen_symbols")) else "absent"
        ),
        "max_collinear": grouped_anchor_metrics(
            records, lambda r: max_collinear_bucket(r.get("max_collinear"))
        ),
        "values_in_diagram": grouped_anchor_metrics(
            records, lambda r: bool_bucket(r.get("values_in_diagram"))
        ),
    }

    # Non-exclusive per-symbol slices are useful inside the unseen variant.
    symbol_rows: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        for symbol in parse_symbol_list(row.get("unseen_symbols")):
            symbol_rows[symbol].append(row)
    if symbol_rows:
        out["unseen_symbol_type"] = {
            symbol: score_anchor_records(rows)
            for symbol, rows in sorted(symbol_rows.items())
        }
    return out


=======
>>>>>>> 499470cfa0cb6010a9ddbc450ed1509fba3563c8
def score_dataset(records: list[dict], dataset_name: str, pair_threshold: float) -> dict[str, Any]:
    if dataset_name == "stage1":
        return {"dataset": dataset_name, "stage1_behavior": score_concept_records(records)}

    if dataset_name == "stage2":
        local = [r for r in records if str(r.get("task_kind")) == "local"]
        anchor = [r for r in records if str(r.get("task_kind")) == "anchor"]
        # score_concept_records annotates local records with predicted_concept; run it first.
        local_metrics = score_concept_records(local)
        anchor_metrics = score_anchor_records(anchor)
        pair_metrics = compute_pair_signature_inconsistency(records, pair_threshold)
        return {
            "dataset": dataset_name,
            "local": local_metrics,
            "anchor": anchor_metrics,
            "pair_consistency": pair_metrics,
        }

<<<<<<< HEAD
    metrics = {
        "dataset": dataset_name,
        "anchor": score_anchor_records(records),
        "stratified": stage3_stratified_metrics(records),
    }
=======
    metrics = {"dataset": dataset_name, "anchor": score_anchor_records(records)}
    if dataset_name == "stage3_wide":
        bins = defaultdict(list)
        for row in records:
            bins[point_count_bin(row.get("gold_point_count"))].append(row)
        metrics["point_count_bins"] = {
            name: score_anchor_records(rows) for name, rows in sorted(bins.items())
        }
>>>>>>> 499470cfa0cb6010a9ddbc450ed1509fba3563c8
    return metrics


def metadata_from_row(row: dict, sample_id: str, dataset_name: str) -> dict[str, Any]:
    keys = [
        "stage",
        "split",
        "task_kind",
        "concept",
        "category",
        "prompt",
        "answer",
        "canonical_answer",
        "point_labels",
        "point_count",
        "triple_labels",
        "image_sha256",
        "problem_id",
        "source_split",
        "values_in_diagram",
        "image_text_labels",
<<<<<<< HEAD
        "ignored_symbol_count",
        "unseen_symbols",
        "max_collinear",
=======
>>>>>>> 499470cfa0cb6010a9ddbc450ed1509fba3563c8
    ]
    out = {
        "id": sample_id,
        "dataset": dataset_name,
        "task_kind": json_safe(row.get("task_kind")),
        "gold_concept": json_safe(row.get("concept")),
        "gold_answer": json_safe(row.get("answer")),
        "gold_canonical_answer": json_safe(row.get("canonical_answer")),
        "gold_point_labels": json_safe(row.get("point_labels")),
        "gold_point_count": json_safe(row.get("point_count")),
        "gold_triple_labels": json_safe(row.get("triple_labels")),
        "image_sha256": json_safe(row.get("image_sha256")),
        "prompt": json_safe(row.get("prompt")),
    }
    for key in keys:
        if key not in out and key in row:
            out[key] = json_safe(row.get(key))
    return out


def run_dataset_inference(
    args: argparse.Namespace,
    dataset_name: str,
    tokenizer,
    model,
    image_processor,
) -> tuple[list[dict], Path]:
    rows = load_parquet_records(args.data_dir, dataset_name)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

<<<<<<< HEAD
    model_dir = args.output_dir / args.model_kind / args.image_mode
    model_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_dir / f"{dataset_name}.jsonl"

    shuffled_images = build_shuffled_image_map(rows, args.seed) if args.image_mode == "shuffled" else {}

=======
    model_dir = args.output_dir / args.model_kind
    model_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_dir / f"{dataset_name}.jsonl"

>>>>>>> 499470cfa0cb6010a9ddbc450ed1509fba3563c8
    existing = read_jsonl(output_path) if args.resume else []
    completed_ids = {str(x["id"]) for x in existing}
    mode = "a" if args.resume else "w"

    with output_path.open(mode, encoding="utf-8") as handle:
<<<<<<< HEAD
        for idx, row in enumerate(tqdm(rows, desc=f"{args.model_kind}:{args.image_mode}:{dataset_name}")):
=======
        for idx, row in enumerate(tqdm(rows, desc=f"{args.model_kind}:{dataset_name}")):
>>>>>>> 499470cfa0cb6010a9ddbc450ed1509fba3563c8
            sample_id = get_sample_id(row, idx)
            if sample_id in completed_ids:
                continue

            task_kind = str(row.get("task_kind") or "")
            is_anchor = task_kind == "anchor" or dataset_name.startswith("stage3_")
            max_new_tokens = args.max_new_tokens_anchor if is_anchor else args.max_new_tokens_local

<<<<<<< HEAD
            source_image_key = str(row.get("image_sha256") or f"__row_{idx}")
            image_override = None
            shuffled_image_key = None
            if args.image_mode == "shuffled":
                image_override, shuffled_image_key = shuffled_images[source_image_key]

=======
>>>>>>> 499470cfa0cb6010a9ddbc450ed1509fba3563c8
            try:
                prediction = generate_one(
                    row,
                    tokenizer=tokenizer,
                    model=model,
                    image_processor=image_processor,
                    conv_mode=args.conv_mode,
                    max_new_tokens=max_new_tokens,
<<<<<<< HEAD
                    image_mode=args.image_mode,
                    image_override=image_override,
=======
>>>>>>> 499470cfa0cb6010a9ddbc450ed1509fba3563c8
                )
                error = None
            except Exception as exc:
                prediction = ""
                error = f"{type(exc).__name__}: {exc}"

            item = metadata_from_row(row, sample_id, dataset_name)
            item.update(
                {
                    "model_kind": args.model_kind,
<<<<<<< HEAD
                    "image_mode": args.image_mode,
                    "shuffled_image_sha256": shuffled_image_key,
=======
>>>>>>> 499470cfa0cb6010a9ddbc450ed1509fba3563c8
                    "prediction": prediction,
                    "error": error,
                }
            )
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()

    results = read_jsonl(output_path)
    if args.max_samples > 0:
        allowed = {get_sample_id(row, i) for i, row in enumerate(rows)}
        results = [r for r in results if str(r.get("id")) in allowed]
    return results, output_path


def write_metrics(args: argparse.Namespace, dataset_name: str, metrics: dict[str, Any]) -> Path:
<<<<<<< HEAD
    metrics_dir = args.output_dir / args.model_kind / args.image_mode / "metrics"
=======
    metrics_dir = args.output_dir / args.model_kind / "metrics"
>>>>>>> 499470cfa0cb6010a9ddbc450ed1509fba3563c8
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / f"{dataset_name}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return path


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model, image_processor, context_len = load_eval_model(args)
    print(f"Context length reported by builder: {context_len}")

    for dataset_name in args.datasets:
        results, prediction_path = run_dataset_inference(
            args,
            dataset_name,
            tokenizer,
            model,
            image_processor,
        )
        metrics = score_dataset(results, dataset_name, args.pair_signature_threshold)
        metrics.update(
            {
                "model_kind": args.model_kind,
<<<<<<< HEAD
                "image_mode": args.image_mode,
=======
>>>>>>> 499470cfa0cb6010a9ddbc450ed1509fba3563c8
                "prediction_file": str(prediction_path),
                "generation": {
                    "greedy": True,
                    "temperature": 0.0,
                    "num_beams": 1,
                    "conv_mode": args.conv_mode,
                },
            }
        )
        metrics_path = write_metrics(args, dataset_name, metrics)
        print(f"Metrics saved to {metrics_path}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
