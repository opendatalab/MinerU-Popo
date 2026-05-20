#!/usr/bin/env python3
"""Run Popo post-processing inference from normalized OCR labels.

Input contract:
  - reads outputs from label_normalization.py in post-processing format
  - block bbox must be xyxy_01: [x1, y1, x2, y2], normalized to 0..1
  - inference.py is responsible for prompt-time conversion to yxyx_1000

Default behavior runs up to 10 documents per model and writes:
  outputs/inference/<model>/<doc>.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

from inference import main as run_one_document


DEFAULT_MODELS = [
    "glm-ocr",
    "mineru",
    "monkeyocr",
    "dolphin",
    "PaddleOCR-VL-1.5",
]

ALLOWED_BLOCK_TYPES = {
    "title",
    "text",
    "list_item",
    "equation",
    "image",
    "table",
    "image_caption",
    "table_caption",
    "image_footnote",
    "table_footnote",
}


def safe_model_name(model: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", model)


def parse_models(value: str) -> list[str]:
    models = [item.strip() for item in value.split(",") if item.strip()]
    return models or list(DEFAULT_MODELS)


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_filename(input_label: str) -> str:
    path = Path(input_label)
    stem = path.stem if path.suffix else path.name
    return f"{stem}.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_bbox(model: str, doc_key: str, page: str, block_index: int, bbox: Any, strict: bool) -> None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"{model} {doc_key} page={page} block={block_index}: bbox must be a 4-item list")
    values = []
    for value in bbox:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{model} {doc_key} page={page} block={block_index}: non-numeric bbox={bbox}") from exc
        if not math.isfinite(number):
            raise ValueError(f"{model} {doc_key} page={page} block={block_index}: non-finite bbox={bbox}")
        values.append(number)
    x1, y1, x2, y2 = values
    if x2 < x1 or y2 < y1:
        raise ValueError(f"{model} {doc_key} page={page} block={block_index}: invalid xyxy order bbox={bbox}")
    if strict and (min(values) < -1e-6 or max(values) > 1.000001):
        raise ValueError(
            f"{model} {doc_key} page={page} block={block_index}: bbox is not xyxy_01 bbox={bbox}; "
            "rerun label_normalization.py first, or pass --allow-out-of-range-bbox only for debugging"
        )


def validate_pages(model: str, doc_key: str, pages: Any, strict_bbox: bool) -> int:
    if not isinstance(pages, dict):
        raise ValueError(f"{model} {doc_key}: expected pages dict, got {type(pages).__name__}")
    block_count = 0
    for page, blocks in pages.items():
        if not isinstance(blocks, list):
            raise ValueError(f"{model} {doc_key} page={page}: expected block list")
        for block_index, block in enumerate(blocks):
            if not isinstance(block, dict):
                raise ValueError(f"{model} {doc_key} page={page} block={block_index}: expected block dict")
            block_type = str(block.get("type", ""))
            if block_type not in ALLOWED_BLOCK_TYPES:
                raise ValueError(
                    f"{model} {doc_key} page={page} block={block_index}: unexpected type={block_type!r}"
                )
            validate_bbox(model, doc_key, str(page), block_index, block.get("bbox"), strict_bbox)
            block_count += 1
    return block_count


def load_normalized_items(model: str, input_dir: Path) -> tuple[str, list[tuple[str, Any]]]:
    safe_model = safe_model_name(model)
    model_dir = input_dir / safe_model
    if model_dir.is_dir():
        items: list[tuple[str, Any]] = []
        for path in sorted(model_dir.glob("*.json")):
            payload = load_json(path)
            if not isinstance(payload, dict):
                raise ValueError(f"Per-doc normalized input must be a JSON object: {path}")
            if "pages" in payload:
                input_label = str(payload.get("input_label") or payload.get("doc_id") or path.stem)
                pages = payload["pages"]
            else:
                input_label = str(payload.get("input_label") or path.stem)
                pages = payload
            items.append((input_label, pages))
        return str(model_dir), items

    input_path = input_dir / f"{safe_model}.post_processing.json"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing normalized input for {model}: expected directory {model_dir} "
            f"or legacy file {input_path}"
        )
    data = load_json(input_path)
    if not isinstance(data, dict):
        raise ValueError(f"Normalized input must be a JSON object: {input_path}")
    return str(input_path), list(data.items())


def select_items(items: list[tuple[str, Any]], limit: int) -> list[tuple[str, Any]]:
    return items if limit <= 0 else items[:limit]


def run_model(
    model: str,
    input_dir: Path,
    output_root: Path,
    raw_output_root: Path | None,
    limit: int,
    strict_bbox: bool,
    resume: bool,
    dry_run: bool,
) -> dict[str, Any]:
    input_source, items = load_normalized_items(model, input_dir)

    output_dir = output_root / safe_model_name(model)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir = raw_output_root / safe_model_name(model) if raw_output_root else None
    if raw_output_dir is not None:
        raw_output_dir.mkdir(parents=True, exist_ok=True)

    selected = select_items(items, limit)
    summary = {
        "model": model,
        "input": input_source,
        "output_dir": str(output_dir),
        "raw_output_dir": str(raw_output_dir) if raw_output_dir else None,
        "selected_docs": len(selected),
        "written": 0,
        "skipped_existing": 0,
        "validated_blocks": 0,
        "dry_run": dry_run,
    }

    for doc_key, pages in tqdm(selected, desc=model):
        block_count = validate_pages(model, doc_key, pages, strict_bbox=strict_bbox)
        summary["validated_blocks"] += block_count

        output_path = output_dir / output_filename(doc_key)
        if resume and output_path.exists():
            summary["skipped_existing"] += 1
            continue
        if dry_run:
            continue

        run_one_document(
            doc_key,
            copy.deepcopy(pages),
            str(output_dir),
            raw_output_dir=str(raw_output_dir) if raw_output_dir else None,
        )
        summary["written"] += 1

    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    repo_root = default_repo_root()
    parser = argparse.ArgumentParser(description="Run Popo inference for normalized OCR outputs.")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model list. Default runs glm-ocr, mineru, monkeyocr, dolphin, PaddleOCR-VL-1.5.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(repo_root / "outputs" / "label_normalization"),
        help="Directory containing per-doc <model>/<doc_id>.json files, or legacy <model>.post_processing.json files.",
    )
    parser.add_argument(
        "--output-root",
        default=str(repo_root / "outputs" / "inference"),
        help="Root directory for inference outputs.",
    )
    parser.add_argument(
        "--raw-output-root",
        default=str(repo_root / "outputs" / "inference_raw"),
        help="Root directory for raw prompts/responses. Pass an empty string to disable.",
    )
    parser.add_argument("--limit", type=int, default=10, help="Documents per model. Use <=0 for all.")
    parser.add_argument("--resume", action="store_true", help="Skip documents whose output JSON already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Validate selected inputs without model generation.")
    parser.add_argument(
        "--allow-out-of-range-bbox",
        action="store_true",
        help="Disable strict xyxy_01 bbox range validation. Only use for debugging old normalized files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    raw_output_root = Path(args.raw_output_root) if args.raw_output_root else None
    strict_bbox = not args.allow_out_of_range_bbox

    summaries = []
    for model in parse_models(args.models):
        summaries.append(
            run_model(
                model=model,
                input_dir=input_dir,
                output_root=output_root,
                raw_output_root=raw_output_root,
                limit=args.limit,
                strict_bbox=strict_bbox,
                resume=args.resume,
                dry_run=args.dry_run,
            )
        )

    print(json.dumps({"models": summaries}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
