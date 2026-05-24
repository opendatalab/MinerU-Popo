#!/usr/bin/env python3
"""Evaluate predictions from normalized OCR/layout outputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import sys
from statistics import mean
from typing import Any

import zss

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_processing.label_normalization import (  # noqa: E402
    NormalizedBlock,
    build_reader_from_input_dir,
    normalize_text,
    title_blocks,
)


TITLE_PREFIX = "<image>\nTitle Level Analysis: "
TITLE_TEDS_MODE = "content_aware"
DEFAULT_GT_JSON_CANDIDATES = [
    Path("eval_gt_dir/title.json"),
]


@dataclass
class GTTitleBlock:
    block_id: str
    page: int
    bbox: list[float]
    content: str
    order: int


def normalize_prediction(pred: Any) -> str:
    if isinstance(pred, list):
        return "\n".join(part for part in pred if isinstance(part, str) and part.strip())
    if pred is None:
        return ""
    return pred if isinstance(pred, str) else str(pred)


def compact_text(text: Any) -> str:
    text = normalize_text(text)
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", text).lower()


def text_similarity(left: Any, right: Any) -> float:
    left_text = compact_text(left)
    right_text = compact_text(right)
    if not left_text and not right_text:
        return 1.0
    if not left_text or not right_text:
        return 0.0
    score = SequenceMatcher(None, left_text, right_text).ratio()
    if left_text in right_text or right_text in left_text:
        score = max(score, 0.9)
    return score


def doc_id_from_path(value: str | Path) -> str:
    doc_id = Path(value).stem
    if not doc_id:
        raise ValueError(f"Cannot resolve doc id from: {value}")
    return doc_id


def parse_title_prompt(prompt: str) -> list[GTTitleBlock]:
    content = prompt or ""
    if content.startswith(TITLE_PREFIX):
        content = content[len(TITLE_PREFIX) :]
    parts = ("\n" + content).split("\n<|id|>")
    if parts and parts[0] == "":
        parts = parts[1:]

    blocks: list[GTTitleBlock] = []
    for order, part in enumerate(parts):
        if not part.strip():
            continue
        match = re.match(
            r"(?P<block_id>[^<]+)<\|page\|>(?P<page>\d+)"
            r"<\|box\|>(?P<box>[^<]+)<\|content\|>(?P<content>.*)",
            part,
            flags=re.DOTALL,
        )
        if not match:
            continue
        raw_box = [float(value) for value in match.group("box").split()[:4]]
        if len(raw_box) != 4:
            bbox = [0.0, 0.0, 0.0, 0.0]
        else:
            # Benchmark prompts serialize boxes as top left bottom right.
            top, left, bottom, right = raw_box
            bbox = [left, top, right, bottom]
        blocks.append(
            GTTitleBlock(
                block_id=match.group("block_id").strip(),
                page=int(match.group("page")),
                bbox=bbox,
                content=normalize_text(match.group("content")),
                order=order,
            )
        )
    return blocks


def parse_title_labels(raw: str) -> list[tuple[str, int]]:
    nodes: list[tuple[str, int]] = []
    for line in normalize_prediction(raw).splitlines():
        match = re.match(r"<\|id\|>([^<]+)<\|level\|>(-?\d+)", line.strip())
        if not match:
            continue
        level = int(match.group(2))
        if level >= 0:
            nodes.append((match.group(1).strip(), level))
    return nodes


def gt_nodes_for_mode(
    raw_nodes: list[tuple[str, int]],
    block_by_id: dict[str, GTTitleBlock],
    mode: str,
) -> list[tuple[str, int]]:
    if mode == "id_aware":
        return raw_nodes
    if mode == "structure_only":
        return [("NODE", level) for _, level in raw_nodes]
    if mode == "content_aware":
        return [(compact_text(block_by_id.get(block_id).content if block_id in block_by_id else ""), level) for block_id, level in raw_nodes]
    raise ValueError(f"Unsupported title TEDS mode: {mode}")


def pred_nodes_for_mode(
    raw_nodes: list[tuple[str, int]],
    block_by_id: dict[str, GTTitleBlock],
    mode: str,
) -> list[tuple[str, int]]:
    return gt_nodes_for_mode(raw_nodes, block_by_id, mode)


def bbox_intersection(left_box: list[float], right_box: list[float]) -> float:
    left = max(left_box[0], right_box[0])
    top = max(left_box[1], right_box[1])
    right = min(left_box[2], right_box[2])
    bottom = min(left_box[3], right_box[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def bbox_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox_iou(left_box: list[float], right_box: list[float]) -> float:
    inter = bbox_intersection(left_box, right_box)
    union = bbox_area(left_box) + bbox_area(right_box) - inter
    return inter / union if union > 0 else 0.0


def bbox_overlap_smaller(left_box: list[float], right_box: list[float]) -> float:
    inter = bbox_intersection(left_box, right_box)
    denom = min(bbox_area(left_box), bbox_area(right_box))
    return inter / denom if denom > 0 else 0.0


def alignment_score(gt_block: GTTitleBlock, model_block: NormalizedBlock) -> float:
    if gt_block.page != model_block.page:
        return 0.0
    box_score = max(bbox_iou(gt_block.bbox, model_block.bbox), bbox_overlap_smaller(gt_block.bbox, model_block.bbox))
    text_score = text_similarity(gt_block.content, model_block.content)
    type_score = 1.0 if model_block.type == "title" else 0.4
    return 0.45 * box_score + 0.35 * text_score + 0.20 * type_score


def align_title_blocks_to_gt(
    model_blocks: list[NormalizedBlock],
    gt_blocks: list[GTTitleBlock],
    min_alignment_score: float = 0.35,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    candidates: list[tuple[float, str, str]] = []
    model_by_id = {block.block_id: block for block in model_blocks}
    gt_by_id = {block.block_id: block for block in gt_blocks}
    for gt_block in gt_blocks:
        for model_block in model_blocks:
            score = alignment_score(gt_block, model_block)
            if score >= min_alignment_score:
                candidates.append((score, gt_block.block_id, model_block.block_id))

    candidates.sort(reverse=True)
    model_to_gt: dict[str, str] = {}
    used_gt: set[str] = set()
    matches: list[dict[str, Any]] = []
    for score, gt_id, model_id in candidates:
        if gt_id in used_gt or model_id in model_to_gt:
            continue
        gt_block = gt_by_id[gt_id]
        model_block = model_by_id[model_id]
        text_score = text_similarity(gt_block.content, model_block.content)
        box_score = max(bbox_iou(gt_block.bbox, model_block.bbox), bbox_overlap_smaller(gt_block.bbox, model_block.bbox))
        if text_score < 0.45 and box_score < 0.8:
            continue
        used_gt.add(gt_id)
        model_to_gt[model_id] = gt_id
        matches.append(
            {
                "gt_id": gt_id,
                "gt_content": gt_block.content,
                "pred_id": model_id,
                "pred_content": model_block.content,
                "pred_level": model_block.title_level,
                "score": round(score, 4),
            }
        )
    return model_to_gt, matches


def build_prediction_nodes(
    predicted_blocks: list[NormalizedBlock],
    gt_blocks: list[GTTitleBlock],
    min_match_score: float,
) -> tuple[list[tuple[str, int]], list[dict[str, Any]]]:
    model_to_gt, matches = align_title_blocks_to_gt(
        predicted_blocks,
        gt_blocks,
        min_alignment_score=min_match_score,
    )
    nodes: list[tuple[str, int]] = []
    seen_gt: set[str] = set()
    for pred in predicted_blocks:
        gt_id = model_to_gt.get(pred.block_id)
        if not gt_id or gt_id in seen_gt:
            continue
        seen_gt.add(gt_id)
        level = pred.title_level if pred.title_level and pred.title_level > 0 else 1
        nodes.append((gt_id, int(level)))
    return nodes, matches


def build_tree(nodes: list[tuple[str, int]]) -> zss.Node:
    root = zss.Node("ROOT")
    stack: list[tuple[zss.Node, int]] = [(root, -1)]
    if not nodes:
        return root
    min_level = min(level for _, level in nodes)
    for label, raw_level in nodes:
        level = int(raw_level) - min_level
        node = zss.Node(str(label))
        while stack and stack[-1][1] >= level:
            stack.pop()
        parent = stack[-1][0] if stack else root
        parent.addkid(node)
        stack.append((node, level))
    return root


def count_tree_nodes(root: zss.Node) -> int:
    def dfs(node: zss.Node) -> int:
        return 1 + sum(dfs(child) for child in node.children)

    return sum(dfs(child) for child in root.children)


def title_teds_score(reference_nodes: list[tuple[str, int]], predicted_nodes: list[tuple[str, int]]) -> float:
    reference_tree = build_tree(reference_nodes)
    predicted_tree = build_tree(predicted_nodes)
    distance = zss.distance(
        reference_tree,
        predicted_tree,
        lambda node: node.children,
        lambda node: 1,
        lambda node: 1,
        lambda left, right: 0 if left.label == right.label else 1,
    )
    denom = max(count_tree_nodes(reference_tree), count_tree_nodes(predicted_tree)) or 1
    return max(0.0, min(1.0, 1.0 - distance / denom))


def evaluate_one_item(
    item: dict[str, Any],
    model_name: str,
    reader,
    mode: str,
    min_match_score: float,
) -> dict[str, Any]:
    doc_id = doc_id_from_path(str(item.get("image", "")))
    gt_blocks = parse_title_prompt(item["conversations"][0]["value"])
    gt_block_by_id = {block.block_id: block for block in gt_blocks}
    gt_nodes_raw = parse_title_labels(item["conversations"][1]["value"])

    reader_result = reader.read_doc(doc_id)
    if reader_result.status != "ok":
        return {
            "doc_id": doc_id,
            "model_name": model_name,
            "mode": mode,
            "status": reader_result.status,
            "message": reader_result.message,
            "score": 0.0,
            "gt_count": len(gt_nodes_raw),
            "pred_count": 0,
            "match_count": 0,
            "candidate_count": len(gt_blocks),
        }

    predicted_title_blocks = title_blocks(reader_result.blocks)
    pred_nodes_raw, matches = build_prediction_nodes(
        predicted_title_blocks,
        gt_blocks,
        min_match_score=min_match_score,
    )
    gt_nodes = gt_nodes_for_mode(gt_nodes_raw, gt_block_by_id, mode)
    pred_nodes = pred_nodes_for_mode(pred_nodes_raw, gt_block_by_id, mode)
    score = title_teds_score(gt_nodes, pred_nodes)
    return {
        "doc_id": doc_id,
        "model_name": model_name,
        "mode": mode,
        "status": "ok",
        "score": round(score, 6),
        "gt_count": len(gt_nodes_raw),
        "pred_count": len(pred_nodes_raw),
        "match_count": len(matches),
        "candidate_count": len(gt_blocks),
        "raw_model_title_count": len(predicted_title_blocks),
        "prediction_nodes": [{"id": node_id, "level": level} for node_id, level in pred_nodes_raw],
        "gt_nodes": [{"id": node_id, "level": level} for node_id, level in gt_nodes_raw],
        "matches": matches,
    }


def summarize_details(details: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [item for item in details if item["status"] == "ok"]
    return {
        "total_docs": len(details),
        "usable_docs": len(usable),
        "missing_docs": len(details) - len(usable),
        "avg_score_usable": mean(item["score"] for item in usable) if usable else 0.0,
        "avg_score_with_missing_zero": mean(item["score"] if item["status"] == "ok" else 0.0 for item in details) if details else 0.0,
        "avg_gt_count": mean(item["gt_count"] for item in usable) if usable else 0.0,
        "avg_pred_count": mean(item["pred_count"] for item in usable) if usable else 0.0,
        "avg_match_count": mean(item["match_count"] for item in usable) if usable else 0.0,
    }


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_gt_json(explicit_path: str = "") -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Ground-truth JSON does not exist: {path}")
        return path
    for path in DEFAULT_GT_JSON_CANDIDATES:
        if path.exists():
            return path
    candidates = ", ".join(str(path) for path in DEFAULT_GT_JSON_CANDIDATES)
    raise FileNotFoundError(f"Cannot find ground-truth title JSON. Tried: {candidates}. Pass --gt-json explicitly.")


def default_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return Path(args.input_dir) / "title_level_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate title-level outputs for OCR/layout models.")
    parser.add_argument("--model", required=True, help="Model name, e.g. mineru or PaddleOCR-VL-1.5.")
    parser.add_argument("--input-dir", required=True, help="Directory containing this model's outputs.")
    parser.add_argument("--output-dir", default="", help="Output directory. Defaults to <input-dir>/title_level_eval.")
    parser.add_argument("--gt-json", default="", help="Optional ground-truth title JSON. Auto-detected in repo-relative eval paths.")
    parser.add_argument("--bbox-scale", choices=["source", "relative", "thousand"], default="source")
    parser.add_argument("--doc-limit", type=int, default=0)
    parser.add_argument("--min-match-score", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_json = resolve_gt_json(args.gt_json)
    title_items = json.loads(gt_json.read_text(encoding="utf-8"))
    if args.doc_limit:
        title_items = title_items[: args.doc_limit]

    mode = TITLE_TEDS_MODE
    model_name = args.model
    output_dir = default_output_dir(args)

    reader = build_reader_from_input_dir(model_name, args.input_dir, bbox_scale=args.bbox_scale)
    details = [
        evaluate_one_item(item, model_name, reader, mode, args.min_match_score)
        for item in title_items
    ]
    summary = summarize_details(details)
    summary["model_name"] = model_name
    summary["mode"] = mode

    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "details.json", details)
    save_json(output_dir / "aggregate_summary.json", {model_name: summary})
    print(
        f"[title-eval] mode={mode} model={model_name} "
        f"usable={summary['usable_docs']}/{summary['total_docs']} "
        f"score={summary['avg_score_with_missing_zero']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
