#!/usr/bin/env python3
"""Normalize major OCR/layout outputs into a Popo-compatible block schema.

The adapter keeps one canonical label space for cross-model comparison:
``title``, ``text``, ``image``, ``table``, and ``caption``.  Each block also
keeps a ``popo_type`` so the same output can be converted into the input shape
expected by ``post_processing/inference.py`` and downstream tree construction.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Any, Iterable


DEFAULT_MODELS = [
    "mineru",
    "monkeyocr",
    "PaddleOCR-VL-1.5",
    "dolphin",
    "glm-ocr",
]
DEFAULT_PDF_DIR_CANDIDATES = [
    Path("eval_pdf_dir"),
]

CANONICAL_TYPES = {"title", "text", "image", "table", "caption"}
DOC_ID_RE = re.compile(r"(doc-[A-Za-z0-9\-]+)")
HEADING_RE = re.compile(r"^(#{1,8})\s+(.*\S)\s*$")
SKIP_TYPE = "__skip__"


@dataclass
class NormalizedBlock:
    block_id: str
    page: int
    bbox: list[float]
    type: str
    content: str
    order: int
    popo_type: str
    title_level: int | None = None
    source_label: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}

    def to_popo_block(self) -> dict[str, Any]:
        block = {
            "type": self.popo_type,
            "content": self.content,
            "bbox": self.bbox,
        }
        if self.title_level is not None:
            block["title_level"] = self.title_level
        if self.source_label is not None:
            block["source_label"] = self.source_label
        block["source_id"] = self.block_id
        return block


@dataclass
class ReaderResult:
    model_name: str
    doc_id: str
    status: str
    blocks: list[NormalizedBlock] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "doc_id": self.doc_id,
            "status": self.status,
            "message": self.message,
            "blocks": [block.to_dict() for block in self.blocks],
        }


class BaseReader:
    model_name: str

    def __init__(self, model_root: str | Path, bbox_scale: str = "source") -> None:
        self.model_root = Path(model_root)
        self.bbox_scale = bbox_scale

    def read_doc(self, doc_id: str) -> ReaderResult:
        raise NotImplementedError

    def make_block(
        self,
        doc_id: str,
        order: int,
        page: int,
        bbox: Iterable[float],
        canonical_type: str,
        content: str,
        popo_type: str | None = None,
        title_level: int | None = None,
        source_label: str | None = None,
        page_width: float | int | None = None,
        page_height: float | int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> NormalizedBlock:
        if canonical_type not in CANONICAL_TYPES:
            canonical_type = "text"
        return NormalizedBlock(
            block_id=f"{doc_id}:{order}",
            page=int(page),
            bbox=convert_bbox(bbox, page_width, page_height, self.bbox_scale),
            type=canonical_type,
            content=normalize_text(content),
            order=int(order),
            popo_type=popo_type or canonical_type,
            title_level=title_level,
            source_label=source_label,
            meta=meta or {},
        )


def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\u3000", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


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
        score = max(score, 0.95)
    return score


def safe_json_load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def doc_id_from_text(value: str) -> str:
    match = DOC_ID_RE.search(value)
    if not match:
        raise ValueError(f"Cannot extract doc id from: {value}")
    return match.group(1)


def convert_bbox(
    bbox: Iterable[float],
    page_width: float | int | None = None,
    page_height: float | int | None = None,
    bbox_scale: str = "source",
) -> list[float]:
    values = [float(value) for value in list(bbox)[:4]]
    if len(values) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    left, top, right, bottom = values

    if bbox_scale == "source":
        return [left, top, right, bottom]

    width = float(page_width or 0)
    height = float(page_height or 0)
    max_value = max(abs(left), abs(top), abs(right), abs(bottom))

    if bbox_scale == "relative":
        if width > 0 and height > 0 and max_value > 1.5:
            return [left / width, top / height, right / width, bottom / height]
        return [left, top, right, bottom]

    if bbox_scale == "thousand":
        if max_value <= 1.5:
            return [left * 1000.0, top * 1000.0, right * 1000.0, bottom * 1000.0]
        if width > 0 and height > 0:
            return [
                left * 1000.0 / width,
                top * 1000.0 / height,
                right * 1000.0 / width,
                bottom * 1000.0 / height,
            ]
        return [left, top, right, bottom]

    raise ValueError(f"Unsupported bbox_scale: {bbox_scale}")


def normalize_bbox_to_unit(
    bbox: Iterable[float],
    page_width: float | int | None = None,
    page_height: float | int | None = None,
    assumed_scale: float | int | None = None,
) -> list[float]:
    """Return xyxy bbox normalized to 0..1 when size information is available."""
    values = [float(value) for value in list(bbox)[:4]]
    if len(values) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    def clamp_unit(items: list[float]) -> list[float]:
        return [min(1.0, max(0.0, value)) for value in items]

    max_value = max(abs(value) for value in values)
    if max_value <= 1.5:
        return clamp_unit(values)
    if assumed_scale:
        scale = float(assumed_scale)
        if scale > 0:
            return clamp_unit([value / scale for value in values])
    width = float(page_width or 0)
    height = float(page_height or 0)
    if width > 0 and height > 0:
        return clamp_unit([
            values[0] / width,
            values[1] / height,
            values[2] / width,
            values[3] / height,
        ])
    return values


def parse_numbered_title_level(text: Any) -> int | None:
    text = normalize_text(text)
    if not text:
        return None
    if re.match(r"^第[一二三四五六七八九十百千万\d]+章\b", text):
        return 1
    if re.match(r"^第[一二三四五六七八九十百千万\d]+节\b", text):
        return 2
    if re.match(r"^第[一二三四五六七八九十百千万\d]+条\b", text):
        return 3
    match = re.match(r"^(\d+(?:\.\d+){0,7})\b", text)
    if match:
        return len(match.group(1).split("."))
    lowered = text.lower()
    if re.match(r"^(chapter|part)\b", lowered):
        return 1
    if re.match(r"^(section|appendix)\b", lowered):
        return 2
    return None


def looks_like_title_text(text: Any) -> bool:
    text = normalize_text(text)
    if not text or len(text) > 120:
        return False
    if parse_numbered_title_level(text) is not None:
        return True
    if len(text) <= 50 and text.upper() == text and re.search(r"[A-Z\u4e00-\u9fff]", text):
        return True
    keywords = (
        "abstract",
        "summary",
        "introduction",
        "references",
        "appendix",
        "目录",
        "摘要",
        "引言",
        "参考文献",
        "附录",
    )
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def infer_missing_title_levels(blocks: list[NormalizedBlock]) -> list[NormalizedBlock]:
    title_blocks = [block for block in blocks if block.type == "title"]
    heights = sorted({round(abs(block.bbox[3] - block.bbox[1]), 3) for block in title_blocks}, reverse=True)
    height_rank = {height: rank + 1 for rank, height in enumerate(heights)}
    for block in title_blocks:
        if block.title_level is not None and block.title_level > 0:
            continue
        parsed = parse_numbered_title_level(block.content)
        if parsed is not None:
            block.title_level = parsed
            continue
        height = round(abs(block.bbox[3] - block.bbox[1]), 3)
        rank = height_rank.get(height)
        if rank is not None:
            block.title_level = min(max(rank, 1), 8)
    return blocks


def sort_blocks(blocks: Iterable[NormalizedBlock]) -> list[NormalizedBlock]:
    return sorted(blocks, key=lambda block: (block.page, block.order, block.bbox[1], block.bbox[0]))


def reassign_block_ids(blocks: Iterable[NormalizedBlock], doc_id: str) -> list[NormalizedBlock]:
    reassigned = []
    for order, block in enumerate(sort_blocks(blocks)):
        block.block_id = f"{doc_id}:{order}"
        block.order = order
        reassigned.append(block)
    return reassigned


def load_markdown_heading_pool(paths: Iterable[Path]) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = HEADING_RE.match(line.strip())
            if match:
                pool.append(
                    {
                        "level": len(match.group(1)),
                        "text": normalize_text(match.group(2)),
                        "used": False,
                    }
                )
    return pool


def consume_heading_level(pool: list[dict[str, Any]], text: str, min_score: float = 0.7) -> int | None:
    best_index = None
    best_score = 0.0
    for index, item in enumerate(pool):
        if item["used"]:
            continue
        score = text_similarity(text, item["text"])
        if score > best_score:
            best_index = index
            best_score = score
    if best_index is None or best_score < min_score:
        return None
    pool[best_index]["used"] = True
    return int(pool[best_index]["level"])


def optional_int_sort_key(item: dict[str, Any], key: str) -> tuple[int, int]:
    value = item.get(key)
    if value is None:
        return (1, 0)
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, 0)


def page_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"_(\d+)(?:_res)?$", path.stem)
    if not match:
        return (10**9, path.name)
    return (int(match.group(1)), path.name)


def extract_middle_text(block: dict[str, Any]) -> str:
    line_texts = []
    for line in block.get("lines", []):
        spans = line.get("spans", [])
        text = "".join(str(span.get("content", "")) for span in spans)
        text = normalize_text(text)
        if text:
            line_texts.append(text)
    return " ".join(line_texts)


def extract_block_content(block: dict[str, Any]) -> str:
    for key in ("content", "text", "html", "words"):
        if block.get(key):
            return normalize_text(block.get(key))
    return extract_middle_text(block)


def iter_model_pages(data: Any) -> Iterable[tuple[int, list[dict[str, Any]]]]:
    if isinstance(data, list):
        for index, page in enumerate(data, start=1):
            if isinstance(page, list):
                yield index, [item for item in page if isinstance(item, dict)]
            elif isinstance(page, dict):
                items = page.get("blocks") or page.get("items") or page.get("elements")
                if isinstance(items, list):
                    if page.get("page") is not None:
                        page_index = int(page["page"])
                    elif page.get("page_idx") is not None:
                        page_index = int(page["page_idx"]) + 1
                    else:
                        page_index = index
                    yield page_index, [item for item in items if isinstance(item, dict)]
                else:
                    yield index, [page]
        return
    if isinstance(data, dict):
        pages = data.get("pages") or data.get("pdf_info")
        if isinstance(pages, list):
            yield from iter_model_pages(pages)


def lookup_content_level(content_items: list[dict[str, Any]], page: int, text: str, bbox: Iterable[float]) -> int | None:
    best_level = None
    best_score = 0.0
    target_bbox = [float(value) for value in list(bbox)[:4]]
    for item in content_items:
        if item.get("text_level") is None:
            continue
        try:
            item_page = int(item.get("page_idx", -1)) + 1
        except (TypeError, ValueError):
            continue
        if item_page != int(page):
            continue
        score = text_similarity(text, item.get("text", ""))
        item_bbox = item.get("bbox")
        if item_bbox is not None:
            score = 0.4 * score + 0.6 * bbox_overlap_smaller(target_bbox, [float(v) for v in item_bbox[:4]])
        if score > best_score:
            best_score = score
            best_level = int(item["text_level"])
    return best_level if best_score >= 0.5 else None


def bbox_overlap_smaller(left_box: list[float], right_box: list[float]) -> float:
    left = max(left_box[0], right_box[0])
    top = max(left_box[1], right_box[1])
    right = min(left_box[2], right_box[2])
    bottom = min(left_box[3], right_box[3])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    left_area = max(0.0, left_box[2] - left_box[0]) * max(0.0, left_box[3] - left_box[1])
    right_area = max(0.0, right_box[2] - right_box[0]) * max(0.0, right_box[3] - right_box[1])
    denom = min(left_area, right_area)
    return inter / denom if denom > 0 else 0.0


class PaddleReader(BaseReader):
    model_name = "PaddleOCR-VL-1.5"
    render_scale = 2.0

    def read_doc(self, doc_id: str) -> ReaderResult:
        doc_root = self.model_root / doc_id
        json_path = doc_root / "layout_parsing.json"
        if json_path.exists():
            return self._read_layout_parsing(doc_id, doc_root, json_path)

        page_paths = sorted(doc_root.glob(f"{doc_id}_*_res.json"), key=page_sort_key)
        if page_paths:
            return self._read_per_page_results(doc_id, doc_root, page_paths)

        return ReaderResult(self.model_name, doc_id, "missing", message=f"Missing {json_path} or per-page *_res.json")

    def _read_layout_parsing(self, doc_id: str, doc_root: Path, json_path: Path) -> ReaderResult:
        heading_pool = load_markdown_heading_pool([doc_root / "layout_parsing.md"])
        data = safe_json_load(json_path)
        blocks: list[NormalizedBlock] = []
        order = 0
        pages = data.get("result", {}).get("layoutParsingResults", [])
        for page_index, page in enumerate(pages, start=1):
            pruned = page.get("prunedResult", {})
            page_width = pruned.get("width")
            page_height = pruned.get("height")
            items = sorted(pruned.get("parsing_res_list", []), key=lambda item: optional_int_sort_key(item, "block_order"))
            for item in items:
                canonical, popo_type = map_paddle_label(str(item.get("block_label", "text")))
                content = normalize_text(item.get("block_content", ""))
                if not content and canonical in {"text", "title", "caption"}:
                    continue
                level = consume_heading_level(heading_pool, content) if canonical == "title" else None
                blocks.append(
                    self.make_block(
                        doc_id,
                        order,
                        page_index,
                        normalize_bbox_to_unit(item.get("block_bbox", [0, 0, 0, 0]), page_width, page_height),
                        canonical,
                        content,
                        popo_type=popo_type,
                        title_level=level,
                        source_label=str(item.get("block_label", "")),
                        page_width=page_width,
                        page_height=page_height,
                    )
                )
                order += 1
        return finalize_reader_result(self.model_name, doc_id, blocks)

    def _read_per_page_results(self, doc_id: str, doc_root: Path, page_paths: list[Path]) -> ReaderResult:
        page_payloads = [safe_json_load(path) for path in page_paths]
        pdf_path = self._resolve_pdf_path(doc_id, page_payloads)
        page_sizes = self._load_pdf_page_sizes(pdf_path)
        if not page_sizes:
            return ReaderResult(
                self.model_name,
                doc_id,
                "missing",
                message=f"Cannot resolve source PDF page sizes for Paddle doc: {doc_id}",
            )

        heading_pool = load_markdown_heading_pool(sorted(doc_root.glob("*.md")))
        blocks: list[NormalizedBlock] = []
        order = 0
        for path, data in zip(page_paths, page_payloads):
            page_index = self._page_index_from_payload_or_path(data, path)
            page_width, page_height = page_sizes.get(page_index, (None, None))
            if not page_width or not page_height:
                return ReaderResult(
                    self.model_name,
                    doc_id,
                    "invalid",
                    message=f"Cannot resolve Paddle page size: doc={doc_id} page={page_index}",
                )
            items = sorted(data.get("parsing_res_list", []), key=lambda item: optional_int_sort_key(item, "block_order"))
            for item in items:
                canonical, popo_type = map_paddle_label(str(item.get("block_label", "text")))
                content = normalize_text(item.get("block_content", ""))
                if not content and canonical in {"text", "title", "caption"}:
                    continue
                level = consume_heading_level(heading_pool, content) if canonical == "title" else None
                blocks.append(
                    self.make_block(
                        doc_id,
                        order,
                        page_index,
                        normalize_bbox_to_unit(item.get("block_bbox", [0, 0, 0, 0]), page_width, page_height),
                        canonical,
                        content,
                        popo_type=popo_type,
                        title_level=level,
                        source_label=str(item.get("block_label", "")),
                        meta={"source": "per_page_res_json", "file": path.name},
                    )
                )
                order += 1
        return finalize_reader_result(self.model_name, doc_id, blocks)

    def _resolve_pdf_path(self, doc_id: str, page_payloads: list[dict[str, Any]]) -> Path | None:
        for payload in page_payloads:
            input_path = payload.get("input_path")
            if input_path:
                path = Path(str(input_path))
                if path.exists():
                    return path
        for candidate_dir in DEFAULT_PDF_DIR_CANDIDATES:
            candidate = candidate_dir / f"{doc_id}.pdf"
            if candidate.exists():
                return candidate
        return None

    def _load_pdf_page_sizes(self, pdf_path: Path | None) -> dict[int, tuple[float, float]]:
        if pdf_path is None or not pdf_path.exists():
            return {}
        try:
            import fitz

            sizes = {}
            doc = fitz.open(str(pdf_path))
            for index, page in enumerate(doc, start=1):
                # Paddle per-page *_res.json bbox is on the 2x-rendered page image.
                sizes[index] = (
                    float(page.rect.width) * self.render_scale,
                    float(page.rect.height) * self.render_scale,
                )
            doc.close()
            return sizes
        except Exception:
            return {}

    def _page_index_from_payload_or_path(self, data: dict[str, Any], path: Path) -> int:
        if data.get("page_index") is not None:
            return int(data["page_index"]) + 1
        match = re.search(r"_(\d+)_res$", path.stem)
        if match:
            return int(match.group(1)) + 1
        return 1

def map_paddle_label(label: str) -> tuple[str, str]:
    text_labels = {
        "text",
        "reference_content",
        "header",
        "footer",
        "abstract",
        "content",
        "number",
        "footnote",
        "aside_text",
        "formula_number",
        "algorithm",
    }
    if label in {"paragraph_title", "doc_title"}:
        return "title", "title"
    if label in {"image", "chart", "footer_image", "header_image", "seal"}:
        return "image", "image"
    if label == "table":
        return "table", "table"
    if label == "figure_title":
        return "caption", "image_caption"
    if label == "vision_footnote":
        return "caption", "image_footnote"
    if label in {"inline_formula", "display_formula"}:
        return "text", "equation"
    if label in text_labels:
        return "text", "text"
    return "text", "text"


class MineruReader(BaseReader):
    model_name = "mineru"
    inner_dir = "vlm"

    def read_doc(self, doc_id: str) -> ReaderResult:
        doc_root = self.model_root / doc_id / self.inner_dir if self.inner_dir else self.model_root / doc_id
        model_path = doc_root / f"{doc_id}_model.json"
        middle_path = doc_root / f"{doc_id}_middle.json"
        content_list_path = doc_root / f"{doc_id}_content_list.json"
        if not model_path.exists() and not middle_path.exists() and not content_list_path.exists():
            return ReaderResult(self.model_name, doc_id, "missing", message=f"Missing {model_path}")

        content_items = []
        if content_list_path.exists():
            loaded = safe_json_load(content_list_path)
            if isinstance(loaded, list):
                content_items = loaded

        if model_path.exists():
            return self._read_model(doc_id, model_path, content_items)
        if middle_path.exists():
            return self._read_middle(doc_id, middle_path, content_items)
        return self._read_content_list(doc_id, content_items)

    def _read_model(self, doc_id: str, model_path: Path, content_items: list[dict[str, Any]]) -> ReaderResult:
        data = safe_json_load(model_path)
        blocks: list[NormalizedBlock] = []
        order = 0
        for page_index, items in iter_model_pages(data):
            for item in items:
                canonical, popo_type = map_mineru_label(str(item.get("type", "text")))
                if popo_type == SKIP_TYPE:
                    continue
                content = extract_block_content(item)
                if not content and canonical in {"text", "title", "caption"}:
                    continue
                level = None
                if canonical == "title":
                    level = lookup_content_level(content_items, page_index, content, item.get("bbox", [0, 0, 0, 0]))
                blocks.append(
                    self.make_block(
                        doc_id,
                        order,
                        page_index,
                        normalize_bbox_to_unit(item.get("bbox", [0, 0, 0, 0])),
                        canonical,
                        content,
                        popo_type=popo_type,
                        title_level=level,
                        source_label=str(item.get("type", "")),
                        meta={"source": "model_json"},
                    )
                )
                order += 1
        return finalize_reader_result(self.model_name, doc_id, blocks)

    def _read_middle(self, doc_id: str, middle_path: Path, content_items: list[dict[str, Any]]) -> ReaderResult:
        data = safe_json_load(middle_path)
        blocks: list[NormalizedBlock] = []
        order = 0
        for page in data.get("pdf_info", []):
            page_index = int(page.get("page_idx", 0)) + 1
            page_size = page.get("page_size") or [None, None]
            page_width = page_size[0] if len(page_size) > 0 else None
            page_height = page_size[1] if len(page_size) > 1 else None
            for item in page.get("para_blocks", []):
                canonical, popo_type = map_mineru_label(str(item.get("type", "text")))
                if popo_type == SKIP_TYPE:
                    continue
                content = extract_block_content(item)
                if not content and canonical in {"text", "title", "caption"}:
                    continue
                level = None
                if canonical == "title":
                    level = lookup_content_level(content_items, page_index, content, item.get("bbox", [0, 0, 0, 0]))
                blocks.append(
                    self.make_block(
                        doc_id,
                        order,
                        page_index,
                        normalize_bbox_to_unit(item.get("bbox", [0, 0, 0, 0]), page_width, page_height),
                        canonical,
                        content,
                        popo_type=popo_type,
                        title_level=level,
                        source_label=str(item.get("type", "")),
                        page_width=page_width,
                        page_height=page_height,
                    )
                )
                order += 1
        return finalize_reader_result(self.model_name, doc_id, blocks)

    def _read_content_list(self, doc_id: str, content_items: list[dict[str, Any]]) -> ReaderResult:
        blocks: list[NormalizedBlock] = []
        order = 0
        for item in content_items:
            if not isinstance(item, dict) or item.get("bbox") is None:
                continue
            canonical, popo_type = map_mineru_label(str(item.get("type", "text")))
            if popo_type == SKIP_TYPE:
                continue
            level = None
            if item.get("text_level") is not None:
                canonical, popo_type = "title", "title"
                level = int(item["text_level"])
            content = extract_block_content(item)
            if not content and canonical in {"text", "title", "caption"}:
                continue
            blocks.append(
                self.make_block(
                    doc_id,
                    order,
                    int(item.get("page_idx", 0)) + 1,
                    normalize_bbox_to_unit(item.get("bbox", [0, 0, 0, 0]), assumed_scale=1000),
                    canonical,
                    content,
                    popo_type=popo_type,
                    title_level=level,
                    source_label=str(item.get("type", "")),
                    meta={"source": "content_list"},
                )
            )
            order += 1
        return finalize_reader_result(self.model_name, doc_id, blocks)


class MonkeyOCRReader(MineruReader):
    model_name = "monkeyocr"
    inner_dir = ""

    def read_doc(self, doc_id: str) -> ReaderResult:
        doc_root = self.model_root / doc_id
        middle_path = doc_root / f"{doc_id}_middle.json"
        content_list_path = doc_root / f"{doc_id}_content_list.json"
        if not middle_path.exists():
            return ReaderResult(self.model_name, doc_id, "missing", message=f"Missing {middle_path}")

        content_items = []
        if content_list_path.exists():
            loaded = safe_json_load(content_list_path)
            if isinstance(loaded, list):
                content_items = loaded
        return self._read_middle(doc_id, middle_path, content_items)


def map_mineru_label(label: str) -> tuple[str, str]:
    if label == "title":
        return "title", "title"
    if label == "image":
        return "image", "image"
    if label == "table":
        return "table", "table"
    if label in {"image_caption", "table_caption", "image_footnote", "table_footnote"}:
        return "caption", label
    if label in {"equation", "interline_equation", "inline_equation"}:
        return "text", "equation"
    if label == "list":
        return "text", "list_item"
    if label in {"discarded", "image_body", "table_body"}:
        return SKIP_TYPE, SKIP_TYPE
    return "text", "text"


class DolphinReader(BaseReader):
    model_name = "dolphin"

    def read_doc(self, doc_id: str) -> ReaderResult:
        json_path = self.model_root / doc_id / "recognition_json" / f"{doc_id}.json"
        if not json_path.exists():
            return ReaderResult(self.model_name, doc_id, "missing", message=f"Missing {json_path}")

        data = safe_json_load(json_path)
        page_sizes = self._load_page_sizes(data.get("source_file"), doc_id)
        if not page_sizes:
            return ReaderResult(
                self.model_name,
                doc_id,
                "missing",
                message=f"Cannot resolve source PDF page sizes for Dolphin doc: {doc_id}",
            )
        blocks: list[NormalizedBlock] = []
        order = 0
        for fallback_page_index, page in enumerate(data.get("pages", []), start=1):
            page_number = page.get("page_number")
            page_index = int(page_number) if page_number is not None and int(page_number) > 0 else fallback_page_index
            page_width, page_height = page_sizes.get(page_index, (None, None))
            items = sorted(page.get("elements", []), key=lambda item: optional_int_sort_key(item, "reading_order"))
            for item in items:
                canonical, popo_type, level = map_dolphin_label(item)
                content = normalize_text(item.get("text", ""))
                if not content and canonical in {"text", "title", "caption"}:
                    continue
                blocks.append(
                    self.make_block(
                        doc_id,
                        order,
                        page_index,
                        normalize_bbox_to_unit(item.get("bbox", [0, 0, 0, 0]), page_width, page_height),
                        canonical,
                        content,
                        popo_type=popo_type,
                        title_level=level,
                        source_label=str(item.get("label", "")),
                    )
                )
                order += 1
        return finalize_reader_result(self.model_name, doc_id, blocks)

    def _load_page_sizes(self, source_file: Any, doc_id: str) -> dict[int, tuple[float, float]]:
        pdf_path = Path(str(source_file)) if source_file else None
        if pdf_path is None or not pdf_path.exists():
            for candidate_dir in DEFAULT_PDF_DIR_CANDIDATES:
                candidate = candidate_dir / f"{doc_id}.pdf"
                if candidate.exists():
                    pdf_path = candidate
                    break
        if pdf_path is None or not pdf_path.exists():
            return {}
        try:
            import fitz

            sizes = {}
            doc = fitz.open(str(pdf_path))
            for index, page in enumerate(doc, start=1):
                sizes[index] = (float(page.rect.width), float(page.rect.height))
            doc.close()
            return sizes
        except Exception:
            return {}


def map_dolphin_label(item: dict[str, Any]) -> tuple[str, str, int | None]:
    label = str(item.get("label", ""))
    if re.fullmatch(r"sec_[1-9]\d*", label):
        level = int(label.split("_", 1)[1])
        return "title", "title", level
    if label == "catalogue":
        return "title", "title", 1
    if label == "fig":
        return "image", "image", None
    if label == "tab":
        return "table", "table", None
    if label == "cap":
        text = normalize_text(item.get("text", ""))
        lowered = text.lower()
        if lowered.startswith("table") or lowered.startswith("tab.") or text.startswith("表"):
            return "caption", "table_caption", None
        return "caption", "image_caption", None
    if label == "list":
        return "text", "list_item", None
    if label == "equ":
        return "text", "equation", None
    return "text", "text", None


class GlmOCRReader(BaseReader):
    model_name = "glm-ocr"

    def read_doc(self, doc_id: str) -> ReaderResult:
        doc_root = self.model_root / doc_id
        model_path = doc_root / f"{doc_id}_model.json"
        merged_path = doc_root / "glm_ocr.json"
        page_paths = sorted(doc_root.glob("page_*.json"))
        if model_path.exists():
            return finalize_reader_result(self.model_name, doc_id, self._load_model_json(doc_id, model_path))
        if merged_path.exists():
            data = safe_json_load(merged_path)
            pages = data.get("pages", [])
            if pages:
                try:
                    return finalize_reader_result(self.model_name, doc_id, self._load_pages(doc_id, pages))
                except ValueError as exc:
                    return ReaderResult(self.model_name, doc_id, "invalid", message=str(exc))
        if page_paths:
            pages = [safe_json_load(path) for path in page_paths]
            try:
                return finalize_reader_result(self.model_name, doc_id, self._load_pages(doc_id, pages))
            except ValueError as exc:
                return ReaderResult(self.model_name, doc_id, "invalid", message=str(exc))
        return ReaderResult(self.model_name, doc_id, "missing", message=f"Missing GLM OCR output under {doc_root}")

    def _load_model_json(self, doc_id: str, model_path: Path) -> list[NormalizedBlock]:
        data = safe_json_load(model_path)
        blocks: list[NormalizedBlock] = []
        order = 0
        for page_index, items in iter_model_pages(data):
            for item in items:
                canonical, popo_type = map_glm_label(str(item.get("label", "text")))
                content = extract_block_content(item)
                if not content and canonical in {"text", "title", "caption"}:
                    continue
                blocks.append(
                    self.make_block(
                        doc_id,
                        order,
                        page_index,
                        normalize_bbox_to_unit(item.get("bbox_2d", item.get("bbox", [0, 0, 0, 0])), assumed_scale=1000),
                        canonical,
                        content,
                        popo_type=popo_type,
                        title_level=parse_numbered_title_level(content) if canonical == "title" else None,
                        source_label=str(item.get("label", "")),
                        meta={"source": "model_json"},
                    )
                )
                order += 1
        return blocks

    def _load_pages(self, doc_id: str, pages: list[dict[str, Any]]) -> list[NormalizedBlock]:
        blocks: list[NormalizedBlock] = []
        order = 0
        for page_index, page in enumerate(pages, start=1):
            page_width, page_height = self._load_fallback_page_size(page)
            response = page.get("response", page)
            for item in response.get("words_result", []):
                content = normalize_text(item.get("words", ""))
                if not content:
                    continue
                location = item.get("location", {})
                bbox = [
                    float(location.get("left", 0)),
                    float(location.get("top", 0)),
                    float(location.get("left", 0)) + float(location.get("width", 0)),
                    float(location.get("top", 0)) + float(location.get("height", 0)),
                ]
                if max(abs(value) for value in bbox) > 1.5 and (not page_width or not page_height):
                    raise ValueError(
                        f"Cannot normalize GLM fallback bbox without page size: doc={doc_id} page={page_index}"
                    )
                canonical = "title" if looks_like_title_text(content) else "text"
                blocks.append(
                    self.make_block(
                        doc_id,
                        order,
                        page_index,
                        normalize_bbox_to_unit(bbox, page_width, page_height),
                        canonical,
                        content,
                        popo_type="title" if canonical == "title" else "text",
                        title_level=parse_numbered_title_level(content) if canonical == "title" else None,
                        source_label="words_result",
                    )
                )
                order += 1
        return blocks

    def _load_fallback_page_size(self, page: dict[str, Any]) -> tuple[float | None, float | None]:
        response = page.get("response", page)
        for holder in (page, response):
            for width_key, height_key in (
                ("width", "height"),
                ("image_width", "image_height"),
                ("page_width", "page_height"),
            ):
                if holder.get(width_key) and holder.get(height_key):
                    return float(holder[width_key]), float(holder[height_key])

        image_path = page.get("image_path") or page.get("img_path") or response.get("image_path") or response.get("img_path")
        if image_path:
            path = Path(str(image_path))
            if path.exists():
                try:
                    from PIL import Image

                    with Image.open(path) as image:
                        return float(image.width), float(image.height)
                except Exception:
                    return None, None
        return None, None


def map_glm_label(label: str) -> tuple[str, str]:
    if label in {"doc_title", "paragraph_title"}:
        return "title", "title"
    if label in {"image", "chart", "seal"}:
        return "image", "image"
    if label == "table":
        return "table", "table"
    if label == "figure_title":
        return "caption", "image_caption"
    if label == "vision_footnote":
        return "caption", "image_footnote"
    if label in {"display_formula", "inline_formula"}:
        return "text", "equation"
    return "text", "text"


def finalize_reader_result(model_name: str, doc_id: str, blocks: list[NormalizedBlock]) -> ReaderResult:
    blocks = infer_missing_title_levels(reassign_block_ids(blocks, doc_id))
    return ReaderResult(model_name=model_name, doc_id=doc_id, status="ok", blocks=blocks)


def build_reader_from_input_dir(model_name: str, input_dir: str | Path, bbox_scale: str = "source") -> BaseReader:
    root = Path(input_dir)
    if model_name == "mineru":
        return MineruReader(root, bbox_scale=bbox_scale)
    if model_name == "monkeyocr":
        return MonkeyOCRReader(root, bbox_scale=bbox_scale)
    if model_name == "PaddleOCR-VL-1.5":
        return PaddleReader(root, bbox_scale=bbox_scale)
    if model_name == "dolphin":
        return DolphinReader(root, bbox_scale=bbox_scale)
    if model_name == "glm-ocr":
        return GlmOCRReader(root, bbox_scale=bbox_scale)
    raise ValueError(f"Unsupported model: {model_name}")


def build_reader(model_name: str, model_outputs_root: str | Path, bbox_scale: str = "source") -> BaseReader:
    return build_reader_from_input_dir(model_name, Path(model_outputs_root) / model_name, bbox_scale=bbox_scale)


def resolve_doc_ids(args: argparse.Namespace) -> list[str]:
    doc_ids: list[str] = []
    if args.doc_id:
        doc_ids.extend(args.doc_id)
    if args.dataset_json:
        data = safe_json_load(args.dataset_json)
        for item in data:
            if isinstance(item, dict) and item.get("image"):
                doc_ids.append(doc_id_from_text(str(item["image"])))
    if not doc_ids:
        doc_ids.extend(infer_doc_ids_from_input_dir(args.input_dir))
    unique_doc_ids = sorted(dict.fromkeys(doc_ids))
    if args.doc_limit:
        unique_doc_ids = unique_doc_ids[: args.doc_limit]
    return unique_doc_ids


def infer_doc_ids_from_input_dir(input_dir: str | Path) -> list[str]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    doc_ids: list[str] = []
    for path in sorted(root.iterdir()):
        try:
            doc_ids.append(doc_id_from_text(path.name))
        except ValueError:
            continue
    return sorted(dict.fromkeys(doc_ids))


def load_pdf_map(path: str | Path) -> dict[str, str]:
    if not path:
        return {}
    raw = safe_json_load(path)
    if not isinstance(raw, dict):
        raise ValueError(f"PDF map must be a JSON object: {path}")
    return {str(key): str(value) for key, value in raw.items()}


def resolve_pdf_dir(explicit_dir: str = "") -> Path | None:
    if explicit_dir:
        path = Path(explicit_dir)
        if not path.exists():
            raise FileNotFoundError(f"PDF directory does not exist: {path}")
        return path
    for path in DEFAULT_PDF_DIR_CANDIDATES:
        if path.exists():
            return path
    return None


def resolve_post_processing_key(doc_id: str, pdf_map: dict[str, str], pdf_dir: Path | None) -> str:
    if doc_id in pdf_map:
        return pdf_map[doc_id]
    if pdf_dir is not None:
        pdf_path = pdf_dir / f"{doc_id}.pdf"
        if pdf_path.exists():
            return str(pdf_path)
    return doc_id


def to_popo_pages(blocks: Iterable[NormalizedBlock]) -> dict[str, list[dict[str, Any]]]:
    pages: dict[str, list[dict[str, Any]]] = {}
    for block in sort_blocks(blocks):
        pages.setdefault(str(block.page), []).append(block.to_popo_block())
    return pages


def title_blocks(blocks: Iterable[NormalizedBlock]) -> list[NormalizedBlock]:
    return [block for block in sort_blocks(blocks) if block.type == "title"]


def serialize_output(result: ReaderResult, output_format: str) -> dict[str, Any]:
    base = {
        "status": result.status,
        "message": result.message,
        "doc_id": result.doc_id,
        "model_name": result.model_name,
    }
    if output_format == "normalized":
        base["blocks"] = [block.to_dict() for block in result.blocks]
    elif output_format == "popo-pages":
        base["pages"] = to_popo_pages(result.blocks)
    elif output_format == "title-blocks":
        base["blocks"] = [block.to_dict() for block in title_blocks(result.blocks)]
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize OCR outputs into Popo-compatible blocks.")
    parser.add_argument("--model", required=True, choices=DEFAULT_MODELS, help="Model name to normalize.")
    parser.add_argument("--input-dir", required=True, help="Directory containing this model's OCR/layout outputs.")
    parser.add_argument("--doc-id", nargs="*", default=[], help="Document ids to normalize.")
    parser.add_argument("--dataset-json", default="", help="Optional benchmark JSON used to extract doc ids from image paths.")
    parser.add_argument("--doc-limit", type=int, default=0, help="Optional debugging limit after doc-id resolution.")
    parser.add_argument("--output-dir", required=True, help="Output root. Writes <model>/<doc_id>.json files.")
    parser.add_argument("--pdf-dir", default="", help="Directory containing source PDFs. Auto-detected in repo-relative eval paths.")
    parser.add_argument("--pdf-map-json", default="", help="Optional JSON mapping doc_id to source PDF path.")
    parser.add_argument("--format", choices=["post-processing", "normalized", "popo-pages", "title-blocks"], default="post-processing")
    parser.add_argument("--bbox-scale", choices=["source", "relative", "thousand"], default="source")
    return parser.parse_args()


def default_output_json(args: argparse.Namespace) -> Path:
    safe_model = re.sub(r"[^0-9A-Za-z_.-]+", "_", args.model)
    return Path(args.output_dir) / f"{safe_model}.post_processing.json"


def default_model_output_dir(args: argparse.Namespace) -> Path:
    safe_model = re.sub(r"[^0-9A-Za-z_.-]+", "_", args.model)
    return Path(args.output_dir) / safe_model


def write_per_doc_output(
    args: argparse.Namespace,
    doc_id: str,
    input_label: str,
    result: ReaderResult,
    output_dir: Path,
) -> Path:
    if args.format == "post-processing":
        payload = {
            "model": args.model,
            "doc_id": doc_id,
            "input_label": input_label,
            "pages": to_popo_pages(result.blocks),
        }
    else:
        payload = {
            "model": args.model,
            "doc_id": doc_id,
            "input_label": input_label,
            "format": args.format,
            **serialize_output(result, args.format),
        }

    output_path = output_dir / f"{doc_id}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    doc_ids = resolve_doc_ids(args)
    if not doc_ids:
        raise ValueError("Cannot resolve any doc ids from --input-dir. Pass --doc-id or --dataset-json explicitly.")

    reader = build_reader_from_input_dir(args.model, args.input_dir, bbox_scale=args.bbox_scale)
    pdf_map = load_pdf_map(args.pdf_map_json)
    pdf_dir = resolve_pdf_dir(args.pdf_dir)

    output_dir = default_model_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    skipped: list[dict[str, str]] = []
    for doc_id in doc_ids:
        result = reader.read_doc(doc_id)
        if result.status != "ok":
            skipped.append({"doc_id": doc_id, "status": result.status, "message": result.message})
            continue
        input_label = resolve_post_processing_key(doc_id, pdf_map, pdf_dir)
        output_path = write_per_doc_output(args, doc_id, input_label, result, output_dir)
        written.append(str(output_path))

    print(
        f"[label-normalization] model={args.model} docs={len(written)}/{len(doc_ids)} "
        f"format={args.format} output_dir={output_dir}",
        flush=True,
    )
    if skipped:
        print(f"[label-normalization] skipped={len(skipped)} first={skipped[0]}", flush=True)


if __name__ == "__main__":
    main()
