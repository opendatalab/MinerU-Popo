"""Tests for img_path propagation through the normalization layer.

Regression tests for https://github.com/opendatalab/MinerU-Popo/issues/11:
MineruReader was reading img_path from content_list only to infer page
dimensions via PIL, then dropping it. These tests verify that img_path is
now propagated onto NormalizedBlock so downstream consumers (build_tree,
document-tree users like passion-index) can locate image files directly.
"""
import json
import sys
from pathlib import Path

# Allow running from post_processing/ or repo root
sys.path.insert(0, str(Path(__file__).parent))

from label_normalization import MineruReader  # noqa: E402


def _write_content_list(root: Path, doc_id: str, content_list: list[dict]) -> Path:
    """Build a synthetic MinerU doc dir with a content_list.json inside <root>/<doc>/vlm/."""
    doc_dir = root / doc_id / "vlm"
    doc_dir.mkdir(parents=True, exist_ok=True)
    path = doc_dir / f"{doc_id}_content_list.json"
    path.write_text(json.dumps(content_list), encoding="utf-8")
    return path


def test_image_block_carries_img_path(tmp_path):
    """img_path on an image item in content_list should land on the
    corresponding NormalizedBlock.img_path."""
    content_list = [
        {"type": "text", "content": "hello", "bbox": [0, 0, 100, 100], "page_idx": 0},
        {
            "type": "image",
            "img_path": "images/abc123.jpg",
            "bbox": [0, 0, 100, 100],
            "page_idx": 0,
            "content": "",
        },
    ]
    _write_content_list(tmp_path, "doc1", content_list)

    reader = MineruReader(tmp_path)
    result = reader.read_doc("doc1")

    image_blocks = [b for b in result.blocks if b.type == "image"]
    assert len(image_blocks) == 1, f"expected 1 image block, got {len(image_blocks)}"
    assert image_blocks[0].img_path == "images/abc123.jpg", (
        f"img_path should propagate from content_list, got {image_blocks[0].img_path!r}"
    )


def test_non_image_blocks_have_empty_img_path(tmp_path):
    """Text/title/caption blocks should keep img_path="" (MinerU only puts
    img_path on image items)."""
    content_list = [
        {"type": "text", "content": "hello", "bbox": [0, 0, 100, 100], "page_idx": 0},
        {"type": "title", "content": "Chapter", "bbox": [0, 0, 100, 100], "page_idx": 0},
    ]
    _write_content_list(tmp_path, "doc1", content_list)

    reader = MineruReader(tmp_path)
    result = reader.read_doc("doc1")

    assert len(result.blocks) > 0
    for block in result.blocks:
        assert block.img_path == "", (
            f"{block.type} block should have empty img_path, got {block.img_path!r}"
        )


def test_img_path_absent_in_source_defaults_empty(tmp_path):
    """Defensive: if an image item somehow lacks img_path, the field should
    default to "" rather than raising KeyError."""
    content_list = [
        {
            "type": "image",
            # NOTE: no "img_path" key
            "bbox": [0, 0, 100, 100],
            "page_idx": 0,
            "content": "",
        },
    ]
    _write_content_list(tmp_path, "doc1", content_list)

    reader = MineruReader(tmp_path)
    result = reader.read_doc("doc1")

    image_blocks = [b for b in result.blocks if b.type == "image"]
    assert len(image_blocks) == 1
    assert image_blocks[0].img_path == ""


def test_cp_init_includes_img_path():
    """get_json_tree.cp_init should accept and propagate img_path to the
    component dict (used for image/table/chart nodes in build_tree)."""
    from get_json_tree import cp_init

    cp = cp_init(cp_type="image", content="", level=42, img_path="images/foo.jpg")
    assert cp["img_path"] == "images/foo.jpg"

    # Default empty string when not passed
    cp2 = cp_init(cp_type="text")
    assert cp2["img_path"] == ""


# --- middle.json path tests (the real path MineruReader takes; fixes #11) ---

def test_extract_image_path_from_middle_json_layout(tmp_path):
    """_extract_image_path should pull image_path from middle.json's deeply
    nested structure: para_block.blocks[*].lines[*].spans[*].image_path."""
    # Synthetic middle.json image block
    item = {
        "type": "image",
        "bbox": [30, 55, 505, 673],
        "blocks": [
            {
                "type": "image_body",
                "lines": [
                    {
                        "bbox": [30, 55, 505, 673],
                        "spans": [
                            {
                                "bbox": [30, 55, 505, 673],
                                "type": "image",
                                "image_path": "abc123.jpg",  # middle.json uses "image_path"
                            }
                        ],
                    }
                ],
            }
        ],
    }
    reader = MineruReader(tmp_path)
    assert reader._extract_image_path(item) == "abc123.jpg"


def test_extract_image_path_missing_or_empty(tmp_path):
    """_extract_image_path returns "" for various missing-field shapes."""
    reader = MineruReader(tmp_path)
    assert reader._extract_image_path({}) == ""
    assert reader._extract_image_path({"type": "image"}) == ""  # no blocks
    assert reader._extract_image_path({"blocks": []}) == ""
    assert reader._extract_image_path({"blocks": [{"lines": []}]}) == ""
    # Span without image_path key
    assert reader._extract_image_path(
        {"blocks": [{"lines": [{"spans": [{"type": "image"}]}]}]}
    ) == ""


def test_read_middle_extracts_img_path(tmp_path):
    """End-to-end: read_doc on a doc with only middle.json (no content_list)
    should populate NormalizedBlock.img_path from the nested image_path field.

    This is the PRIMARY path — read_doc prefers middle.json over content_list.
    """
    middle_json = {
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [600, 800],
                "para_blocks": [
                    {
                        "type": "image",
                        "bbox": [30, 55, 505, 673],
                        "blocks": [
                            {
                                "type": "image_body",
                                "lines": [
                                    {
                                        "bbox": [30, 55, 505, 673],
                                        "spans": [
                                            {
                                                "bbox": [30, 55, 505, 673],
                                                "type": "image",
                                                "image_path": "xyz789.jpg",
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }

    doc_id = "doc1"
    doc_dir = tmp_path / doc_id / "vlm"
    doc_dir.mkdir(parents=True)
    (doc_dir / f"{doc_id}_middle.json").write_text(json.dumps(middle_json))

    reader = MineruReader(tmp_path)
    result = reader.read_doc(doc_id)

    image_blocks = [b for b in result.blocks if b.type == "image"]
    assert len(image_blocks) == 1, f"expected 1 image block, got {len(image_blocks)}"
    assert image_blocks[0].img_path == "xyz789.jpg", (
        f"_read_middle should extract image_path from nested structure, "
        f"got {image_blocks[0].img_path!r}"
    )


def test_read_middle_extracts_img_path_for_table(tmp_path):
    """Table blocks also carry an embedded raster image (MinerU renders tables
    as images). _read_middle should extract img_path for them too, not just
    for type=image blocks.
    """
    middle_json = {
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [600, 800],
                "para_blocks": [
                    {
                        "type": "table",
                        "bbox": [30, 55, 505, 673],
                        "blocks": [
                            {
                                "type": "table_body",
                                "lines": [
                                    {
                                        "bbox": [30, 55, 505, 673],
                                        "spans": [
                                            {
                                                "bbox": [30, 55, 505, 673],
                                                "type": "image",
                                                "image_path": "table_raster.jpg",
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }

    doc_id = "doc1"
    doc_dir = tmp_path / doc_id / "vlm"
    doc_dir.mkdir(parents=True)
    (doc_dir / f"{doc_id}_middle.json").write_text(json.dumps(middle_json))

    reader = MineruReader(tmp_path)
    result = reader.read_doc(doc_id)

    table_blocks = [b for b in result.blocks if b.type == "table"]
    assert len(table_blocks) == 1
    assert table_blocks[0].img_path == "table_raster.jpg", (
        f"_read_middle should extract img_path for table blocks too, "
        f"got {table_blocks[0].img_path!r}"
    )


def test_read_middle_extracts_caption_for_table(tmp_path):
    """Table blocks should carry caption text extracted from middle.json's
    sub_blocks (type=table_caption, text in lines[*].spans[*].content).
    """
    middle_json = {
        "pdf_info": [{
            "page_idx": 0,
            "page_size": [600, 800],
            "para_blocks": [{
                "type": "table",
                "bbox": [30, 55, 505, 200],
                "blocks": [
                    {
                        "type": "table_caption",
                        "lines": [{"spans": [{"content": "Table 1. Patient Characteristics."}]}],
                    },
                    {
                        "type": "table_body",
                        "lines": [{"spans": [{"image_path": "table1.jpg"}]}],
                    },
                ],
            }],
        }]
    }
    doc_id = "doc1"
    doc_dir = tmp_path / doc_id / "vlm"
    doc_dir.mkdir(parents=True)
    (doc_dir / f"{doc_id}_middle.json").write_text(json.dumps(middle_json))

    reader = MineruReader(tmp_path)
    result = reader.read_doc(doc_id)

    tables = [b for b in result.blocks if b.type == "table"]
    assert len(tables) == 1
    assert tables[0].caption == "Table 1. Patient Characteristics.", (
        f"expected caption extracted, got {tables[0].caption!r}"
    )
    assert tables[0].img_path == "table1.jpg"


def test_to_popo_block_emits_caption(tmp_path):
    """to_popo_block should emit caption when set (alongside img_path)."""
    content_list = [
        {
            "type": "table",
            "img_path": "images/t1.jpg",
            "table_caption": ["Table 2. Efficacy Outcomes."],
            "bbox": [0, 0, 100, 100],
            "page_idx": 0,
            "content": "",
        }
    ]
    _write_content_list(tmp_path, "doc1", content_list)
    reader = MineruReader(tmp_path)
    result = reader.read_doc("doc1")

    tables = [b for b in result.blocks if b.type == "table"]
    assert len(tables) == 1
    pb = tables[0].to_popo_block()
    assert pb.get("caption") == "Table 2. Efficacy Outcomes.", f"caption missing in to_popo_block: {pb!r}"
    assert pb.get("img_path") == "images/t1.jpg"


def test_to_popo_block_emits_img_path(tmp_path):
    """to_popo_block (used by label_normalization's to_popo_pages output)
    should emit img_path when set, so the field survives into the JSON that
    inference.py + build_tree.py consume.

    Without this, the field is dropped at serialization even though
    NormalizedBlock carries it (regression test for the original #11 fix
    that was incomplete).
    """
    from label_normalization import to_popo_pages

    content_list = [
        {
            "type": "image",
            "img_path": "images/abc.jpg",
            "bbox": [0, 0, 100, 100],
            "page_idx": 0,
            "content": "",
        },
        {
            "type": "text",
            "content": "hello",
            "bbox": [0, 0, 100, 100],
            "page_idx": 0,
        },
    ]
    _write_content_list(tmp_path, "doc1", content_list)
    reader = MineruReader(tmp_path)
    result = reader.read_doc("doc1")

    # to_popo_block on the image block should include img_path
    image_blocks = [b for b in result.blocks if b.type == "image"]
    assert len(image_blocks) == 1
    pb = image_blocks[0].to_popo_block()
    assert pb.get("img_path") == "images/abc.jpg", (
        f"to_popo_block should emit img_path, got {pb!r}"
    )

    # text blocks should NOT carry img_path (keep output clean)
    text_blocks = [b for b in result.blocks if b.type == "text"]
    for tb in text_blocks:
        assert "img_path" not in tb.to_popo_block(), (
            f"text block to_popo_block should not emit empty img_path, got {tb.to_popo_block()!r}"
        )

    # to_popo_pages output (what label_normalization actually writes)
    pages = to_popo_pages(result.blocks)
    flat = [b for blocks in pages.values() for b in blocks]
    image_dicts = [b for b in flat if b.get("type") == "image"]
    assert len(image_dicts) == 1
    assert image_dicts[0].get("img_path") == "images/abc.jpg"
