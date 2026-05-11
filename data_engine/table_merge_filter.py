import re
import unicodedata
from bs4 import BeautifulSoup

from table_utils import (
    full_to_half,
    calculate_table_total_columns,
    calculate_row_columns,
    calculate_visual_columns,
    calculate_row_effective_columns,
    build_table_occupied_matrix,
    detect_table_headers_visual,
)

CONTINUATION_END_MARKERS = [
    "(续)",
    "(续表)",
    "(续上表)",
    "(continued)",
    "(cont.)",
    "(cont'd)",
    "(…continued)",
    "续表",
]

CONTINUATION_INLINE_MARKERS = [
    "(continued)",
]

def _get_caption_for_table(blocks, table_idx):
    """Find the caption associated with a table block.

    Looks at the block immediately before the table. Accepts:
      - table_caption, tab-title, tab-caption  (explicit caption types)
      - title  (many captions are recognized as titles)
      - text if short (<150 chars) and matches a caption-like pattern

    Args:
        blocks: list of all blocks in reading order
        table_idx: index of the table block in blocks

    Returns:
        str or None
    """
    if table_idx <= 0:
        return None

    prev = blocks[table_idx - 1]
    typ = prev.get("type", "")
    content = (prev.get("content", "") or "").strip()

    # 1. Explicit caption types
    if typ in ("table_caption", "tab-title", "tab-caption"):
        return content

    # 2. Title (many captions are recognized as "title")
    if typ == "title":
        return content

    # 3. Short text that looks like a caption
    if typ == "text" and len(content) < 150:
        pattern = r'^(table|表|exhibit|figure|图)\s*\d+|^\d+(\.\d+)*\s+|^[一二三四五六七八九十]+、'
        if re.match(pattern, content, re.IGNORECASE):
            return content

    return None

def _get_footnotes_for_table(blocks, table_idx):
    """Collect footnote blocks associated with a table.

    Looks at blocks after the table on the same page that are
    table_footnote or table_footnote-like blocks.

    Args:
        blocks: list of all blocks in reading order
        table_idx: index of the table block in blocks

    Returns:
        list of footnote blocks
    """
    footnotes = []
    table_page = blocks[table_idx].get("page")
    for i in range(table_idx + 1, len(blocks)):
        b = blocks[i]
        if b.get("page") != table_page:
            break
        if b.get("type") in ("table_footnote", "table_caption", "tab-caption", "tab-title"):
            footnotes.append(b)
            continue
        # Stop at the next non-footnote block
        if b.get("type") != "table_footnote":
            break
    return footnotes



def _check_text_between_tables(blocks, table1_idx, table2_idx):
    """Reject if any TEXT block sits between the two tables on their pages.

    - On page N-1: scan blocks AFTER table1 → reject if any "text" block found
    - On page N:   scan blocks BEFORE table2  → reject if any "text" block found
    """
    table1_page = blocks[table1_idx].get("page")
    table2_page = blocks[table2_idx].get("page")

    # After table1 on its page
    for i in range(table1_idx + 1, len(blocks)):
        b = blocks[i]
        if b.get("page") != table1_page:
            break
        if b.get("type") in ("text", "list_item", "list"):
            return False, "text block found after table1 on same page"

    # Before table2 on its page
    for i in range(table2_idx - 1, -1, -1):
        b = blocks[i]
        if b.get("page") != table2_page:
            break
        if b.get("type") in ("text", "list_item", "list"):
            return False, "text block found before table2 on same page"

    return True, ""

def _check_caption_consistency(blocks, table1_idx, table2_idx):
    """Check if table captions are consistent.

    Logic (same as Magic-PDF):
      - Neither has caption   → consistent (continuation inferred)
      - Table1 has, Table2 not → consistent (continuation table)
      - Table1 not, Table2 has → INCONSISTENT (new table, should NOT merge)
      - Both have captions     → compare table numbers, then compare cleaned text
    """
    cap1 = _get_caption_for_table(blocks, table1_idx)
    cap2 = _get_caption_for_table(blocks, table2_idx)

    if not cap1 and not cap2:
        return True, "both tables have no caption"
    if cap1 and not cap2:
        return True, "only table1 has caption, table2 is continuation"
    if not cap1 and cap2:
        return False, "table2 has caption but table1 does not — new table"

    # Both have captions — compare table numbers
    patterns = [
        (r'[Ee]xhibit\s*(\d+)', 'Exhibit'),
        (r'EXHIBIT\s*(\d+)', 'Exhibit'),
        (r'[Tt]able\s*([A-Za-z]?\s*-?\s*[\d.]+)', 'Table'),
        (r'[Tt]ab\.?\s*([A-Za-z]?\s*-?\s*[\d.]+)', 'Table'),
        (r'TABLE\s*([A-Za-z]?\s*-?\s*[\d.]+)', 'Table'),
        (r'表\s*([A-Za-z]?\s*-?\s*[\d.]+)', 'Table'),
        (r'[Ff]igure\s*(\d+)', 'Figure'),
        (r'图\s*(\d+)', 'Figure'),
        (r'^([一二三四五六七八九十]+)、', 'CN_Num'),
    ]

    def _extract_table_number(caption):
        for pat, num_type in patterns:
            m = re.search(pat, caption)
            if m:
                num_str = m.group(1).replace(' ', '').replace('-', '').upper()
                return num_type, num_str
        return None

    r1 = _extract_table_number(cap1)
    r2 = _extract_table_number(cap2)

    if r1 and r2:
        if r1[0] != r2[0] or r1[1] != r2[1]:
            return False, "table numbers differ: {} vs {}".format(r1, r2)
        return True, "table numbers match: {}".format(r1)

    # No table numbers — compare cleaned text
    def _clean_caption(text):
        text = full_to_half(text).lower()
        for marker in CONTINUATION_END_MARKERS:
            if text.endswith(marker.lower()):
                text = text[:-len(marker)]
        text = re.sub(r'[^\w\u4e00-\u9fa5]', '', text)
        return text

    if _clean_caption(cap1) != _clean_caption(cap2):
        return False, "caption text differs too much"
    return True, "captions are consistent"



def _check_continuation_marker(blocks, table2_idx):
    """If table2 has captions, at least one must contain a continuation marker.

    e.g. "(续)", "(续表)", "(continued)", "(cont.)", etc.
    """
    cap2 = _get_caption_for_table(blocks, table2_idx)
    if not cap2:
        return True, "table2 has no caption, no continuation marker needed"

    cap_lower = full_to_half(cap2).lower()
    for marker in CONTINUATION_END_MARKERS:
        if cap_lower.endswith(marker.lower()):
            return True, "continuation marker found: {}".format(marker)
    for marker in CONTINUATION_INLINE_MARKERS:
        if marker.lower() in cap_lower:
            return True, "continuation marker found: {}".format(marker)

    return False, "table2 has caption but no continuation marker: {}".format(cap2[:80])



def _check_footnote_count(blocks, table1_idx, table2_idx):
    """If table2 has captions with continuation marker → allow ≤ 1 footnote on table1.
    If table2 has no caption → require 0 footnotes on table1.
    """
    footnotes = _get_footnotes_for_table(blocks, table1_idx)
    fn_count = len(footnotes)

    cap2 = _get_caption_for_table(blocks, table2_idx)
    if cap2:
        # Continuation marker already verified by check 3, relax to ≤ 1 footnote
        if fn_count > 1:
            return False, "too many footnotes on table1: {}".format(fn_count)
    else:
        if fn_count > 0:
            return False, "table1 has footnotes but table2 has no caption"

    return True, ""



def _check_width_difference(blocks, table1_idx, table2_idx):
    """Reject if bbox widths differ by >= 10%."""
    bbox1 = blocks[table1_idx].get("bbox", [0, 0, 0, 0])
    bbox2 = blocks[table2_idx].get("bbox", [0, 0, 0, 0])

    w1 = bbox1[2] - bbox1[0]
    w2 = bbox2[2] - bbox2[0]
    if min(w1, w2) <= 0:
        return True, "zero width, skip width check"

    diff_ratio = abs(w1 - w2) / min(w1, w2)
    if diff_ratio >= 0.1:
        return False, "width diff {:.1%} >= 10% (w1={:.2f}, w2={:.2f})".format(diff_ratio, w1, w2)
    return True, ""



def _check_rows_match(soup1, soup2):
    """Check if the last data row of table1 and first data row of table2
    match in column structure (effective, actual, or visual columns).

    Same logic as Magic-PDF's check_rows_match.
    """
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")
    if not rows1 or not rows2:
        return False

    # Last row of table1 that has cells
    last_row, last_row_idx = None, None
    for idx in range(len(rows1) - 1, -1, -1):
        if rows1[idx].find_all(["td", "th"]):
            last_row = rows1[idx]
            last_row_idx = idx
            break

    # Detect header rows between the two tables
    from table_utils import detect_table_headers
    header_count, _, _ = detect_table_headers(soup1, soup2)

    # First data row of table2
    first_data_row, first_data_row_idx = None, None
    if len(rows2) > header_count:
        first_data_row = rows2[header_count]
        first_data_row_idx = header_count

    if not (last_row and first_data_row):
        return False

    # Effective columns (accounting for rowspan)
    last_eff = calculate_row_effective_columns(soup1, last_row_idx)
    first_eff = calculate_row_effective_columns(soup2, first_data_row_idx)

    # Column-span columns
    last_cols = calculate_row_columns(last_row)
    first_cols = calculate_row_columns(first_data_row)

    # Visual columns (raw cell count)
    last_vis = calculate_visual_columns(last_row)
    first_vis = calculate_visual_columns(first_data_row)

    return (last_eff == first_eff or
            last_cols == first_cols or
            last_vis == first_vis)


def _check_column_count(blocks, table1_idx, table2_idx):
    """Check if table column structures are compatible.

    Two checks:
      1. Total columns match (overall table structure)
      2. Row-level columns match (last row of table1 vs first data row of table2)

    If EITHER matches, the tables are considered structurally compatible.
    """
    html1 = blocks[table1_idx].get("content", "")
    html2 = blocks[table2_idx].get("content", "")

    if not html1 or not html2:
        return False, "empty HTML content"

    try:
        soup1 = BeautifulSoup(html1, "html.parser")
        soup2 = BeautifulSoup(html2, "html.parser")
    except Exception:
        return False, "failed to parse HTML"

    if not soup1.find_all("tr") or not soup2.find_all("tr"):
        return False, "no table rows found in HTML"

    cols1 = calculate_table_total_columns(soup1)
    cols2 = calculate_table_total_columns(soup2)

    if cols1 == cols2:
        return True, "total columns match: {}".format(cols1)

    rows_match = _check_rows_match(soup1, soup2)
    if rows_match:
        return True, "row-level columns match (total: {} vs {})".format(cols1, cols2)

    return False, "column structures incompatible (total: {} vs {})".format(cols1, cols2)



def filter_table_merge_candidates(blocks, table1_idx, table2_idx):
    """Screen whether two table blocks should be considered for merging.

    Runs all 6 heuristic checks in order. Returns early on first rejection.

    Args:
        blocks: flat list of all document blocks (already with 'id', 'page', 'type', 'content', 'bbox')
        table1_idx: index of the first table (on page N-1)
        table2_idx: index of the second table (on page N)

    Returns:
        (can_merge: bool, reason: str)
    """
    checks = [
        ("text_between_tables", _check_text_between_tables),
        ("caption_consistency", _check_caption_consistency),
        ("continuation_marker", _check_continuation_marker),
        ("footnote_count", _check_footnote_count),
        ("width_difference", _check_width_difference),
        ("column_count", _check_column_count),
    ]

    for name, check_fn in checks:
        if name in ("text_between_tables", "caption_consistency"):
            ok, reason = check_fn(blocks, table1_idx, table2_idx)
        elif name == "continuation_marker":
            ok, reason = check_fn(blocks, table2_idx)
        elif name == "footnote_count":
            ok, reason = check_fn(blocks, table1_idx, table2_idx)
        elif name in ("width_difference", "column_count"):
            ok, reason = check_fn(blocks, table1_idx, table2_idx)
        else:
            ok, reason = check_fn(blocks, table1_idx, table2_idx)

        if not ok:
            return False, "[{}] {}".format(name, reason)

    return True, "all checks passed"