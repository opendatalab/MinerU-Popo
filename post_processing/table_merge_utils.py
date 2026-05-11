import json
from bs4 import BeautifulSoup


def calculate_table_total_columns(soup):
    rows = soup.find_all("tr")
    if not rows:
        return 0

    max_cols = 0
    occupied = {}
    for row_idx, row in enumerate(rows):
        col_idx = 0
        cells = row.find_all(["td", "th"])
        if row_idx not in occupied:
            occupied[row_idx] = {}
        for cell in cells:
            while col_idx in occupied[row_idx]:
                col_idx += 1
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
            for r in range(row_idx, row_idx + rowspan):
                if r not in occupied:
                    occupied[r] = {}
                for c in range(col_idx, col_idx + colspan):
                    occupied[r][c] = True
            col_idx += colspan
            max_cols = max(max_cols, col_idx)
    return max_cols


def calculate_row_columns(row):
    return sum(int(cell.get("colspan", 1)) for cell in row.find_all(["td", "th"]))


def calculate_visual_columns(row):
    return len(row.find_all(["td", "th"]))


def check_row_columns_match(row1, row2):
    cells1 = row1.find_all(["td", "th"])
    cells2 = row2.find_all(["td", "th"])
    if len(cells1) != len(cells2):
        return False
    for cell1, cell2 in zip(cells1, cells2):
        if int(cell1.get("colspan", 1)) != int(cell2.get("colspan", 1)):
            return False
    return True


def adjust_table_rows_colspan(rows, start_idx, end_idx,
                              reference_structure, reference_visual_cols,
                              target_cols, current_cols, reference_row):
    reference_row_copy = BeautifulSoup(str(reference_row), "html.parser").find(["tr"])

    for i in range(start_idx, end_idx):
        row = rows[i]
        cells = row.find_all(["td", "th"])
        if not cells:
            continue

        current_row_cols = calculate_row_columns(row)
        if current_row_cols >= target_cols:
            continue

        if calculate_visual_columns(row) == reference_visual_cols and check_row_columns_match(row, reference_row_copy):
            if len(cells) <= len(reference_structure):
                for j, cell in enumerate(cells):
                    if j < len(reference_structure) and reference_structure[j] > 1:
                        cell["colspan"] = str(reference_structure[j])
        else:
            last_cell = cells[-1]
            current_last_span = int(last_cell.get("colspan", 1))
            last_cell["colspan"] = str(current_last_span + (target_cols - current_cols))


def detect_table_headers(soup1, soup2, max_header_rows=5):
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")
    min_rows = min(len(rows1), len(rows2), max_header_rows)
    header_rows = 0
    header_texts = []

    for i in range(min_rows):
        cells1 = rows1[i].find_all(["td", "th"])
        cells2 = rows2[i].find_all(["td", "th"])
        structure_match = True
        if len(cells1) != len(cells2):
            structure_match = False
        else:
            for cell1, cell2 in zip(cells1, cells2):
                colspan1 = int(cell1.get("colspan", 1))
                rowspan1 = int(cell1.get("rowspan", 1))
                colspan2 = int(cell2.get("colspan", 1))
                rowspan2 = int(cell2.get("rowspan", 1))
                text1 = "".join(cell1.get_text().split())
                text2 = "".join(cell2.get_text().split())
                if colspan1 != colspan2 or rowspan1 != rowspan2 or text1 != text2:
                    structure_match = False
                    break
        if structure_match:
            header_rows += 1
            header_texts.append([cell.get_text(strip=True) for cell in cells1])
        else:
            break
    return header_rows, header_rows > 0, header_texts


def get_visual_last_row_cells(soup):
    rows = soup.find_all("tr")
    if not rows:
        return []

    last_row_idx = len(rows) - 1
    occupied = {}
    for row_idx, row in enumerate(rows):
        col_idx = 0
        cells = row.find_all(["td", "th"])
        if row_idx not in occupied:
            occupied[row_idx] = {}
        for cell in cells:
            while col_idx in occupied[row_idx]:
                col_idx += 1
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
            for r in range(row_idx, row_idx + rowspan):
                if r not in occupied:
                    occupied[r] = {}
                for c in range(col_idx, col_idx + colspan):
                    occupied[r][c] = (cell, row_idx)
            col_idx += colspan

    if last_row_idx not in occupied or not occupied[last_row_idx]:
        return []

    sorted_cols = sorted(occupied[last_row_idx].keys())
    seen_cells = set()
    result = []
    for col_idx in sorted_cols:
        cell, origin_row_idx = occupied[last_row_idx][col_idx]
        cell_id = id(cell)
        if cell_id not in seen_cells:
            seen_cells.add(cell_id)
            result.append((cell, origin_row_idx))
    return result


def merge_row_cells_by_cell_list(visual_last_row_cells, first_data_row2, cell_list, total_rows1):
    cells2 = first_data_row2.find_all(["td", "th"])
    max_idx = min(len(visual_last_row_cells), len(cells2), len(cell_list))
    last_row_idx = total_rows1 - 1
    cells_to_remove = []

    for i in range(max_idx):
        if cell_list[i] != 1:
            continue

        c1, origin_row_idx = visual_last_row_cells[i]
        c2 = cells2[i]
        text1 = c1.get_text(" ", strip=True)
        text2 = c2.get_text(" ", strip=True)
        merged_text = "".join(part for part in [text1, text2] if part)

        for child in list(c1.contents):
            child.extract()
        if merged_text:
            c1.string = merged_text

        c2_rowspan = int(c2.get("rowspan", 1))
        if origin_row_idx < last_row_idx:
            current_rowspan = int(c1.get("rowspan", 1))
            c1["rowspan"] = str(current_rowspan + c2_rowspan)
            cells_to_remove.append(c2)
        elif c2_rowspan > 1:
            c1["rowspan"] = str(c2_rowspan)
            cells_to_remove.append(c2)
        else:
            for child in list(c2.contents):
                child.extract()

    for cell in cells_to_remove:
        cell.decompose()

    cells2_remaining = first_data_row2.find_all(["td", "th"])
    if len(cells2_remaining) == 0:
        return True
    if all(not c.get_text(strip=True) for c in cells2_remaining):
        return True
    return False


def merge_table_html(previous_html, current_html, cell_list):
    """Merge two cross-page table HTML fragments using Magic-PDF style semantic merge."""
    soup1 = BeautifulSoup(previous_html, "html.parser")
    soup2 = BeautifulSoup(current_html, "html.parser")

    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")
    if not rows1 or not rows2:
        return previous_html

    tbody1 = soup1.find("tbody") or soup1.find("table")
    tbody2 = soup2.find("tbody") or soup2.find("table")
    if not tbody1 or not tbody2:
        return previous_html

    header_count, _, _ = detect_table_headers(soup1, soup2)
    last_row1 = rows1[-1]
    first_data_row2 = rows2[header_count] if header_count < len(rows2) else None
    skip_first_data_row = False

    if rows1 and rows2 and header_count < len(rows2):
        visual_last_row_cells = get_visual_last_row_cells(soup1)
        table_cols1 = calculate_table_total_columns(soup1)
        table_cols2 = calculate_table_total_columns(soup2)

        if table_cols1 >= table_cols2 and first_data_row2 is not None:
            reference_structure = [int(cell.get("colspan", 1)) for cell in last_row1.find_all(["td", "th"])]
            reference_visual_cols = calculate_visual_columns(last_row1)
            adjust_table_rows_colspan(
                rows2, header_count, len(rows2),
                reference_structure, reference_visual_cols,
                table_cols1, table_cols2, first_data_row2
            )
        elif first_data_row2 is not None:
            reference_structure = [int(cell.get("colspan", 1)) for cell in first_data_row2.find_all(["td", "th"])]
            reference_visual_cols = calculate_visual_columns(first_data_row2)
            adjust_table_rows_colspan(
                rows1, 0, len(rows1),
                reference_structure, reference_visual_cols,
                table_cols2, table_cols1, last_row1
            )

        if first_data_row2 is not None and isinstance(cell_list, list):
            skip_first_data_row = merge_row_cells_by_cell_list(
                visual_last_row_cells,
                first_data_row2,
                cell_list,
                len(rows1),
            )

    start_idx = header_count + 1 if skip_first_data_row else header_count
    for row in rows2[start_idx:]:
        row.extract()
        tbody1.append(row)

    return str(soup1)


def merge_cross_page_tables(elements):
    id_to_index = {element["id"]: idx for idx, element in enumerate(elements) if "id" in element}
    merged_partner_ids = set()

    for idx, element in enumerate(elements):
        if element.get("type") != "table":
            continue
        partner_id = element.get("table_merge", -1)
        cell_list = element.get("cell_list", [])
        if partner_id < 0 or partner_id in merged_partner_ids:
            continue
        partner_index = id_to_index.get(partner_id)
        if partner_index is None or partner_index <= idx:
            continue

        partner = elements[partner_index]
        if partner.get("type") != "table":
            continue

        merged_html = merge_table_html(element.get("content", ""), partner.get("content", ""), cell_list)
        element["content"] = merged_html
        element["merged_locations"] = [
            {"bbox": element["bbox"], "page": element["page"]},
            {"bbox": partner["bbox"], "page": partner["page"]},
        ]
        element["merged_block_ids"] = [element["id"], partner["id"]]

        for other in elements:
            if other.get("image") == partner["id"]:
                other["image"] = element["id"]

        merged_partner_ids.add(partner_id)

    return [element for element in elements if element.get("id") not in merged_partner_ids]
