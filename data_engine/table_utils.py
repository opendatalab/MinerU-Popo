import re
import ast
import unicodedata
from bs4 import BeautifulSoup


def full_to_half(text):
    if not text:
        return ""
    return unicodedata.normalize('NFKC', text)

def is_hyphen_at_line_end(line):
    return bool(re.search(r'[A-Za-z]+-\s*$', line))

def calculate_row_columns(row):
    cells = row.find_all(["td", "th"])
    column_count = 0
    for cell in cells:
        colspan = int(cell.get("colspan", 1))
        column_count += colspan
    return column_count

def calculate_visual_columns(row):
    cells = row.find_all(["td", "th"])
    return len(cells)

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

def detect_table_headers(soup1, soup2, max_header_rows=5):
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")
    min_rows = min(len(rows1), len(rows2), max_header_rows)
    header_rows = 0
    headers_match = True
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
                text1 = ''.join(full_to_half(cell1.get_text()).split())
                text2 = ''.join(full_to_half(cell2.get_text()).split())
                if colspan1 != colspan2 or rowspan1 != rowspan2 or text1 != text2:
                    structure_match = False
                    break
        if structure_match:
            header_rows += 1
            row_texts = [full_to_half(cell.get_text().strip()) for cell in cells1]
            header_texts.append(row_texts)
        else:
            headers_match = header_rows > 0
            break
    if header_rows == 0:
        headers_match = False
    return header_rows, headers_match, header_texts

def get_visual_last_row_cells_content_with_span_info(soup):
    rows = soup.find_all("tr")
    if not rows:
        return []
    total_cols = calculate_table_total_columns(soup)
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
    final_result = []
    for curr_r_idx in range(last_row_idx, max(-1, last_row_idx - 5), -1):
        if curr_r_idx not in occupied:
            continue
        last_row_positions = occupied[curr_r_idx]
        if not last_row_positions:
            continue
        sorted_cols = sorted(last_row_positions.keys())
        seen_cells = set()
        current_row_data = []
        current_colspan_sum = 0
        for col_idx in sorted_cols:
            cell, origin_row_idx = last_row_positions[col_idx]
            cell_id = id(cell)
            if cell_id not in seen_cells:
                seen_cells.add(cell_id)
                colspan = int(cell.get("colspan", 1))
                rowspan = int(cell.get("rowspan", 1))
                text = cell.get_text(strip=True)
                current_colspan_sum += colspan
                span_parts = []
                if rowspan > 1:
                    span_parts.append("rowspan={}".format(rowspan))
                if colspan > 1:
                    span_parts.append("colspan={}".format(colspan))
                if span_parts:
                    current_row_data.append("{}, {}".format(','.join(span_parts), text))
                else:
                    current_row_data.append("{}".format(text))
        if current_row_data:
            final_result.append(current_row_data)
        cell_count = len(current_row_data)
        is_full_width_line = (cell_count == 1 and current_colspan_sum >= total_cols)
        if not is_full_width_line:
            break
    return final_result[::-1]

def get_table_first_data_row_cells_with_span_info(table_soup, header_rows):
    if table_soup is None:
        return []
    rows = table_soup.find_all("tr")
    if not rows:
        return []
    total_cols = calculate_table_total_columns(table_soup)
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
    start_idx = header_rows if header_rows < len(rows) else len(rows) - 1
    final_result = []
    for curr_r_idx in range(start_idx, min(start_idx + 5, len(rows))):
        if curr_r_idx not in occupied:
            continue
        row_positions = occupied[curr_r_idx]
        if not row_positions:
            continue
        sorted_cols = sorted(row_positions.keys())
        seen_cells = set()
        current_row = []
        current_colspan_sum = 0
        for col_idx in sorted_cols:
            cell, origin_row_idx = row_positions[col_idx]
            cell_id = id(cell)
            if cell_id not in seen_cells:
                seen_cells.add(cell_id)
                colspan = int(cell.get("colspan", 1))
                rowspan = int(cell.get("rowspan", 1))
                current_colspan_sum += colspan
                text = full_to_half(cell.get_text(strip=True))
                span_parts = []
                if rowspan > 1:
                    span_parts.append("rowspan={}".format(rowspan))
                if colspan > 1:
                    span_parts.append("colspan={}".format(colspan))
                if span_parts:
                    current_row.append("{}, {}".format(','.join(span_parts), text))
                else:
                    current_row.append("{}".format(text))
        if current_row:
            final_result.append(current_row)
        cell_count = len(current_row)
        is_full_width_line = (cell_count == 1 and current_colspan_sum >= total_cols)
        if not is_full_width_line:
            break
    return final_result

def build_table_occupied_matrix(soup):
    rows = soup.find_all("tr")
    if not rows:
        return {}
    occupied = {}
    row_effective_cols = {}
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
        if occupied[row_idx]:
            row_effective_cols[row_idx] = max(occupied[row_idx].keys()) + 1
        else:
            row_effective_cols[row_idx] = 0
    return row_effective_cols

def calculate_row_effective_columns(soup, row_idx):
    row_effective_cols = build_table_occupied_matrix(soup)
    return row_effective_cols.get(row_idx, 0)

def detect_table_headers_visual(soup1, soup2, max_header_rows=5):
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")
    min_rows = min(len(rows1), len(rows2), max_header_rows)
    effective_cols1 = build_table_occupied_matrix(soup1)
    effective_cols2 = build_table_occupied_matrix(soup2)
    header_rows = 0
    headers_match = True
    header_texts = []
    for i in range(min_rows):
        cells1 = rows1[i].find_all(["td", "th"])
        cells2 = rows2[i].find_all(["td", "th"])
        texts1 = [''.join(full_to_half(cell.get_text()).split()) for cell in cells1]
        texts2 = [''.join(full_to_half(cell.get_text()).split()) for cell in cells2]
        effective_cols_match = effective_cols1.get(i, 0) == effective_cols2.get(i, 0)
        if texts1 == texts2 and effective_cols_match:
            header_rows += 1
            row_texts = [full_to_half(cell.get_text().strip()) for cell in cells1]
            header_texts.append(row_texts)
        else:
            headers_match = header_rows > 0
            break
    if header_rows == 0:
        headers_match = False
    return header_rows, headers_match, header_texts

def extract_last_coordinates(response_text):
    ed = response_text.rfind(']')
    if ed == -1:
        return None
    balance = 0
    st = -1
    for i in range(ed, -1, -1):
        char = response_text[i]
        if char == ']':
            balance += 1
        elif char == '[':
            balance -= 1
        if balance == 0:
            st = i
            break
    if st != -1:
        target_str = response_text[st : ed + 1]
        try:
            result = ast.literal_eval(target_str)
            return result
        except Exception as e:
            print("extract_last_coordinates parse error: {}".format(e))
            return None
    return None
