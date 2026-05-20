import copy
import os
import sys
import json
import re
import ast
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import fitz
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import threading
import base64
import io
import traceback
from bs4 import BeautifulSoup
from model_utils import *

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_engine'))
from table_merge_filter import filter_table_merge_candidates
from table_utils import (
    detect_table_headers,
    get_visual_last_row_cells_content_with_span_info,
    get_table_first_data_row_cells_with_span_info,
    extract_last_coordinates,
)


def concatenate_pdf_pages_with_border(doc_label, pages, border_width=5, border_color='black'):
    """Filter related pages and synthesize them"""
    pdf_path = doc_label
    
    doc = fitz.open(pdf_path)
    pil_images = []
    if not pages:
        pages = [1]
    for page_num in pages:
        page = doc[page_num - 1]
        pix = page.get_pixmap()
        img_data = pix.tobytes("jpeg")
        img = Image.open(io.BytesIO(img_data))
        
        draw = ImageDraw.Draw(img)
        
        avg_cell_size = (page.rect.width + page.rect.height) / 2
        font_size = int(avg_cell_size * 0.1)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
        draw.text((10, 10), str(page_num), fill=(255, 0, 0), font=font)
        
        pil_images.append(img)
    doc.close()

    
    total_height = sum(img.height for img in pil_images) + border_width * (len(pil_images) - 1)
    max_width = max(img.width for img in pil_images)
    
    result = Image.new('RGB', (max_width, total_height), color='white')
    y_offset = 0
    for i, img in enumerate(pil_images):
        x_offset = (max_width - img.width) // 2
        result.paste(img, (x_offset, y_offset))
        
        if i < len(pil_images) - 1:
            y_border = y_offset + img.height
            draw = ImageDraw.Draw(result)
            draw.rectangle(
                [0, y_border, max_width - 1, y_border + border_width - 1],
                fill=border_color
            )
            y_offset = y_border + border_width
        else:
            y_offset += img.height
    
    buffered = io.BytesIO()

    
    buffered.seek(0)
    buffered.truncate(0)
    result.save(buffered, format="JPEG", quality=100, optimize=True)
        
    base64_result = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return base64_result

termination_chars = {
    ".",
    "。",
    "?",
    "!",
    "？",
    "！",
    "¿",
    "¡",
    "؟",
    "ฯ",
    "۔",
    ":",   
    "：",
    "……",
    ";",
    "；"
}

close_chars = {
    "’",
    "”",
    "'",
    "\"",
    "）",
    "」",
    "】",
    "]",
    ")",
}

termination_pattern = "[" + re.escape("".join(termination_chars)) + "]"

def get_tail_sentence(text):
    """Get the last sentence"""
    if not text.strip():
        return text.strip()
    parts = re.split(termination_pattern, text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts[-1] if parts else text.strip()

def get_head_sentence(text):
    """Get the first sentence"""
    if not text.strip():
        return text.strip()
    match = re.search(termination_pattern, text)
    if match:
        end_index = match.end()
        return text[:end_index].strip()
    else:
        return text.strip()

def is_list_item(s):
    """Judge if the text is a preifx of a list item"""

    pattern = r'''
        ^
        (?:
            # 1. 2. or 1) 2)
            \d+[\.\)] | 
            \d+） | 
            \d+．| 
            \d+、|
            # (1) (2)
            \(\d+\) |
            \([a-z]\) |
            \([A-Z]\) |
            （\d+） |
            # A. B. or A) B)
            [A-Z][\.\)] | 
            # a. b. or a) b)
            [a-z][\.\)] | 
            # CN number（一、 二、）
            [一二三四五六七八九十百千万]+、 | 
            \([一二三四五六七八九十百千万]+\) | 
            （一二三四五六七八九十百千万]+） | 
            # ① ② ③
            [①-⑳㉑-㉟] | 
            # • ▪ ▫
            [•▪▫] | 
            # Ⅰ. Ⅱ. Ⅲ.
            [IVXLCDM]+\. |
            - |
            \$ |
            \\t |
            [\[\(「【（] |
            # CN section
            第[一二三四五六七八九十百千万][条节章]
        )
        \s*                    
    '''

    return re.match(pattern, s, re.IGNORECASE | re.VERBOSE) is not None

def merge_rules(str1, str2):
    """Heuristics of text termination, prefix and length"""
    if not str1 or not str2:
        return False

    str1_trimmed = str1.strip()
    if str1_trimmed and str1_trimmed[-1] in termination_chars:
        return False
    if len(str1_trimmed) > 1 and str1_trimmed[-1] in close_chars and str1_trimmed[-2] in termination_chars:
        return False
    
    if len(str1) < 10:
        return False
    
    if is_list_item(str2):
        return False
    
    if "\t" in str1 or "\t" in str2:
        return False

    if str1[0].isdigit() and str2[0].isdigit():
        return False

    return True

def filter_contd(blocks):
    """Filter input for Text Truncation Analysis"""
    potential_idx = []
    judge_blocks = []
    valid_pairs = {}
    for i, block in enumerate(blocks):
        if block["type"] in ["text", "list_item"]:
            for pos in range(i+1, len(blocks)):
                if 'equation' in blocks[pos]['type'] or 'title' in blocks[pos]['type']:
                    break
                if blocks[pos]['type'] in ["text", "list_item"] and merge_rules(block["content"], blocks[pos]['content']):
                    if i not in potential_idx:
                        potential_idx.append(i)
                    if pos not in potential_idx:
                        potential_idx.append(pos)
                    valid_pairs[i] = pos
                    break
    
    for i in potential_idx:
        text = blocks[i]['content']
        bbox = blocks[i]['bbox']
        head = get_head_sentence(text)
        tail = get_tail_sentence(text)
        text_short = head if head == tail else head + ' ... ' + tail
        text_short = text_short if len(text_short) <= 103 else text_short[:50] + '...' + text_short[-50:]
        judge_blocks.append({'idx': i, 'content': text_short, 'page':blocks[i]['page'], 'bbox': [str(int(bbox[1]*1000)), str(int(bbox[0]*1000)), str(int(bbox[3]*1000)), str(int(bbox[2]*1000))]})

    return judge_blocks


def filter_title(blocks, tol = 0.05):
    """Filter input for Title Hierarchy Analysis"""
    judge_blocks = []
    res = []
    for i, block in enumerate(blocks):
        bbox = blocks[i]['bbox']
        if block["type"] in ["title", "TOC-title", "section-title"]:
            judge_blocks.append({'idx': i, 'content': blocks[i]['content'][:50], 'page':blocks[i]['page'], 'bbox': [str(int(bbox[1]*1000)), str(int(bbox[0]*1000)), str(int(bbox[3]*1000)), str(int(bbox[2]*1000))]})

    return judge_blocks

def check_overlap(visual_blocks, large_blocks):
    judge_blocks = []
    large_block_linking = {}
    for i, block in visual_blocks.items():
        flag = True
        bbox = block["bbox"]
        for j, lblock in large_blocks.items():
            lbbox = lblock["bbox"]
            if (block["type"] in ['image', 'chart', 'image_footnote', 'image_caption', 'figure', 'fig-title', 'fig-caption']) and (bbox[0] >= 0.95*lbbox[0] and bbox[1] >= 0.95*lbbox[1] and bbox[2] <= 1.05*lbbox[2] and bbox[3] <= 1.05*lbbox[3]):
                large_block_linking[i] = j
                flag = False
                break
        if flag:
            typ = block['type']
            if typ in ['image_block', 'chart', 'figure']:
                typ = 'image'
            elif typ in ["title", "TOC-title", "section-title"]:
                typ = 'title'
            elif typ in ['fig-title', 'fig-caption']:
                typ = 'image_caption'
            elif typ in ['tab-title', 'tab-caption']:
                typ = 'table_caption'
            content =  "None" if typ in ['image', 'table'] else block['content']
            judge_blocks.append({'idx': i, 'type': typ, 'content': content[:50], 'page':block['page'], 'bbox': [str(int(bbox[1]*1000)), str(int(bbox[0]*1000)), str(int(bbox[3]*1000)), str(int(bbox[2]*1000))]})
    return judge_blocks, large_block_linking

def filter_image(blocks, tol = 0.05):
    """Filter input for Text-Image Association Analysis"""
    visual_blocks = {}
    large_blocks = {}
    res = []
    for i, block in enumerate(blocks):
        if block["type"] == 'seal':
            block["type"] = 'image'
        if block["type"] in ['image_block', 'image', 'table', 'chart', 'table_footnote', 'image_footnote', 'table_caption', 'image_caption', "title", "TOC-title", "section-title", 'figure', 'fig-title', 'fig-caption', 'tab-title', 'tab-caption']:
            visual_blocks[i] = block
        if block["type"] in ['image_block']:
            large_blocks[i] = block

    judge_blocks, exist_linking = check_overlap(visual_blocks, large_blocks)

    return judge_blocks, exist_linking

def filter_table_merge(blocks):
    """Filter input for Table Merge Detection.

    Finds adjacent-page table pairs that pass pre-LLM screening,
    and prepares the row data needed for the merge prompt.

    Returns:
        list of dicts: [{"table1_idx": int, "table2_idx": int,
                          "upper_row_ss": list, "lower_row_ss": list}]
    """
    tables_by_page = {}
    for i, block in enumerate(blocks):
        if block["type"] == "table":
            page = block["page"]
            if page not in tables_by_page:
                tables_by_page[page] = []
            tables_by_page[page].append(i)

    if len(tables_by_page) < 2:
        return []

    merge_inputs = []
    sorted_pages = sorted(tables_by_page.keys())

    for p_idx in range(len(sorted_pages) - 1):
        page1 = sorted_pages[p_idx]
        page2 = sorted_pages[p_idx + 1]
        if page2 != page1 + 1:
            continue

        table1_idx = tables_by_page[page1][-1]
        table2_idx = tables_by_page[page2][0]

        can_merge, reason = filter_table_merge_candidates(blocks, table1_idx, table2_idx)
        if not can_merge:
            continue

        try:
            soup1 = BeautifulSoup(blocks[table1_idx].get("content", ""), "html.parser")
            soup2 = BeautifulSoup(blocks[table2_idx].get("content", ""), "html.parser")
        except Exception:
            continue

        if not soup1.find_all("tr") or not soup2.find_all("tr"):
            continue

        header_rows, _, _ = detect_table_headers(soup1, soup2)
        upper_row_ss = get_visual_last_row_cells_content_with_span_info(soup1)
        lower_row_ss = get_table_first_data_row_cells_with_span_info(soup2, header_rows)
        merge_inputs.append({
            "table1_idx": table1_idx,
            "table2_idx": table2_idx,
            "upper_row_ss": upper_row_ss,
            "lower_row_ss": lower_row_ss,
        })

    return merge_inputs

def add_contd(judge_blocks):
    input_list = []
    output_list = []
    for block in judge_blocks:
        input_list.append(f"<|id|>{block['idx']}<|page|>{block['page']}<|box|>{' '.join(block['bbox'])}<|content|>{block['content']}")
    input_txt = '\n'.join(input_list)
    
    return f"<image>\nTruncation Detection: {input_txt}"

def add_title(judge_blocks):
    input_list = []
    output_list = []
    for block in judge_blocks:
        input_list.append(f"<|id|>{block['idx']}<|page|>{block['page']}<|box|>{' '.join(block['bbox'])}<|content|>{block['content']}")
    input_txt = '\n'.join(input_list)
    
    return f"<image>\nTitle Level Analysis: {input_txt}"

def add_image(judge_blocks):
    input_list = []
    output_list = []
    for block in judge_blocks:
        input_list.append(f"<|id|>{block['idx']}<|type|>{block['type']}<|page|>{block['page']}<|box|>{' '.join(block['bbox'])}<|content|>{block['content']}")
    input_txt = '\n'.join(input_list)

    return f"<image>\nImage-Text Correlation Analysis: {input_txt}"

def add_table_merge(upper_row_ss, lower_row_ss):
    prompt = f"""
## Table 1 (Previous Page - Last Table)

**Caption:** :""
**Last Row(s) Data:**
{upper_row_ss}

---

## Table 2 (Current Page - First Table)

**Caption:** :""
**First Data Row(s):**
{lower_row_ss}
"""
    return prompt

def extract_label1(s):

    result = []
    for line in s.strip().split('\n'):
        try:
            if not line:
                continue
            src_part, tgt_part = line.split("<|tgt_id|>")
            src_id = src_part.split("<|src_id|>")[1]
            tgt_id = tgt_part
            result.append({"src_id": int(src_id), "tgt_id": int(tgt_id)})
        except Exception as e:
            None
            # print(f"Error format: {line}")
    return result

def extract_label2(s):

    result = []
    for line in s.strip().split('\n'):
        try:
            if not line:
                continue
            id_part, level_part = line.split("<|level|>")
            idx = id_part.split("<|id|>")[1]
            level = level_part
            if int(level) >= 0:
                result.append({"id": int(idx), "level": int(level)})
        except Exception as e:
            None
            # print(f"Error format: {line}")
    return result

def parse_string_notype(input_string):
    prefix1 = "<image>\nTruncation Detection: "
    prefix2 = "<image>\nTitle Level Analysis: "
    if input_string.startswith(prefix1):
        content = input_string[len(prefix1):]
    elif input_string.startswith(prefix2):
        content = input_string[len(prefix2):]
    else:
        content = input_string
    
    content = '\n' + content
    parts = content.split("\n<|id|>")
    
    if parts and parts[0] == "":
        parts = parts[1:]
    
    results = []
    
    for part in parts:
        if not part.strip():
            continue
        
        page_split = part.split("<|page|>")
        id_value = page_split[0].strip()
        
        box_split = page_split[1].split("<|box|>")
        page_value = box_split[0].strip()
        
        content_split = box_split[1].split("<|content|>")
        box_value = content_split[0].strip()
        content_value = content_split[1].strip() if len(content_split) > 1 else ""
        
        item = {
            'id': id_value,
            'page': int(page_value),
            'box': box_value,
            'content': content_value
        }
        results.append(item)
    return results

def parse_string_type(input_string):
    prefix = "<image>\nImage-Text Correlation Analysis: "
    if input_string.startswith(prefix):
        content = input_string[len(prefix):]
    else:
        content = input_string
    
    content = '\n' + content
    parts = content.split("\n<|id|>")
    if parts and parts[0] == "":
        parts = parts[1:]
    
    results = []
    
    for part in parts:
        if not part.strip():
            continue

        type_split = part.split("<|type|>")
        id_value = type_split[0].strip()

        page_split = type_split[1].split("<|page|>")
        type_value = page_split[0].strip()
        
        box_split = page_split[1].split("<|box|>")
        page_value = box_split[0].strip()
        
        content_split = box_split[1].split("<|content|>")
        box_value = content_split[0].strip()
        content_value = content_split[1].strip() if len(content_split) > 1 else ""
        
        item = {
            'id': id_value,
            'type': type_value,
            'page': int(page_value),
            'box': box_value,
            'content': content_value
        }
        results.append(item)
    return results


def adaptive_chunk(items, chunk_size=50, overlap=1):
    if not items:
        return [], []
    
    sorted_items = sorted(items, key=lambda x: x['page'])
    pages = [item['page'] for item in sorted_items]
    unique_pages = sorted(set(pages))
    
    boundaries = []
    current_min = unique_pages[0]
    
    while current_min < unique_pages[-1]:
        target = current_min + chunk_size
        
        # Search from pages in a range
        search_range = range(max(unique_pages[0], target-5), 
                           min(unique_pages[-1], target+5)+1)
        
        # Count the frequency of the target type of item in the page
        freq = {}
        for page in search_range:
            if page in unique_pages:
                freq[page] = pages.count(page)
        
        if freq:
            boundary = max(freq, key=freq.get)
        else:
            boundary = min((x for x in unique_pages if x > target), default=unique_pages[-1])
        
        boundaries.append(boundary)
        current_min = boundary
    
    # Chunking
    ranges = []
    chunks = []
    prev_boundary = unique_pages[0]
    
    for boundary in boundaries:
        chunk_items = [item for item in sorted_items 
                      if prev_boundary - overlap <= item['page'] <= boundary + overlap]
        if chunk_items:
            chunks.append(chunk_items)
            start = prev_boundary if prev_boundary == unique_pages[0] else prev_boundary - overlap
            end = min(unique_pages[-1], boundary + overlap)
            ranges.append([start, end])
        prev_boundary = boundary
    
    # The last chunk
    last_chunk = [item for item in sorted_items 
                 if prev_boundary - overlap <= item['page'] <= unique_pages[-1]]
    if last_chunk:
        
        start = prev_boundary if prev_boundary == unique_pages[0] else prev_boundary - overlap
        if unique_pages[-1] - start > 2:
            chunks.append(last_chunk)
            ranges.append([start, unique_pages[-1]])
    
    return ranges, chunks


def safe_doc_stem(input_label):
    base = os.path.basename(str(input_label))
    stem, ext = os.path.splitext(base)
    return stem if ext else base


def write_raw_record(raw_doc_dir, task, chunk_index, rng, pages, prompt, raw_response, parsed=None, extra=None):
    if not raw_doc_dir:
        return
    os.makedirs(raw_doc_dir, exist_ok=True)
    payload = {
        "task": task,
        "chunk_index": chunk_index,
        "range": rng,
        "pages": pages,
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed": parsed if parsed is not None else [],
    }
    if extra:
        payload.update(extra)
    output_path = os.path.join(raw_doc_dir, f"{task}_chunk_{chunk_index:04d}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)


def write_raw_summary(raw_doc_dir, payload):
    if not raw_doc_dir:
        return
    os.makedirs(raw_doc_dir, exist_ok=True)
    with open(os.path.join(raw_doc_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)


def main(input_label, ocr_res, output_dir, raw_output_dir=None):
    """
    Post-Processing the OCR result of each document
    """
    doc_stem = safe_doc_stem(input_label)
    raw_doc_dir = os.path.join(raw_output_dir, doc_stem) if raw_output_dir else None
    print(os.path.join(output_dir, f"{doc_stem}.json"))
    item = ocr_res
    doc_blocks = []
    doc_pages = []
    title_pages = []
    
    idx = 1
    for page_num, blocks in item.items():
        for block in blocks:
            block['page'] = int(page_num)
            block['id'] = idx
            idx += 1
        doc_blocks.extend(blocks)
    
    for block in doc_blocks:
        block["contd"] = -1
        block["level"] = -1
        block["image"] = -1

    contd = add_contd(filter_contd(doc_blocks))
    title = add_title(filter_title(doc_blocks))

    image_judge_blocks, large_block_linking = filter_image(doc_blocks)
    image = add_image(image_judge_blocks)
    
    # Text Truncation Analysis with Dynamic Chunking
    ranges, chunks = adaptive_chunk(parse_string_notype(contd))
    results = []

    async def process_chunk_contd(chunk_index, rng, chunk, input_label, results):
        pages = list(set([block['page'] for block in chunk]))
        pages = sorted([page for page in pages if rng[0] <= page <= rng[1]])
        base64_image = concatenate_pdf_pages_with_border(input_label, pages)
        texts = []
        for block in chunk:
            texts.append(f"<|id|>{block['id']}<|page|>{block['page']}<|box|>{block['box']}<|content|>{block['content']}")
        text = '\n'.join(texts)
        text = '<image>\nTruncation Detection: ' + text

        raw_response = popo_generate(text, base64_image)
        id_pairs = extract_label1(raw_response.replace('<|from|>','<|src_id|>').replace('<|to|>','<|tgt_id|>'))
        write_raw_record(raw_doc_dir, "contd", chunk_index, rng, pages, text, raw_response, id_pairs)
        for pair in id_pairs:
            if pair not in results:
                results.append(pair)
    
    async def process_contd(ranges, chunks, input_label, results):
        tasks = [
            process_chunk_contd(index, rng, chunk, input_label, results)
            for index, (rng, chunk) in enumerate(zip(ranges, chunks))
        ]
        result = await asyncio.gather(*tasks)
    asyncio.run(process_contd(ranges, chunks, input_label, results))

    texts = []
    for pair in results:
        texts.append(f"<|src_id|>{pair['src_id']}<|tgt_id|>{pair['tgt_id']}")
    contd_output = '\n'.join(texts)
    

    # Title Hierarchy Analysis with Dynamic Chunking
    ranges, chunks = adaptive_chunk(parse_string_notype(title))
    order_res = {}
    results = []

    async def process_chunk_title(chunk_index, rng, chunk, input_label, order_res):
        pages = list(set([block['page'] for block in chunk]))
        pages = sorted([page for page in pages if rng[0] <= page <= rng[1]])
        base64_image = concatenate_pdf_pages_with_border(input_label, pages)
        texts = []
        for block in chunk:
            texts.append(f"<|id|>{block['id']}<|page|>{block['page']}<|box|>{block['box']}<|content|>{block['content']}")
        text = '\n'.join(texts)
        text = '<image>\nTitle Level Analysis: ' + text
        
        raw_response = popo_generate(text, base64_image)
        id_pairs = extract_label2(raw_response)
        write_raw_record(raw_doc_dir, "title", chunk_index, rng, pages, text, raw_response, id_pairs)
        order_res[rng[0]+rng[1]] = id_pairs
    
    async def process_title(ranges, chunks, input_label, order_res):
        tasks = [
            process_chunk_title(index, rng, chunk, input_label, order_res)
            for index, (rng, chunk) in enumerate(zip(ranges, chunks))
        ]
        result = await asyncio.gather(*tasks)
    asyncio.run(process_title(ranges, chunks, input_label, order_res))
    
    # Chunk level Synchronization
    order_res = dict(sorted(order_res.items()))
    for key, id_pairs in order_res.items():
        bias = []
        for pair in id_pairs:
            idx = pair['id']
            for exist in results:
                if idx == exist["id"]:
                    if pair['level'] < 0 or exist['level'] < 0:
                        pair['level'] = -1
                        exist['level'] = -1
                    else:
                        bias.append(pair['level'] - exist['level'])
                        pair['level'] = exist['level']
                    break
        
        avg_bias = round(sum(bias) / len(bias)) if len(bias) > 0 else 0
        for pair in id_pairs:
            idx = pair['id']
            ext_flag = False
            for exist in results:
                if idx == exist["id"]:
                    ext_flag =True
                    if pair['level'] != exist['level']:
                        print(f"Level Alert!!!")
                        print(id_pairs)
                    break
            if not ext_flag:
                pair['level'] = pair['level'] - avg_bias if pair['level'] > 0 else pair['level']
                results.append(pair)

    texts = []
    for pair in results:
        texts.append(f"<|id|>{pair['id']}<|level|>{pair['level']}")      
    title_output = '\n'.join(texts)
    
    # Association Analysis with Dynamic Chunking
    ranges, chunks = adaptive_chunk(parse_string_type(image))
    results = []
    async def process_chunk_image(chunk_index, rng, chunk, input_label, results):
        pages = list(set([block['page'] for block in chunk]))
        pages = sorted([page for page in pages if rng[0] <= page <= rng[1]])
        base64_image = concatenate_pdf_pages_with_border(input_label, pages)
        texts = []
        for block in chunk:
            texts.append(f"<|id|>{block['id']}<|type|>{block['type']}<|page|>{block['page']}<|box|>{block['box']}<|content|>{block['content']}")
        text = '\n'.join(texts)
        text = '<image>\nImage-Text Correlation Analysis: ' + text
        
        raw_response = popo_generate(text, base64_image)
        id_pairs = extract_label1(raw_response)
        write_raw_record(raw_doc_dir, "image", chunk_index, rng, pages, text, raw_response, id_pairs)
        for pair in id_pairs:
            if pair not in results:
                results.append(pair)

    async def process_image(ranges, chunks, input_label, results):
        tasks = [
            process_chunk_image(index, rng, chunk, input_label, results)
            for index, (rng, chunk) in enumerate(zip(ranges, chunks))
        ]
        result = await asyncio.gather(*tasks)
    asyncio.run(process_image(ranges, chunks, input_label, results))

    texts = []
    for pair in results:
        texts.append(f"<|src_id|>{pair['src_id']}<|tgt_id|>{pair['tgt_id']}")
        
    image_output = '\n'.join(texts)
    
    contd_label = extract_label1(contd_output)
    title_label = extract_label2(title_output)
    image_label = extract_label1(image_output)
    
    for label_pair in contd_label:
        try:
            doc_blocks[label_pair["src_id"]]["contd"] = label_pair["tgt_id"] + 1
        except Exception as e:
            print(e)
    for label_pair in image_label:
        try:
            doc_blocks[label_pair["src_id"]]["image"] = label_pair["tgt_id"] + 1
        except Exception as e:
            print(e)
    for label_pair in title_label:
        try:
            doc_blocks[label_pair["id"]]["level"] = label_pair["level"]
        except Exception as e:
            print(e)

    for src, tgt in large_block_linking.items():
        doc_blocks[src]["image"] = tgt + 1

    # ============================================================
    # Table Merge Detection (text-only, EVAL-style prompt)
    # ============================================================
    for block in doc_blocks:
        if block["type"] == "table":
            block["table_merge"] = -1

    async def process_table_merge_pair(chunk_index, mi):
        prompt = add_table_merge(mi["upper_row_ss"], mi["lower_row_ss"])
        try:
            raw_output = popo_generate(prompt, None)
            cell_list = extract_last_coordinates(raw_output)
            write_raw_record(
                raw_doc_dir,
                "table_merge",
                chunk_index,
                None,
                [],
                prompt,
                raw_output,
                cell_list,
                extra={"table1_idx": mi["table1_idx"], "table2_idx": mi["table2_idx"]},
            )
            if cell_list and isinstance(cell_list, list) and len(cell_list) > 0:
                doc_blocks[mi["table1_idx"]]["table_merge"] = doc_blocks[mi["table2_idx"]]["id"]
                doc_blocks[mi["table2_idx"]]["table_merge"] = doc_blocks[mi["table1_idx"]]["id"]
                doc_blocks[mi["table1_idx"]]["cell_list"] = cell_list
                doc_blocks[mi["table2_idx"]]["cell_list"] = cell_list
        except Exception as e:
            print(f"Table merge error: {e}")

    merge_inputs = filter_table_merge(doc_blocks)
    if merge_inputs:
        async def process_all_merges():
            tasks = [process_table_merge_pair(index, mi) for index, mi in enumerate(merge_inputs)]
            await asyncio.gather(*tasks)
        asyncio.run(process_all_merges())

    write_raw_summary(
        raw_doc_dir,
        {
            "input_label": input_label,
            "output_json": os.path.join(output_dir, f"{doc_stem}.json"),
            "contd_output": contd_output,
            "title_output": title_output,
            "image_output": image_output,
            "contd_label": contd_label,
            "title_label": title_label,
            "image_label": image_label,
            "table_merge_candidates": merge_inputs,
        },
    )

    with open(os.path.join(output_dir, f"{doc_stem}.json"), 'w', encoding='utf-8') as f:
        json.dump(doc_blocks, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    output_dir = ""
    input_file = ""
    os.makedirs(output_dir, exist_ok=True)
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for input_label, ocr_res in tqdm(data.items()):
        main(input_label, ocr_res, output_dir)
