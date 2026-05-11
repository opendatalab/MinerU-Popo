import copy
import os
import json
import re
import ast
from PIL import Image, ImageDraw, ImageFont
import fitz
import base64
import io
from bs4 import BeautifulSoup

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import traceback

from api_utils import *
from table_utils import (
    full_to_half,
    is_hyphen_at_line_end,
    calculate_row_columns,
    calculate_visual_columns,
    calculate_table_total_columns,
    detect_table_headers,
    get_visual_last_row_cells_content_with_span_info,
    get_table_first_data_row_cells_with_span_info,
    extract_last_coordinates,
)
from table_merge_filter import filter_table_merge_candidates

txt_contd_prompt = """
### Instruction
Please identify which text blocks in the document are truncated due to column breaks, page breaks, inserted images or tables, and therefore need to be reconnected into a coherent and complete sentence. 
Consider visual factors that may cause truncation, as well as grammatical correctness and semantic coherence from a content perspective. Output the idx of text blocks that you believe require reconnection and briefly explain the reason.

### Input Schema
The input is a **single JSON object** containing:

* `image`: the image of the document (servused as visual evidence, and multiple pages are separated by black lines)
* `blocks`: a list of text blocks with potential truncation, where each element contains:

  * `idx`: unique identifier of the block (**must be referenced in the output**)
  * `content`: completed or broken sentences (the middle parts of long paragraphs are omitted)
  * `page`: number of the page where the block is located
  * `bbox`: location of the block on the page

### Input Blocks
_RAW_LIST_

### Output Schema
```json
[
  {
    "src": <int>, // The idx of the text block that needs connection at its end 
    "tgt": <int>, // The idx of the text block that needs connection at its start
    reason: <str> // Brief explanation. For example, 'The text in block x and block y are broken due to ...; the complete sentence at the connection point is ...'
  },
  ... // More connections
] 
```

### Judgment Criteria
(1) The src and tgt must be two adjacent blocks in Input Blocks.
(2) Merge when two text pieces are separated due to layout factors, and the connection can be grammatically and semantically coherent to form a complete sentence.
(3) Truncated text blocks usually have similar font style and size.
(4) If there are a line break after standalone phrases or complete sentences between src and tgt, no reconnection is needed. 

### Note
Respond strictly in the specified JSON output format.
"""

title_level_prompt = """
### Instruction
Given a list of titles (marked by blue boxes in page images) in a document, please analyze title levels according to the page layout, visual clues and semantics. 
Identify different levels with different numbers. Use 1 for the highest level title, 2 for the second level, and so on.
Some non-title content may be mixed into the input blocks, and you can mark them with the -1 level.

### Few-shot Examples of Title Hierarchy
["2000", "2010", "2020"] ---> ["#2000", "#2010", "#2020"]
["Report", "1.Challenge" , "2.Method", "3.Result"] ---> ["#Report", "##1.Challenge" , "##2.Method", "##3.Result"]
["Schedule", "Day1", "Afternoon", "Night", "Day2", "Morning"] ---> ["#Schedule", "##Day1", "###Afternoon", "###Night", "##Day2", "###Morning"]

### Input Schema
The input is a **single JSON object** containing:

* `image`: the image of the document (servused as visual evidence, and multiple pages are separated by black lines)
* `blocks`: a list of titles blocks, where each element contains:

  * `idx`: unique identifier of the block (**must be referenced in the output**)
  * `content`: the textual content of the title
  * `page`: number of the page where the block is located
  * `bbox`: location of the block on the page

### Input Blocks
_RAW_LIST_

### Output Schema
```json
[
  {
    idx: <int>, // The idx of the title block
    level: <int> // The level of the title (-1 if not a title)
  },
  ... // More results
] 
```

### Judgment Criteria
(1) A title, semantically, serves as a summary and overarching structure for the main content of a document. Visually, it should stand on its own line and may have a distinct font style.
(2) The title numbering in normative documents naturally indicates their hierarchy like 1. | 1.1 | 1.2 | 2.;
(3) In page layout, the overall titles of each main block are usually parallel, while multiple titles within a block may be parallel or nested;
(4) Visually, multiple consecutive parallel titles usually have the same font and size, and larger font sizes generally indicate higher levels, but visual judgment rules are not always reliable;
(5) For title levels cannot be judged by title numbering, layout, and visual features, ultimately determine the hierarchy based on your understanding of the title and context semantics.

### Note
Respond strictly in the specified JSON output format. Each input block should be referred once in the output to identify its level.
"""

image_link_prompt = """
### Instruction
Given a list of element blocks in a document, please analyze the correlation between images, tables and text according to the criterias. 

### Input Schema
The input is a **single JSON object** containing:

* `image`: the image of the document (servused as visual evidence, and multiple pages are separated by black lines)
* `blocks`: a list of titles blocks, where each element contains:

  * `idx`: unique identifier of the block (**must be referenced in the output**)
  * `type`: the type of document elements (related to the criterias)
  * `content`: the textual content of the block (None for images and tables)
  * `page`: number of the page where the block is located
  * `bbox`: location of the block on the page

### Input Blocks
_RAW_LIST_

### Output Schema
```json
[
  {
    "src": <int>, // The idx of the current block
    "tgt": <int>, // The idx of the target block to which the current block should be linked
    reason: <str> // Carefully examine the reasonableness of the correlation to ensure it aligns with the actual document structure and the criterias
  },
  ... // More results
] 
```

### Judgment Criteria
(1) The block with type `image` or `table` should be linked to the most related `title` block;
(2) The block with type `image_caption` or `image_footnote` should be linked to the most related `image` block;
(3) The block with type `table_caption` or `table_footnote` should be linked to the most related `table` block;
(4) Links not falling into the above three situations cannot be connected.

### Note
Respond strictly in the specified JSON output format. You can only output links that meet the criterias.
"""

table_merge_prompt = """
### Instruction
Given two table fragments split across consecutive PDF pages, determine whether they belong to the same cross-page table and identify which columns require semantic merging.

### Input Schema
The input contains:

* `upper_caption`: the caption of the table on the previous page
* `upper_row`: the last few rows of the table on the previous page
* `lower_caption`: the caption of the table on the next page
* `lower_row`: the first few data rows of the table on the next page

### Input Blocks
_RAW_LIST_

### Output Schema
[
    {
        judgement:<list> # A list indicating cell-level merging decision
    }
]

### Judgment Criteria
(1) If the two table fragments have different column counts or clearly cannot belong to the same table, output [].
(2) Otherwise, output exactly one object with a judgment list.
(3) The length of judgment must be equal to the number of columns in the table. Each value corresponds to one column from left to right.
(4) Use 1 if the two cells in the same column should be semantically merged into one logical cell, such as in cases of hyphenation, an incomplete phrase, a split date/number, or rowspan continuation.
(5) Use 0 if the two cells in the same column are independent row contents, even if they are similar or identical.
(6) For empty cells, make the decision based on the overall row context and table captions.

### NOTE
Respond strictly in the specified JSON output format. Do not explain your reasoning.
"""

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

def llm_generate_json(prompt, base64_image):
    """Instruct LLM to generate in JSON foramt"""
    try_cnt = 0
    while try_cnt < 5:
        try:
            gpt_output = gpt_generate_pdf(base64_image, prompt)
            json_pattern = r'```json\n(.*?)\n```'
            match = re.search(json_pattern, gpt_output, re.DOTALL)

            if match:
                json_str = match.group(1)
            else:
                start = gpt_output.find('[')
                end = gpt_output.rfind(']')
                
                if start != -1 and end != -1 and end > start:
                    json_str = gpt_output[start:end+1]
                else:
                    json_str = gpt_output
            res = json.loads(json_str)
            return res
        except Exception as e:
            print(f"Fail to parse json:{e}")
            try_cnt += 1

    return "!!LLM_ERROR!!"


def contd_format(judge_blocks, raw_result, img_path):
    input_list = []
    output_list = []
    for block in judge_blocks:
        input_list.append(f"<|id|>{block['idx']}<|page|>{block['page']}<|box|>{' '.join(block['bbox'])}<|content|>{block['content']}")
    input_txt = '\n'.join(input_list)
    
    for one_res in raw_result:
        output_list.append(f"<|src_id|>{one_res['src']}<|tgt_id|>{one_res['tgt']}")
    output_txt = '\n'.join(output_list)

    item = {
        "image": img_path,
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\nTruncation Detection: {input_txt}"
            },
            {
                "from": "gpt",
                "value": output_txt
            }
        ],
            "type": "grounding"
    }
    return item

def title_format(judge_blocks, raw_result, img_path):
    input_list = []
    output_list = []
    for block in judge_blocks:
        input_list.append(f"<|id|>{block['idx']}<|page|>{block['page']}<|box|>{' '.join(block['bbox'])}<|content|>{block['content']}")
    input_txt = '\n'.join(input_list)
    
    for one_res in raw_result:
        output_list.append(f"<|id|>{one_res['idx']}<|level|>{one_res['level']}")
    output_txt = '\n'.join(output_list)

    item = {
        "image": img_path,
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\nTitle Level Analysis: {input_txt}"
            },
            {
                "from": "gpt",
                "value": output_txt
            }
        ],
            "type": "grounding"
    }
    return item

def image_format(judge_blocks, raw_result, img_path):
    input_list = []
    output_list = []
    for block in judge_blocks:
        input_list.append(f"<|id|>{block['idx']}<|type|>{block['type']}<|page|>{block['page']}<|box|>{' '.join(block['bbox'])}<|content|>{block['content']}")
    input_txt = '\n'.join(input_list)
    
    for one_res in raw_result:
        output_list.append(f"<|src_id|>{one_res['src']}<|tgt_id|>{one_res['tgt']}")
    output_txt = '\n'.join(output_list)

    item = {
        "image": img_path,
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\nImage-Text Correlation Analysis: {input_txt}"
            },
            {
                "from": "gpt",
                "value": output_txt
            }
        ],
            "type": "grounding"
    }
    return item

def table_merge_format(judge_blocks, raw_result, img_path):
    input_list = []
    for block in judge_blocks:
        input_list.append("<|id|>{}<|page|>{}<|col_count|>{}<|caption|>{}<|last_rows|>{}<|first_rows|>{}".format(
            block['idx'], block['page'], block['col_count'], block['caption'],
            json.dumps(block['last_rows'], ensure_ascii=False),
            json.dumps(block['first_rows'], ensure_ascii=False)))
    input_txt = '\n'.join(input_list)

    output_list = []
    for one_res in raw_result:
        cell_str = ','.join(str(x) for x in one_res.get('cell_list', []))
        output_list.append("<|src_id|>{}<|tgt_id|>{}<|merge|>{}<|cell_list|>{}".format(
            one_res['src'], one_res['tgt'], one_res.get('merge', 0), cell_str))
    output_txt = '\n'.join(output_list)

    item = {
        "image": img_path,
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nTable Merge Detection: {}".format(input_txt)
            },
            {
                "from": "gpt",
                "value": output_txt
            }
        ],
        "type": "grounding"
    }
    return item

# ============================================================
# Add linking functions
# ============================================================

def add_linkings_contd(doc_label, blocks, contd_cases, image_dir):
    """Text Truncation """
    cnt = 0
    potential_idx = []
    judge_blocks = []
    res = []
    valid_pairs = {}

    # Filtering blocks and prepare input
    for i, block in enumerate(blocks):
        if block["type"] in ["text", "list_item"]: # Filtering by block type
            for pos in range(i+1, len(blocks)):
                if 'equation' in blocks[pos]['type'] or 'title' in blocks[pos]['type']:
                    break
                if blocks[pos]['type'] in ["text", "list_item"] and merge_rules(block["content"], blocks[pos]['content']): # Text truncation heuristics
                    if i not in potential_idx:
                        potential_idx.append(i)
                    if pos not in potential_idx:
                        potential_idx.append(pos)
                    valid_pairs[i] = pos
                    break
    
    # Construct input
    for i in potential_idx: 
        text = blocks[i]['content']
        bbox = blocks[i]['bbox']
        head = get_head_sentence(text)
        tail = get_tail_sentence(text)
        text_short = head if head == tail else head + ' ... ' + tail
        judge_blocks.append({'idx': i, 'content': text_short, 'page':blocks[i]['page'], 'bbox': [bbox[1]*1000, bbox[0]*1000, bbox[3]*1000, bbox[2]*1000]})
    pages = [block["page"] for block in judge_blocks]
    base64_image = concatenate_pdf_pages_with_border(doc_label, pages)
    
    # LLM Annotation
    if len(judge_blocks) > 0:
        prompt = txt_contd_prompt.replace("_RAW_LIST_", json.dumps(judge_blocks, ensure_ascii=False, indent=2))
        res = llm_generate_json(prompt, base64_image)
    
    if res == "!!LLM_ERROR!!":
        return blocks, res, res
    relations = []
    for item in res: # Add truncation judgement to blocks
        blocks[item["src"]]["content"] = blocks[item["src"]]["content"] + "<|txt_contd|>"
        relations.append(blocks[item["src"]]["content"] + blocks[item["tgt"]]["content"])
    
    # Format train and test cases
    img_path = os.path.join(image_dir, f"{os.path.basename(doc_label)[:-4]}_contd.jpeg")
    with open(img_path, 'wb') as f:
        f.write(base64.b64decode(base64_image))
    contd_cases.append(contd_format(judge_blocks, res, img_path))

    return blocks, relations, res


def add_linkings_title(doc_label, blocks, title_cases, image_dir):
    """Title Hierarchy"""
    judge_blocks = []
    res = []

    # Filtering by block type labels and prepare input
    for i, block in enumerate(blocks):
        bbox = blocks[i]['bbox']
        if block["type"] in ["title", "TOC-title", "section-title"]:
            judge_blocks.append({'idx': i, 'content': blocks[i]['content'], 'page':blocks[i]['page'], 'bbox': [bbox[1]*1000, bbox[0]*1000, bbox[3]*1000, bbox[2]*1000]})
    pages = [block["page"] for block in judge_blocks]
    base64_image = concatenate_pdf_pages_with_border(doc_label, pages)
    
    # LLM Annotation
    if len(judge_blocks) > 0:
        prompt = title_level_prompt.replace("_RAW_LIST_", json.dumps(judge_blocks, ensure_ascii=False, indent=2))
        res = llm_generate_json(prompt, base64_image)
    
    if res == "!!LLM_ERROR!!":
        return blocks, res, res
    block_level_map = {}
    relations = []
    for item in res: # Add title hierarchy to blocks
        if "level" not in item:
            continue 
        if item["level"] > 0:
            relations.append(item["level"]*'#' + blocks[item["idx"]]["content"])
            block_level_map[item["idx"]] = item["level"]
        else:
            relations.append(blocks[item["idx"]]["content"])
            blocks[item["idx"]]["type"] = "text"
    
    for i, block in enumerate(blocks):
        if i in block_level_map:
            block["level"] = block_level_map[i]
            link_src = 0
            for pos in range(i, 0, -1):
                if blocks[pos]["level"] > 0 and blocks[pos]["level"] < block["level"]:
                    link_src = blocks[pos]["id"]
                    break

            block["linking"] = [[link_src, block["id"]]]
        
        else:
            block["level"] = -1
            link_src = 0
            for pos in range(i, 0, -1):
                if blocks[pos]["level"] > 0:
                    link_src = blocks[pos]["id"]
                    break
            block["linking"] = [[link_src, block["id"]]]

    # Format train and test cases
    img_path = os.path.join(image_dir, f"{os.path.basename(doc_label)[:-4]}_title.jpeg")
    with open(img_path, 'wb') as f:
        f.write(base64.b64decode(base64_image))
    title_cases.append(title_format(judge_blocks, res, img_path))

    return blocks, relations, res

def check_overlap(visual_blocks, large_blocks): 
    """Construct the LLM input for text-image association"""
    judge_blocks = []
    large_block_linking = {}
    for i, block in visual_blocks.items():
        flag = True
        bbox = block["bbox"]
        for j, lblock in large_blocks.items():
            lbbox = lblock["bbox"]
            if (block["type"] in ['image', 'chart', 'image_footnote', 'image_caption', 'figure', 'fig-title', 'fig-caption']) and (bbox[0] >= 0.95*lbbox[0] and bbox[1] >= 0.95*lbbox[1] and bbox[2] <= 1.05*lbbox[2] and bbox[3] <= 1.05*lbbox[3]):
                large_block_linking[i] = j # nested images and captions are connected to the large frame by default
                flag = False
                break
        if flag: # Normalize block type label
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
            judge_blocks.append({'idx': i, 'type': typ, 'content': content, 'page':block['page'], 'bbox': [bbox[1]*1000, bbox[0]*1000, bbox[3]*1000, bbox[2]*1000]})
    return judge_blocks, large_block_linking

def add_linkings_image(doc_label, blocks, image_cases, image_dir):
    """Text-Image Association"""
    visual_blocks = {}
    large_blocks = {}
    res = []

    # Filtering by block type labels and prepare input
    for i, block in enumerate(blocks):
        if block["type"] in ['image_block', 'image', 'table', 'chart', 'table_footnote', 'image_footnote', 'table_caption', 'image_caption', "title", "TOC-title", "section-title", 'figure', 'fig-title', 'fig-caption', 'tab-title', 'tab-caption']:
            visual_blocks[i] = block
        if block["type"] in ['image_block']:
            large_blocks[i] = block
    
    judge_blocks, exist_linking = check_overlap(visual_blocks, large_blocks)
    pages = [block["page"] for block in judge_blocks]
    base64_image = concatenate_pdf_pages_with_border(doc_label, pages)
    
    # LLM Annotation
    if len(judge_blocks) > 0:
        prompt = image_link_prompt.replace("_RAW_LIST_", json.dumps(judge_blocks, ensure_ascii=False, indent=2))
        res = llm_generate_json(prompt, base64_image)
    
    if res == "!!LLM_ERROR!!":
        return blocks, res, res
    relations = []
    for item in res:
        exist_linking[item['src']] = item['tgt']

    for src, tgt in exist_linking.items(): # Add associations to blocks
        link_src = blocks[tgt]["id"]
        block = blocks[src]
        block["linking"] = [[link_src, block["id"]]]
        relations.append({'linking':[link_src, block["id"]], 'type':[blocks[tgt]['type'], block['type']]})

    # Format train and test cases
    img_path = os.path.join(image_dir, f"{os.path.basename(doc_label)[:-4]}_image.jpeg")
    with open(img_path, 'wb') as f:
        f.write(base64.b64decode(base64_image))
    image_cases.append(image_format(judge_blocks, res, img_path))

    return blocks, relations, res


def add_linkings_table_merge(doc_label, blocks, table_merge_cases, image_dir):
    """Cross-Page Table Merge Detection with pre-LLM screening.

    Flow:
      1. Group table blocks by page
      2. For each adjacent-page pair (last table on page N, first on page N+1):
         a. Apply filter_table_merge_candidates() — 6 heuristic checks
         b. If passed → call LLM to determine merge + per-column cell merge flags
         c. Annotate blocks with table_merge field
         d. Generate training case
    """
    # Initialize all table blocks with no merge
    for block in blocks:
        if block["type"] == "table":
            block["table_merge"] = -1

    # Group table block indices by page
    tables_by_page = {}
    for i, block in enumerate(blocks):
        if block["type"] == "table":
            page = block["page"]
            if page not in tables_by_page:
                tables_by_page[page] = []
            tables_by_page[page].append(i)

    if len(tables_by_page) < 2:
        return blocks, [], []

    sorted_pages = sorted(tables_by_page.keys())
    merge_relations = []

    for p_idx in range(len(sorted_pages) - 1):
        page1 = sorted_pages[p_idx]
        page2 = sorted_pages[p_idx + 1]
        if page2 != page1 + 1:
            continue

        table1_idx = tables_by_page[page1][-1]
        table2_idx = tables_by_page[page2][0]

        # --- Pre-LLM screening ---
        can_merge, reason = filter_table_merge_candidates(blocks, table1_idx, table2_idx)
        if not can_merge:
            print("  [SKIP] table_merge screened out: {}".format(reason))
            continue

        table1 = blocks[table1_idx]
        table2 = blocks[table2_idx]

        # Parse HTML tables
        try:
            soup1 = BeautifulSoup(table1.get("content", ""), "html.parser")
            soup2 = BeautifulSoup(table2.get("content", ""), "html.parser")
        except Exception:
            continue

        if not soup1.find_all("tr") or not soup2.find_all("tr"):
            continue

        # Collect captions
        caption1 = _get_caption_for_block(blocks, table1_idx)
        caption2 = _get_caption_for_block(blocks, table2_idx)

        # Detect header rows and get row data
        header_rows, _, _ = detect_table_headers(soup1, soup2)
        col_count1 = calculate_table_total_columns(soup1)
        col_count2 = calculate_table_total_columns(soup2)
        last_rows = get_visual_last_row_cells_content_with_span_info(soup1)
        first_rows = get_table_first_data_row_cells_with_span_info(soup2, header_rows)

        # Build judge blocks for LLM input
        judge_blocks = [{
            "idx": table1["id"],
            "page": table1["page"],
            "col_count": col_count1,
            "caption": caption1,
            "last_rows": last_rows,
            "first_rows": []
        }, {
            "idx": table2["id"],
            "page": table2["page"],
            "col_count": col_count2,
            "caption": caption2,
            "last_rows": [],
            "first_rows": first_rows
        }]

        pages = [table1["page"], table2["page"]]
        base64_image = concatenate_pdf_pages_with_border(doc_label, pages)

        # LLM Annotation
        res = []
        if len(judge_blocks) > 0:
            prompt = table_merge_prompt.replace("_RAW_LIST_", json.dumps(judge_blocks, ensure_ascii=False, indent=2))
            res = llm_generate_json(prompt, base64_image)

        if res == "!!LLM_ERROR!!":
            continue

        # Parse LLM result and annotate blocks
        if res and isinstance(res, list) and len(res) > 0:
            result = res[0]
            cell_list = result.get("judgement", [])
            if isinstance(cell_list, list) and len(cell_list) > 0:
                table1["table_merge"] = table2["id"]
                table2["table_merge"] = table1["id"]
                merge_relations.append({
                    "src": table1["id"],
                    "tgt": table2["id"],
                    "cell_list": cell_list
                })
            else:
                merge_relations.append({
                    "src": table1["id"],
                    "tgt": table2["id"],
                    "cell_list": []
                })
        # Format train and test cases
        img_path = os.path.join(image_dir, "{}{}".format(os.path.basename(doc_label)[:-4], "_table_merge.jpeg"))
        with open(img_path, 'wb') as f:
            f.write(base64.b64decode(base64_image))
        table_merge_cases.append(table_merge_format(judge_blocks, res, img_path))

    return blocks, merge_relations, res


def _get_caption_for_block(blocks, table_idx):
    """Find the caption text associated with a table block."""
    for b in blocks:
        if b.get("type") in ("table_caption", "tab-title", "tab-caption"):
            if b.get("page") == blocks[table_idx].get("page"):
                return b.get("content", "")
    return ""


def add_linkings(doc_label, blocks, contd_cases, title_cases, image_cases, table_merge_cases, image_dir): 
    """Annotation for each subtask"""
    blocks, relations_contd, raw_output_contd = add_linkings_contd(doc_label, blocks, contd_cases, image_dir)
    blocks, relations_title, raw_output_title = add_linkings_title(doc_label, blocks, title_cases, image_dir)
    blocks, relations_image, raw_output_image = add_linkings_image(doc_label, blocks, image_cases, image_dir)
    blocks, relations_table_merge, raw_output_table_merge = add_linkings_table_merge(doc_label, blocks, table_merge_cases, image_dir)
    relations = {
        'raw_contd':raw_output_contd,
        'contd':relations_contd,
        'raw_title':raw_output_title,
        'title':relations_title,
        'raw_image':raw_output_image,
        'image':relations_image,
        'raw_table_merge':raw_output_table_merge,
        'table_merge':relations_table_merge
    }
    return blocks, relations


def main(doc_label, ocr_res, contd_cases, title_cases, image_cases, table_merge_cases, image_dir, block_output_dir, relation_output_dir):
    """Collect document blocks for annotation"""
    block_output_file = os.path.join(block_output_dir, f"{os.path.splitext(os.path.basename(doc_label))[0]}.json")
    relation_output_file = os.path.join(relation_output_dir, f"{os.path.splitext(os.path.basename(doc_label))[0]}.json")

    item = ocr_res
    doc_blocks = []

    idx = 1
    for page_num, blocks in item.items():
        for block in blocks:
            block['page'] = int(page_num)
            block['id'] = idx
            idx += 1
            doc_blocks.append(block)
    
    if doc_blocks:
        res_blocks, res = add_linkings(doc_label, doc_blocks, contd_cases, title_cases, image_cases, table_merge_cases, image_dir)
        
    with open(block_output_file, 'w', encoding='utf-8') as f:
        json.dump(res_blocks, f, ensure_ascii=False, indent=4)
    with open(relation_output_file, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_file = ""
    block_output_dir = "" # Store detailed document blocks after annotation
    relation_output_dir = "" # Show intuitive annotated relationships
    image_dir = "" # Multi-page images for train and evaluation
    contd_file = "" # Text truncation train and evaluation cases
    title_file = "" # Title hierarchy train and evaluation cases
    image_file = "" # Image-text association train and evaluation cases
    table_merge_file = "" # Table merge train and evaluation cases
    with open(input_file, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    os.makedirs(block_output_dir, exist_ok=True)
    os.makedirs(relation_output_dir, exist_ok=True)
    
    contd_cases = []
    title_cases = []
    image_cases = []
    table_merge_cases = []

    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_label = {executor.submit(main, label, data[label], contd_cases, title_cases, image_cases, table_merge_cases, image_dir, block_output_dir, relation_output_dir): label for label in data.keys()}
        
        for future in tqdm(as_completed(future_to_label), total=len(data)):
            label = future_to_label[future]
            try:
                result = future.result()
            except Exception as e:
                traceback.print_exc()
