import re
import os
import asyncio
import json
from functools import partial
from tqdm import tqdm
from model_utils import qwen_generate, bbox_to_base64

image_enrichment_prompt = """
### Instruction
Extract or conclude the following attributes of the image as required. Return them in the output format.

### Required Attributes
  [T] title (A short title for the image)
  [M] metadata (A few keywords as the image's metadata)
  [C] content (Describe the image content comprehensively and detailedly)

### Output Format Examples
[T] Group Photo of Participants 
[M] Group photo of ten participants; red carpet and blue background 
[C] The photo shows ten men in suits standing on the red carpet, with "the 10th Cooperation Conference" on the blue background, and ...

### Note
You only need to output the three required attributes with fixed and capitalized marks [T], [M], [C], do not miss any of them!
"""

chart_enrichment_prompt = """
### Instruction
Extract or conclude the following attributes of the chart as required. Return them in the output format.

### Required Attributes
  [T] title (A short title for the chart)
  [M] metadata (A few keywords as the chart's metadata)
  [C] content (Describe the chart content comprehensively and detailedly)

### Output Format Examples
[T] Annual Participant Trend Line Chart 
[M] A line chart of the number of participants over the years; horizontal axis by year and vertical axis by number 
[C] The line chart shows the growth trend of the participant numbers over years. The horizontal axis is the year and the vertical axis is the participant number. In 2022 there are 5 participant, in 2023 there are 7 participant, and ...

### Note
You only need to output the three required attributes with fixed and capitalized marks [T], [M], [C], do not miss any of them!
"""

table_enrichment_prompt = """
### Instruction
Extract or conclude the following attributes of the table as required. Return them in the output format.

### Required Attributes
  [T] title (A short title for the table)
  [M] metadata (A few keywords as the table's metadata)
  [C] content (Describe the table content comprehensively and detailedly)

### Output Format Examples
[T] Meeting Schedule Table 
[M] A table about the meeting schedule; three columns about activity, time and location. 
[C] Per the schedule outlined in the table, breakfast will be served at 8:00 on the 2nd floor, followed by the first meeting at 9:00 in Room 404, and then...

### Note
You only need to output the three required attributes with fixed and capitalized marks [T], [M], [C], do not miss any of them!
"""

metadata_generation_prompt = """
### Instruction
I now have some data in text. Please generate a few nominal phrases as the keywords to comprehensively summarize the data, separated by semicolon(;).

### Data
{data}

### Note
You only need to output specified number of nominal phrases as keywords. Do not give any extra explanations.
"""

def qwen_annotation(base64_image, cp_type):

    prompt_map = {
        "table": table_enrichment_prompt,
        "chart": chart_enrichment_prompt, 
        "image": image_enrichment_prompt
    }
    prompt = prompt_map.get(cp_type, image_enrichment_prompt)

    flag = False
    cnt = 0
    pattern = r'\[T\](.*?)\[M\](.*?)\[C\](.*)'
    title = "Default Title"
    metadata = "Default Metadata"
    content = "Default Content"
    
    while cnt<3:
        text = qwen_generate(prompt,base64_image)
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            title = match.group(1).strip()
            metadata = match.group(2).strip()
            content = match.group(3).strip()
            break
        else:
            cnt = cnt + 1

    return title, metadata, content

async def generate_metadata_async(json_file, pdf_path, output_file):
    loop = asyncio.get_event_loop()

    def read_json():
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    tree = await loop.run_in_executor(None, read_json)

    async def process_node(node):
        child_tasks = []
        for child in node.get('children', []):
            child_tasks.append(asyncio.create_task(process_node(child)))
        for subnode in node.get('subnode', []):
            child_tasks.append(asyncio.create_task(process_node(subnode)))

        current_task = asyncio.create_task(process_current_node(node, pdf_path))
        all_tasks = [current_task] + child_tasks
        await asyncio.gather(*all_tasks, return_exceptions=True)

    async def process_current_node(node, pdf_path):
        typ = node.get('type', '')
        title = node.get('title', '')
        meta = node.get('metadata', '')
        content = node.get('content', '')
        location = node.get('location', [])
        
        
        if typ in ['table', 'chart', 'image', 'seal', 'image_block']:
            base64_image = await asyncio.to_thread(bbox_to_base64, pdf_path, location)
            gen_title, gen_metadata, gen_content = await asyncio.to_thread(
                qwen_annotation, base64_image, typ
            )
            if not title:
                node['title'] = gen_title
            if not meta:
                node['metadata'] = gen_metadata
            if not content:
                node['content'] = gen_content
        
        elif typ in ['text', 'sub_text'] and content:
            prompt = metadata_generation_prompt.format(data=content)
            result = await asyncio.to_thread(qwen_generate, prompt, None)
            gen_metadata = result
            if not meta:
                node['metadata'] = gen_metadata
        
    await process_node(tree)

    def write_json():
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tree, f, ensure_ascii=False, indent=4)
    await loop.run_in_executor(None, write_json)

def generate_metadata(json_file, pdf_path, output_file):
    asyncio.run(generate_metadata_async(json_file, pdf_path, output_file))

def main(pdf_dir, tem_tree_dir, com_tree_dir):
    os.makedirs(com_tree_dir, exist_ok=True)
    for json_file in tqdm(os.listdir(tem_tree_dir)):
        print(json_file)
        output_file = os.path.join(com_tree_dir, json_file)
        pdf_path = os.path.join(pdf_dir, json_file[:-4]+'pdf')
        json_file = os.path.join(tem_tree_dir, json_file)
        generate_metadata(json_file, pdf_path, output_file)


pdf_dir = '' # The original documents
tem_tree_dir = '' # The primary tree after construction
com_tree_dir = '' # The tree after summary generation
main(pdf_dir, tem_tree_dir, com_tree_dir)