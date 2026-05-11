import re
import os
import json
from tqdm import tqdm

def split_subnode(json_file, pdf_path, output_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        tree = json.load(f)

    def traverse_tree(node):

        title = node.get('title', '')
        typ = node.get('type', '')
        content = node.get('content', '')
        children = node.get('children', [])
        location = node.get('location', [])
        ids = node.get('block_ids', [])

        if typ in ['table', 'chart', 'image', 'seal', 'image_block'] and len(children)>0:
            node['subnode'] = children
            node['children'] = []
            #print(1)
        
        if typ == 'text' and len(content)>500:
            segments = re.split(r'<\|txt_split\|\>|<\|txt_contd\|\>', content)
            segments = [s for s in segments if s]
            
            result_chunks = []
            chunk_segments = []
            
            current_chunk = ""
            current_segments = []
            current_length = 0
            segment_index = 1
            
            i = 0
            while i < len(segments):
                segment = segments[i]
                
                if segment in ['<|txt_split|>', '<|txt_contd|>']:
                    current_chunk += segment
                    i += 1
                    continue
                    
                current_chunk += segment
                current_length += len(segment)
                current_segments.append(segment_index)
                segment_index += 1
                
                if current_length > 500:
                    result_chunks.append(current_chunk)
                    chunk_segments.append(current_segments)
                    current_chunk = ""
                    current_segments = []
                    current_length = 0
                
                i += 1
            
            if current_chunk:
                result_chunks.append(current_chunk)
                chunk_segments.append(current_segments)

            node['subnode'] = []
            
            if len(result_chunks) > 1:
                for i, (sub_content, sub_index) in enumerate(zip(result_chunks, chunk_segments)):
                    try:
                        if title == "Default Title":
                            sub_index = [index-1 for index in sub_index]
                        elif 0 not in sub_index:
                            sub_index = [0] + sub_index
                        node['subnode'].append({
                            "type": "sub_text",
                            "title": f"{title}_{i+1}",
                            "metadata": "",
                            "content": sub_content,
                            "level": 1,
                            "location": [location[x] for x in sub_index],
                            "block_ids": [ids[x] for x in sub_index],
                            "children": []
                        })
                    except Exception as e:
                        print(json_file)
                        print("error_index")
                        print(sub_index)
                        print("error_idx")
                        print(ids)


        else:
            node['subnode'] = []

        for child in children:
            traverse_tree(child)
    
    traverse_tree(tree)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tree, f, ensure_ascii=False, indent=4)

pdf_dir = '' # The original documents
tem_tree_dir = '' # The tree after summary generation
com_tree_dir = '' # The tree after subnode chunking (final result)
os.makedirs(com_tree_dir, exist_ok=True)
for json_file in tqdm(os.listdir(tem_tree_dir)):
    output_file = os.path.join(com_tree_dir, json_file)
    pdf_path = os.path.join(pdf_dir, json_file[:-4]+'pdf')
    json_file = os.path.join(tem_tree_dir, json_file)
    split_subnode(json_file, pdf_path, output_file)