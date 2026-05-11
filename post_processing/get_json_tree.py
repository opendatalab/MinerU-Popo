import os
import json
from table_merge_utils import merge_cross_page_tables


special_types = ['table_footnote', 'table', 'chart', 'table_caption', 'image_footnote', 'image', 'image_caption', 'seal']
large_block_types = ['super', 'list', 'ref_block', 'equation_block', 'image_block']
supplement_types = ['page_title', 'page_number', 'page_footnote', 'header', 'aside_text', 'footer']

def cp_init(cp_type="",title="",metadata="",content="",level=-1,location=None,block_ids=None):
    # Create a component
    cp = {
        'type': cp_type,
        'title': title,
        'metadata': metadata,
        'content': content,
        'level': level,
        'location': [] if not location else location,
        'block_ids': [] if not block_ids else block_ids
    }
    return cp


def construct_json_tree(input_file, output_dir, txt_dir):
    with open(input_file, 'r', encoding='utf-8') as f:
        elements = json.load(f)
    elements = merge_cross_page_tables(elements)
    
    def get_text_components(elements):
        text_components = []
        contd_list = []
        cur_text_title = "Default Title"
        cur_text_cp = cp_init(cp_type="text", title = cur_text_title, level=1)

        for element in elements:
            if element["type"] == "title" and element["level"] < 0:
                element["type"] = "text"

            if element["type"] == "title":
                cur_text_title = element['content']
                if cur_text_cp['title'] != "Default Title" or cur_text_cp['content'] != "":
                    text_components.append(cur_text_cp)
                cur_text_cp = cp_init(cp_type="text", title = cur_text_title, level = element['level'], location = [{'bbox':element['bbox'], 'page':element['page']}], block_ids = [element['id']])

            elif element["type"] not in special_types + large_block_types + supplement_types:
                if element['contd'] >= 0:
                    contd_list.append(element['contd'])
                contd_label = '<|txt_contd|>' if element['id'] in contd_list else '<|txt_split|>'
                cur_text_cp['content'] = cur_text_cp['content'] + contd_label + element['content'] if cur_text_cp['content'] else element['content']
                cur_text_cp['location'].append({'bbox':element['bbox'], 'page':element['page']})
                cur_text_cp['block_ids'].append(element['id'])

        text_components.append(cur_text_cp)

        return text_components
    
    def construct_by_level(text_components):
        # Initialize the root and stack
        root = cp_init(cp_type="root", level=0)
        root['children'] = []
        stack = [{'node': root, 'level': 0}]
        
        # Traverse in reading order
        for cp in text_components:

            cp['children'] = []
            level = cp['level'] if cp['level']> 0 else 100
            
            # Pop until a higher level (small number)
            while stack[-1]["level"] >= level:
                stack.pop()
            parent = stack[-1]["node"]
        
            # Link to the parent
            parent["children"].append(cp)
            
            # Push
            stack.append({"node": cp, "level": level})
        
        return root

    text_tree = construct_by_level(get_text_components(elements))
    def add_special_elements(text_tree, elements):
        visual_components = []
        for element in elements:
            if element['type'] in ['table', 'chart', 'image', 'seal', 'image_block']:
                locations = element.get('merged_locations', [{'bbox':element['bbox'], 'page':element['page']}])
                block_ids = element.get('merged_block_ids', [element['id']])
                visual_component = cp_init(cp_type=element['type'], content = element['content'], level = element['image'], location = locations, block_ids = block_ids)
                
                for elem in elements:
                    if elem['image'] == element['id']:
                        if 'caption' in elem['type']:
                            visual_component['title'] = visual_component['title'] + " " + elem['content'] if visual_component['title'] else elem['content']
                            visual_component['location'].append({'bbox':elem['bbox'], 'page':elem['page']})
                            visual_component['block_ids'].append(elem['id'])
                        elif 'footnote' in elem['type']:
                            visual_component['metadata'] = visual_component['metadata'] + " " + elem['content'] if visual_component['metadata']  else elem['content']
                            visual_component['location'].append({'bbox':elem['bbox'], 'page':elem['page']})
                            visual_component['block_ids'].append(elem['id'])
                visual_components.append(visual_component)

        for visual_component in visual_components:
            visual_component['children'] = []

        for visual_component in visual_components:
            for v_cp in visual_components:
                if visual_component['level'] in v_cp['block_ids']:
                    v_cp['children'].append(visual_component)
                    visual_components.remove(visual_component)
        
        def get_node_by_id(root, idx):
            if idx in root['block_ids']:
                return root
            for children in root['children']:
                check_child = get_node_by_id(children, idx)
                if check_child != None:
                    return check_child
            return None

        def find_former_title(elements, idx):
            former_title = 0
            for element in elements:
                if element['type'] == 'title' and element['id'] < idx and element['id'] > former_title:
                    former_title = element['id']
            return former_title


        for visual_component in visual_components:
            
            idx = visual_component['level'] if visual_component['level'] >= 0 else find_former_title(elements, min(visual_component['block_ids']))
            tree_node = get_node_by_id(text_tree, idx)
            if tree_node:
                tree_node['children'].append(visual_component)

        return text_tree

    add_special_elements(text_tree, elements)


    def add_supplement(text_tree, elements):
        exist = []
        for element in elements:
            if element['type'] in supplement_types:
                title = f"Page {element['page']} - {element['type']}"
                cnt = 0
                while title in exist:
                    cnt += 1
                    title = f"Page {element['page']} - {element['type']} - {cnt}"
                exist.append(title)
                supp_component = cp_init(cp_type=element['type'], title = title, metadata = element['content'], content = element['content'], location = [{'bbox':element['bbox'], 'page':element['page']}], block_ids = [element['id']])
                text_tree['children'].append(supp_component)

    add_supplement(text_tree, elements)

    with open(os.path.join(output_dir, os.path.basename(input_file)), 'w', encoding='utf-8') as f:
        json.dump(text_tree, f, ensure_ascii=False, indent=4)



    def traverse_tree(node, depth=0, lines_list=None):

        if lines_list is None:
            lines_list = []
        
        indent = ' ' * (depth * 4)
        title = node.get('title', 'N/A')
        data = node.get('content', '')
        
        if not data:
            data = ""
        data_preview = data[:30] + ('...' if len(data) > 30 else '')
        
        line = f"{indent}{title}|{data_preview}"
        lines_list.append(line)
        
        children = node.get('children', [])
        for child in children:
            traverse_tree(child, depth + 1, lines_list)
        
        return lines_list
    
    output_lines = []
    tree_lines = traverse_tree(text_tree)
    output_lines.extend(tree_lines)
    txt_file = os.path.join(txt_dir, os.path.basename(input_file)[:-4] + "txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + '\n')
    
input_dir = '' # The inference output dir
tree_dir = '' # Store the complete tree
txt_dir = '' # Show the tree overview
os.makedirs(tree_dir, exist_ok=True)
os.makedirs(txt_dir, exist_ok=True)
for file in os.listdir(input_dir):
    input_file = os.path.join(input_dir, file)
    construct_json_tree(input_file, tree_dir, txt_dir)
