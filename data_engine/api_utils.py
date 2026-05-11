import requests
import json
import os
from tqdm import tqdm
from openai import OpenAI

import fitz
from PIL import Image
import io
import base64


API_KEY= ""
API_URL= ""

def bbox_to_base64(pdf_path, bbox_list):

    pdf_document = fitz.open(pdf_path)
    images = []

    try:
        for bbox in bbox_list:
            page_num = bbox['page_id'] - 1
            crop_range = [bbox['block_bbox'][0]/2,bbox['block_bbox'][1]/2,bbox['block_bbox'][2]/2,bbox['block_bbox'][3]/2]
            if page_num < 0 or page_num >= len(pdf_document):
                raise ValueError(f"页面索引 {page_num} 超出范围, 共有 {len(pdf_document)} 页")

            page = pdf_document[page_num]
            pix = page.get_pixmap(clip=crop_range)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)

        if not images:
            raise ValueError("Empty Image!")

        total_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)
        merged_image = Image.new("RGB", (total_width, total_height))

        y_offset = 0
        for img in images:
            merged_image.paste(img, (0, y_offset))
            y_offset += img.height

        buffered = io.BytesIO()
        merged_image.save(buffered, format="JPEG")
        buffered.seek(0)
        base64_image = base64.b64encode(buffered.read()).decode('utf-8')

        return base64_image

    except Exception as e:
        empty_image = Image.new("RGB", (1, 1), color=(255, 255, 255))
        buffered = io.BytesIO()
        empty_image.save(buffered, format="JPEG")
        buffered.seek(0)
        
        # 转为 Base64 编码
        base64_empty_image = base64.b64encode(buffered.read()).decode('utf-8')
        
        return base64_empty_image

    finally:
        pdf_document.close()

def gpt_generate_pdf(base64_image, prompt, key=API_KEY, url=API_URL, base_model = "gemini-3-flash-preview"):
    client = OpenAI(
        base_url=url,
        api_key=key
    )
    res = ""
    cnt = 0

    while cnt < 3:
        try:
            response = client.chat.completions.create(
                model=base_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                temperature = 1
            )
            res = response.choices[0].message.content

            return res

        except Exception as e:
            cnt += 1
            print(e)

            base64_img = f"data:image/png;base64,{base64_image}"
            if ',' in base64_img:
                base64_img = base64_img.split(',')[1]

            decoded_data = base64.b64decode(base64_img)
            bytes_size = len(decoded_data)
            mb_size = bytes_size / (1024 * 1024)
            print(f"Image Size: {mb_size:.6f} MB")

            import time; time.sleep(0.1)


    return "[\"!!LLM_ERROR!!\"]"