import json
import numpy as np
from PIL import Image
import os
import cv2


def labelme_json_to_mask(json_file, output_mask_path):
    # 讀取 Labelme JSON 文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 通過 labelme Shape 轉換為 Segmentation Mask
    labelme_shapes = data['shapes']
    img_shape = data['imageHeight'], data['imageWidth']

    # 初始化 Mask
    mask = np.zeros(img_shape, dtype=np.uint8)

    # 設置標記顏色為白色 (灰度值 255)
    white_color = 255

    for shape in labelme_shapes:
        points = shape['points']
        mask_points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [mask_points], white_color)

    # 保存 Mask 為圖像
    mask_img = Image.fromarray(mask)
    mask_img.save(output_mask_path)

    print(f'Mask saved to: {output_mask_path}')


def process_folder(input_folder, output_folder):
    # 創建輸出資料夾（如果不存在的話）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍歷資料夾中的所有文件
    for filename in os.listdir(input_folder):
        json_file = os.path.join(input_folder, filename)

        # 檢查是否為 .json 文件
        if os.path.isfile(json_file) and filename.endswith('.json'):
            output_mask_path = os.path.join(
                output_folder, filename.replace('.json', '.png'))
            print(f"Processing {json_file}...")  # 這行會幫助你檢查路徑
            labelme_json_to_mask(json_file, output_mask_path)


# 使用示例
if __name__ =="__main__":
    input_folder = r'val_json'
    output_folder = r'masks'
    process_folder(input_folder, output_folder)
