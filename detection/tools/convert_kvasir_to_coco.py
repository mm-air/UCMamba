import os
import json
import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split


def binary_mask_to_polygon(mask):
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        if contour.size >= 6:  # Filter small artifacts
            segmentation.append(contour.flatten().tolist())
    return segmentation


def process_dataset(image_paths, mask_dir, out_file):
    annotations = []
    images = []
    obj_count = 0

    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        # Add image info
        images.append({
            "id": idx,
            "file_name": filename,
            "height": height,
            "width": width
        })

        # Read corresponding mask
        mask_path = os.path.join(mask_dir, filename)
        if not os.path.exists(mask_path):
            continue

        mask = cv2.imread(mask_path, 0)  # Read as grayscale
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Get bounding box and segmentation
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            # Create polygon for segmentation
            poly = contour.flatten().tolist()
            if len(poly) < 6: continue

            annotations.append({
                "id": obj_count,
                "image_id": idx,
                "category_id": 1,  # 1 class: polyp
                "bbox": [x, y, w, h],
                "area": area,
                "segmentation": [poly],
                "iscrowd": 0
            })
            obj_count += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "polyp"}]
    }

    with open(out_file, 'w') as f:
        json.dump(coco_format, f)
    print(f"Saved {out_file} with {len(images)} images and {len(annotations)} objects.")


# --- Execution ---
ROOT_PATH = '/home/shirley/Desktop/Projects/Dataset/kvasir_seg_raw/Kvasir-SEG'
img_dir = os.path.join(ROOT_PATH,'images')
mask_dir = os.path.join(ROOT_PATH,'masks')
all_images = glob.glob(os.path.join(img_dir, '*.jpg'))

train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)

process_dataset(train_imgs, mask_dir, os.path.join(ROOT_PATH,'train.json'))
process_dataset(val_imgs, mask_dir, os.path.join(ROOT_PATH,'val.json'))

###############################################################Fix the problem of missing 'info' & 'License'
def fix_json_file(file_path):
    print(f"Fixing {file_path}...")

    with open(file_path, 'r') as f:
        data = json.load(f)

    # Add missing "info" field
    if 'info' not in data:
        data['info'] = {
            "description": "Kvasir-SEG COCO Format",
            "url": "",
            "version": "1.0",
            "year": 2024,
            "contributor": "User",
            "date_created": "2024-01-01"
        }
        print(" -> Added 'info' field")

    # Add missing "licenses" field
    if 'licenses' not in data:
        data['licenses'] = [
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"
            }
        ]
        print(" -> Added 'licenses' field")

    # Ensure images have license IDs
    if 'images' in data:
        for img in data['images']:
            if 'license' not in img:
                img['license'] = 1

    # Overwrite the file with fixed data
    with open(file_path, 'w') as f:
        json.dump(data, f)
    print("Done.\n")


# Run on your files
fix_json_file('/home/shirley/Desktop/Projects/Dataset/kvasir_seg_raw/Kvasir-SEG/train.json')
fix_json_file('/home/shirley/Desktop/Projects/Dataset/kvasir_seg_raw/Kvasir-SEG/val.json')
