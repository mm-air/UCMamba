import os
import shutil
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import datetime

import json
import cv2


def prepare_kvasir(source_root, target_root, split_ratios=(0.8, 0.1, 0.1), seed=42):
    """
    Splits Kvasir-SEG into train/val/test and converts masks to class indices (0, 1).
    """
    # 1. Setup paths
    source_images = os.path.join(source_root, 'images')
    source_masks = os.path.join(source_root, 'masks')

    # Define output directories matching your config
    subsets = ['train', 'val', 'test']
    dirs = {
        'images': {s: os.path.join(target_root, 'images', s) for s in subsets},
        'masks': {s: os.path.join(target_root, 'masks', s) for s in subsets}
    }

    # Create directories
    for cat in dirs:
        for s in subsets:
            os.makedirs(dirs[cat][s], exist_ok=True)

    # 2. Get file lists
    # Kvasir-SEG images are usually jpg, masks are jpg or png.
    # We match them by filename (minus extension).
    all_files = [f for f in os.listdir(source_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    all_files.sort()  # Ensure consistent order before shuffling

    # Check for matching masks
    valid_pairs = []
    for img_file in all_files:
        basename = os.path.splitext(img_file)[0]
        # Kvasir masks usually have same basename
        # Try finding the mask with common extensions
        mask_found = False
        for ext in ['.jpg', '.jpeg', '.png']:
            mask_name = basename + ext
            if os.path.exists(os.path.join(source_masks, mask_name)):
                valid_pairs.append((img_file, mask_name))
                mask_found = True
                break

        if not mask_found:
            print(f"Warning: Mask not found for {img_file}")

    print(f"Found {len(valid_pairs)} valid image-mask pairs.")

    # 3. Shuffle and Split
    random.seed(seed)
    random.shuffle(valid_pairs)

    total = len(valid_pairs)
    n_train = int(total * split_ratios[0])
    n_val = int(total * split_ratios[1])
    # Remaining goes to test to avoid rounding errors

    train_set = valid_pairs[:n_train]
    val_set = valid_pairs[n_train:n_train + n_val]
    test_set = valid_pairs[n_train + n_val:]

    sets = {
        'train': train_set,
        'val': val_set,
        'test': test_set
    }

    print(f"Split counts - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    # 4. Process and Copy
    for subset, pairs in sets.items():
        print(f"Processing {subset} set...")
        for img_name, mask_name in tqdm(pairs):
            # --- Process Image ---
            src_img_path = os.path.join(source_images, img_name)
            dst_img_path = os.path.join(dirs['images'][subset], img_name)
            shutil.copy2(src_img_path, dst_img_path)

            # --- Process Mask ---
            # Open mask, convert to index map (0=bg, 1=polyp)
            src_mask_path = os.path.join(source_masks, mask_name)

            # Use PIL to read
            mask = Image.open(src_mask_path).convert('L')  # Convert to grayscale
            mask_np = np.array(mask)

            # Thresholding: Kvasir masks are not always binary 0/1.
            # They are often 0/255 or soft boundaries.
            # We treat anything > 127 as polyp (class 1), else background (class 0).
            mask_index = np.zeros_like(mask_np, dtype=np.uint8)
            mask_index[mask_np > 127] = 1

            # Save as PNG (lossless) to preserve class indices
            # Changing extension to .png is recommended for masks
            dst_mask_name = os.path.splitext(mask_name)[0] + '.png'
            dst_mask_path = os.path.join(dirs['masks'][subset], dst_mask_name)

            Image.fromarray(mask_index).save(dst_mask_path)

    print("\nProcessing complete! dataset is ready in:", target_root)

def images_to_coco(img_dir, mask_dir, out_file):
    images = []
    annotations = []
    obj_count = 0

    # Define standard COCO info block
    info = dict(
        year=datetime.datetime.now().year,
        version="1.0",
        description="Kvasir-SEG Dataset (Polyp Detection)",
        contributor="Simula Research Laboratory / User Processed",
        url="https://datasets.simula.no/kvasir-seg/",
        date_created=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    file_names = sorted(os.listdir(img_dir))

    print(f"Generating COCO annotations for {img_dir}...")

    for idx, filename in enumerate(file_names):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # --- Image Info ---
        img_path = os.path.join(img_dir, filename)
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error reading image {filename}: {e}")
            continue

        images.append(dict(
            id=idx,
            file_name=filename,
            height=height,
            width=width
        ))

        # --- Mask / Annotation Info ---
        basename = os.path.splitext(filename)[0]
        mask_name = basename + '.png'  # Assuming .png from your preprocess step
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.exists(mask_path):
            continue

        # Read mask using OpenCV to ensure binary format for finding contours
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if contour.size < 6:  # Skip artifacts/noise (need at least 3 points)
                continue

            flattened = contour.flatten().tolist()
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            annotations.append(dict(
                id=obj_count,
                image_id=idx,
                category_id=1,  # 1 = polyp
                segmentation=[flattened],
                area=area,
                bbox=[x, y, w, h],
                iscrowd=0
            ))
            obj_count += 1

    coco_format = dict(
        info=info,  # <--- 'info' section added here
        images=images,
        annotations=annotations,
        categories=[dict(id=1, name='polyp')]
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(out_file, 'w') as f:
        json.dump(coco_format, f)
    print(f"Saved {out_file} with {len(images)} images and {len(annotations)} annotations.")
# --- Configuration ---
# CHANGE THESE PATHS
SOURCE_PATH = '/home/shirley/Desktop/Projects/Dataset/kvasir_seg_raw/Kvasir-SEG'  # Folder containing 'images' and 'masks' subfolders
TARGET_PATH = '/home/shirley/Desktop/Projects/Dataset/kvasir_seg_raw/data/kvasir_seg'

if __name__ == '__main__':
    #if not os.path.exists(SOURCE_PATH):
    #    print(f"Error: Source path '{SOURCE_PATH}' does not exist.")
    #else:
    #    prepare_kvasir(SOURCE_PATH, TARGET_PATH)
    # CALL THIS FUNCTION AT THE END OF YOUR SCRIPT
    # After the split loops:
    dirs = TARGET_PATH
    target_root = TARGET_PATH
    images_to_coco(dirs+'/images/train', dirs+'/masks/train', os.path.join(target_root, 'train.json'))
    images_to_coco(dirs+'/images/val', dirs+'/masks/val', os.path.join(target_root, 'val.json'))
    images_to_coco(dirs+'/images/test', dirs+'/masks/test', os.path.join(target_root, 'test.json'))
