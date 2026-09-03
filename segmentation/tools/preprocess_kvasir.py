import os
import shutil
import random
import numpy as np
from PIL import Image
from tqdm import tqdm


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


# --- Configuration ---
# CHANGE THESE PATHS
SOURCE_PATH = '/home/shirley/Desktop/Projects/Dataset/kvasir_seg_raw/Kvasir-SEG'  # Folder containing 'images' and 'masks' subfolders
TARGET_PATH = '/home/shirley/Desktop/Projects/Dataset/kvasir_seg_raw/data/kvasir_seg'

if __name__ == '__main__':
    if not os.path.exists(SOURCE_PATH):
        print(f"Error: Source path '{SOURCE_PATH}' does not exist.")
    else:
        prepare_kvasir(SOURCE_PATH, TARGET_PATH)
