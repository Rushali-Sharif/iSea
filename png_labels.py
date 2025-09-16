import os
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image

# === Configuration ===
binary_masks_folder = r"E:/iSea/valid/masks"   # <-- Replace with your absolute path
output_instance_folder = r"E:/iSea/valid/labels"
output_debug_folder = r"E:/iSea/debug/valid_debug"
empty_masks_log = r"E:/iSea/debug/empty_valid_mask_images.txt"  # File to save image names with only empty masks

os.makedirs(output_instance_folder, exist_ok=True)
os.makedirs(output_debug_folder, exist_ok=True)

def read_mask_strict(path):
    """
    Attempts to read a grayscale mask image robustly.
    Returns None if file is empty or unreadable instead of raising.
    """
    if not os.path.exists(path):
        print(f"WARNING: File does not exist: {path}")
        return None
    if os.path.getsize(path) == 0:
        # Empty file
        return None
    
    # Try OpenCV
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img
    
    # OpenCV failed, try PIL fallback
    try:
        pil_img = Image.open(path).convert("L")
        img = np.array(pil_img)
        return img
    except Exception as e:
        print(f"WARNING: Failed to read image {path} with OpenCV and PIL. Error: {e}")
        return None

# List all mask files
mask_files = [f for f in os.listdir(binary_masks_folder) if f.lower().endswith((".jpg", ".png"))]
if not mask_files:
    raise RuntimeError(f"No mask files found in {binary_masks_folder}")

# Group masks by image base name (e.g., coral_1001)
grouped_masks = defaultdict(list)
for fname in mask_files:
    if "_mask_" not in fname:
        print(f"WARNING: Mask filename does not contain '_mask_': {fname}")
        continue
    base = fname.rsplit("_mask_", 1)[0]
    grouped_masks[base].append(fname)

print(f"Found {len(grouped_masks)} image groups with masks.")

empty_only_images = []

# Process each image group
for base_name, masks in grouped_masks.items():
    masks.sort()

    # Read all masks, keep only valid (non-empty)
    valid_masks = []
    for mask_file in masks:
        mask_path = os.path.join(binary_masks_folder, mask_file)
        mask_img = read_mask_strict(mask_path)
        if mask_img is None:
            print(f"Empty or unreadable mask skipped: {mask_file}")
            continue
        valid_masks.append((mask_file, mask_img))

    if len(valid_masks) == 0:
        # All masks empty for this image
        print(f"All masks empty for image: {base_name}")
        empty_only_images.append(base_name)
        continue

    # Use shape of first valid mask
    height, width = valid_masks[0][1].shape
    instance_mask = np.zeros((height, width), dtype=np.uint8)

    # Merge valid masks with instance IDs
    for idx, (mask_file, mask_img) in enumerate(valid_masks):
        if mask_img.shape != (height, width):
            raise ValueError(f"Mask size mismatch: {mask_file} shape {mask_img.shape} != first mask shape {(height, width)}")

        _, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

        instance_id = idx + 1
        overlap_pixels = np.sum((instance_mask > 0) & (binary_mask == 255))
        if overlap_pixels > 0:
            print(f"WARNING: Overlapping pixels detected in {mask_file} for image {base_name}")

        instance_mask[binary_mask == 255] = instance_id

    # Save merged instance mask
    out_path = os.path.join(output_instance_folder, f"{base_name}.png")
    cv2.imwrite(out_path, instance_mask)

    # Save brightened debug mask
    max_id = instance_mask.max()
    scale_factor = 255 / max_id if max_id > 0 else 0
    debug_mask = (instance_mask * scale_factor).clip(0, 255).astype(np.uint8)

    debug_out_path = os.path.join(output_debug_folder, f"{base_name}.png")
    cv2.imwrite(debug_out_path, debug_mask)

    print(f"[OK] Merged {len(valid_masks)} masks into: {out_path}")
    print(f"[OK] Saved brightened debug mask to: {debug_out_path}")

# Write images with only empty masks to log file
if empty_only_images:
    with open(empty_masks_log, "w") as f:
        for img_name in empty_only_images:
            f.write(img_name + "\n")
    print(f"\nSaved list of images with only empty masks to: {empty_masks_log}")
else:
    print("\nNo images found with only empty masks.")

print("\nAll masks processed successfully.")
