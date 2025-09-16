import os
import cv2
import numpy as np

def convert_masks_to_yolo_labels(mask_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(mask_dir):
        if not fname.endswith(".png"):
            continue

        mask_path = os.path.join(mask_dir, fname)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[ERROR] Could not read: {fname}")
            continue

        h, w = mask.shape[:2]  # get actual size of the mask

        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids != 0]

        label_lines = []
        for inst_id in instance_ids:
            inst_mask = np.uint8(mask == inst_id)
            contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if len(cnt) < 3:
                    continue
                cnt = cnt.reshape(-1, 2).astype(np.float32)
                cnt[:, 0] /= w  # normalize x
                cnt[:, 1] /= h  # normalize y

                coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in cnt])
                label_lines.append(f"0 {coords}")  # class 0

        out_path = os.path.join(out_dir, fname.replace(".png", ".txt"))
        with open(out_path, "w") as f:
            f.write("\n".join(label_lines))

        print(f"[OK] Converted: {fname} → {os.path.basename(out_path)} ({len(label_lines)} instances)")

# === Run for train, val, test ===
base_path = "E:/iSea"

convert_masks_to_yolo_labels(f"{base_path}/train/labels", f"{base_path}/train/yolo")
convert_masks_to_yolo_labels(f"{base_path}/valid/labels", f"{base_path}/valid/yolo")
convert_masks_to_yolo_labels(f"{base_path}/test/labels", f"{base_path}/test/yolo")
