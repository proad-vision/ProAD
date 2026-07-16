import os
import json
import shutil
import argparse

def process_realiad_dataset(categories, root_path, json_root_path):
    print(f"Current working directory (Source & Dest): {os.path.abspath(root_path)}")
    print(f"JSON directory: {os.path.abspath(json_root_path)}\n")

    total_train_good = 0
    total_test_good = 0
    total_test_bad = 0
    total_masks = 0

    for category in categories:
        print(f"--- Processing category: {category} ---")
        
        json_file_path = os.path.join(json_root_path, f"{category}.json")
        category_root = os.path.join(root_path, category)
        
        dest_train_good = os.path.join(category_root, "train", "good")
        dest_test_good = os.path.join(category_root, "test", "good")
        dest_test_bad = os.path.join(category_root, "test", "bad")
        dest_gt_bad = os.path.join(category_root, "ground_truth", "bad")

        if not os.path.exists(json_file_path):
            print(f"Warning: JSON file for '{category}' not found, skipping.")
            continue
        
        os.makedirs(dest_train_good, exist_ok=True)
        os.makedirs(dest_test_good, exist_ok=True)
        os.makedirs(dest_test_bad, exist_ok=True)
        os.makedirs(dest_gt_bad, exist_ok=True)

        with open(json_file_path, 'r') as f:
            data = json.load(f)

        for item in data.get("train", []):
            if item.get("anomaly_class") == "OK":
                image_path = item.get("image_path")
                if image_path:
                    src_full = os.path.join(category_root, image_path)
                    dst_full = os.path.join(dest_train_good, os.path.basename(image_path))
                    
                    if os.path.exists(src_full):
                        shutil.copy2(src_full, dst_full)
                        total_train_good += 1
                    else:
                        if not os.path.exists(dst_full):
                             print(f"[Train] Source file missing: {src_full}")

        for item in data.get("test", []):
            anomaly_class = item.get("anomaly_class")
            image_path = item.get("image_path")
            mask_path = item.get("mask_path")

            if image_path:
                src_full = os.path.join(category_root, image_path)
                
                if os.path.exists(src_full):
                    if anomaly_class == "OK":
                        dst_full = os.path.join(dest_test_good, os.path.basename(image_path))
                        shutil.copy2(src_full, dst_full)
                        total_test_good += 1
                    else:
                        dst_full = os.path.join(dest_test_bad, os.path.basename(image_path))
                        shutil.copy2(src_full, dst_full)
                        total_test_bad += 1

                        if mask_path:
                            src_mask = os.path.join(category_root, mask_path)
                            if os.path.exists(src_mask):
                                dst_mask = os.path.join(dest_gt_bad, os.path.basename(mask_path))
                                shutil.copy2(src_mask, dst_mask)
                                total_masks += 1
                            else:
                                if not os.path.exists(os.path.join(dest_gt_bad, os.path.basename(mask_path))):
                                    print(f"[Mask] Mask file missing: {src_mask}")

        folders_to_remove = ["OK", "NG"]
        
        print(f"Cleaning old data for '{category}'...")
        for folder_name in folders_to_remove:
            folder_path = os.path.join(category_root, folder_name)
            if os.path.exists(folder_path):
                try:
                    shutil.rmtree(folder_path)
                    print(f"Removed old folder: {folder_path}")
                except Exception as e:
                    print(f"Failed to remove {folder_path}: {e}")

    print("\n" + "="*30)
    print("Processing complete! Statistics:")
    print(f"Train Good (OK): {total_train_good}")
    print(f"Test Good (OK):  {total_test_good}")
    print(f"Test Bad (NG):   {total_test_bad}")
    print(f"Masks:           {total_masks}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    
    args = parser.parse_args()

    categories = [
        "audiojack", "bottle_cap", "button_battery", "end_cap", "eraser",
        "fire_hood", "mint", "mounts", "pcb", "phone_battery", "plastic_nut",
        "plastic_plug", "porcelain_doll", "regulator", "rolled_strip_base",
        "sim_card_set", "switch", "tape", "terminalblock", "toothbrush", "toy",
        "toy_brick", "transistor1", "u_block", "usb", "usb_adaptor", "vcpill",
        "wooden_beads", "woodstick", "zipper"
    ]

    if not os.path.exists(args.data_path):
        print(f"Error: Path not found {os.path.abspath(args.data_path)}")
    else:
        process_realiad_dataset(categories, args.data_path, args.json_path)


#python ./prepare_data/prepare_Real-IAD.py --data_path ./Real-IAD --json_path ./Real-IAD/realiad_jsons