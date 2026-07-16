import cv2
import numpy as np
import os
import random
import glob
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from PIL import Image
import argparse
import sys

DATASET_CONFIG = {
    "mvtec": [
        "bottle", "capsule", "hazelnut", "metal_nut", 
        "pill", "screw", "toothbrush", "zipper"
    ],
    "realiad": [
        "audiojack", "bottle_cap", "button_battery", "end_cap", "eraser", 
        "fire_hood", "mint", "mounts", "pcb", "phone_battery", "plastic_nut", 
        "plastic_plug", "porcelain_doll", "regulator", "rolled_strip_base", 
        "sim_card_set", "switch", "tape", "terminalblock", "toothbrush", "toy", 
        "toy_brick", "transistor1", "u_block", "usb", "usb_adaptor", "vcpill", 
        "wooden_beads", "woodstick"
    ],
    "visa": [
        "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", 
        "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
    ]
}

WEIGHTS_DIR = os.path.join(os.getcwd(), "weights", "BiRefNet")

def init_segmentation_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Initializing Segmentation Model ---")
    print(f"Using device: {device}")
    
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    model_id = 'ZhengPeng7/BiRefNet-HRSOD'
    
    try:
        print(f"Loading model: {model_id} (Cache: {WEIGHTS_DIR})...")
        model = AutoModelForImageSegmentation.from_pretrained(
            model_id, 
            trust_remote_code=True,
            cache_dir=WEIGHTS_DIR
        )
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"[CRITICAL ERROR] Model loading failed: {e}")
        sys.exit(1)

    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return model, device, transform_image

def get_mask_from_model(image_path, model, device, transform_image, threshold=0.5):
    try:
        image_pil = Image.open(image_path).convert('RGB')
        original_w, original_h = image_pil.size
        
        input_tensor = transform_image(image_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds = model(input_tensor)
            if isinstance(preds, (list, tuple)):
                pred_logits = preds[-1]
            else:
                pred_logits = preds
        
        pred_map = torch.sigmoid(pred_logits)
        pred_map = F.interpolate(pred_map, size=(original_h, original_w), mode='bilinear', align_corners=False)
        pred_map = pred_map.squeeze().cpu()
        
        binary_mask = (pred_map > threshold).float().numpy()
        mask_cv2 = (binary_mask * 255).astype(np.uint8)
        
        return mask_cv2

    except Exception as e:
        print(f"    [ERROR] Model inference failed for {image_path}: {e}")
        return None

def imread_safe(file_path, flags=cv2.IMREAD_COLOR):
    try:
        n = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(f"    [ERROR] Read file failed {file_path}: {e}")
        return None

def imwrite_safe(file_path, img):
    try:
        ext = os.path.splitext(file_path)[1]
        result, n = cv2.imencode(ext, img)
        if result:
            with open(file_path, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(f"    [ERROR] Write file failed {file_path}: {e}")
        return False

def get_random_safe_point(safe_coords, width, height):
    if safe_coords is not None and len(safe_coords[0]) > 0:
        safe_y, safe_x = safe_coords
        idx = random.randint(0, len(safe_y) - 1)
        return safe_x[idx], safe_y[idx]
    else:
        return random.randint(0, width), random.randint(0, height)

def apply_scratches(image, scale_factor=1.0, safe_coords=None):
    height, width, _ = image.shape
    num_scratches = random.randint(3, 15)
    scratches_layer = image.copy()
    for _ in range(num_scratches):
        x1, y1 = get_random_safe_point(safe_coords, width, height)
        angle = random.uniform(0, 180)
        length = int(random.randint(int(height*0.2), int(height*0.8)) * scale_factor)
        x2 = int(x1 + length * np.cos(np.deg2rad(angle)))
        y2 = int(y1 + length * np.sin(np.deg2rad(angle)))
        color_val = random.randint(50, 200)
        color = (color_val, color_val, color_val)
        thickness = random.randint(1, 2)
        cv2.line(scratches_layer, (x1, y1), (x2, y2), color, thickness)
    alpha = random.uniform(0.6, 0.95)
    return cv2.addWeighted(scratches_layer, alpha, image, 1 - alpha, 0)

def apply_stains(image, scale_factor=1.0, safe_coords=None):
    height, width, _ = image.shape
    num_stains = random.randint(1, 2)
    stain_layer = image.astype(np.float32)
    for _ in range(num_stains):
        stain_mask = np.zeros((height, width), dtype=np.uint8)
        cx, cy = get_random_safe_point(safe_coords, width, height)
        ax_w = int(random.randint(int(width*0.04), int(width*0.12)) * scale_factor)
        ax_h = int(random.randint(int(height*0.04), int(height*0.12)) * scale_factor)
        axes = (max(1, ax_w), max(1, ax_h))
        angle = random.randint(0, 360)
        cv2.ellipse(stain_mask, (cx, cy), axes, angle, 0, 360, 255, -1)
        blur_val = random.randrange(51, 151, 2)
        stain_mask = cv2.GaussianBlur(stain_mask, (blur_val, blur_val), 0)
        stain_mask = (stain_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
        stain_color_effect = np.zeros_like(stain_layer)
        stain_color_effect[:,:] = (10, 25, 30)
        stain_layer = stain_layer * (1.0 - stain_mask * random.uniform(0.3, 0.6))
        stain_layer += stain_color_effect * stain_mask * random.uniform(0.2, 0.5)
    stain_layer = np.clip(stain_layer, 0, 255)
    return stain_layer.astype(np.uint8)

def apply_tape_residue(image, scale_factor=1.0, safe_coords=None):
    height, width, _ = image.shape
    tape_w = int(random.randint(int(width * 0.05), int(width * 0.2)) * scale_factor)
    tape_h = int(random.randint(int(height * 0.05), int(height * 0.2)) * scale_factor)
    if tape_w < 1 or tape_h < 1: return image
    
    pts = np.array([
        [0, 0], [tape_w + random.randint(-10, 10), random.randint(-10, 10)],
        [tape_w + random.randint(-10, 10), tape_h + random.randint(-10, 10)],
        [random.randint(-10, 10), tape_h + random.randint(-10, 10)]
    ], dtype=np.float32)
    pts = pts.reshape(-1, 1, 2)
    angle = random.uniform(0, 360)
    
    x_offset, y_offset = get_random_safe_point(safe_coords, width, height)
    
    center = (tape_w / 2, tape_h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += x_offset
    M[1, 2] += y_offset
    transformed_pts = cv2.transform(pts, M)
    tape_layer = image.copy()
    color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
    cv2.fillPoly(tape_layer, [np.int32(transformed_pts)], color)
    alpha = random.uniform(0.7, 1.0)
    return cv2.addWeighted(tape_layer, alpha, image, 1 - alpha, 0)

def apply_splatter(image):
    height, width, _ = image.shape
    splatter_layer = image.copy()
    cx = random.randint(0, width)
    cy = random.randint(0, height)
    num_splats = random.randint(15, 40)
    splat_range = int(min(width, height) * random.uniform(0.05, 0.15))
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for _ in range(num_splats):
        offsetX = int(random.gauss(0, splat_range))
        offsetY = int(random.gauss(0, splat_range))
        radius = random.randint(2, 8)
        cv2.circle(splatter_layer, (cx + offsetX, cy + offsetY), radius, color, -1)
    splatter_layer = cv2.GaussianBlur(splatter_layer, (3, 3), 0)
    alpha = random.uniform(0.8, 1.0)
    return cv2.addWeighted(splatter_layer, alpha, image, 1 - alpha, 0)

def apply_cracks(image, scale_factor=1.0, safe_coords=None):
    height, width, _ = image.shape
    crack_layer = image.copy()
    start_x, start_y = get_random_safe_point(safe_coords, width, height)
    num_steps = int(random.randint(80, 200) * scale_factor)
    if num_steps < 10: return image
    
    x, y = start_x, start_y
    for _ in range(num_steps):
        px, py = x, y
        x += random.randint(-1, 1) * random.randint(1,5)
        y += random.randint(-1, 1) * random.randint(1,5)
        x = np.clip(x, 0, width-1)
        y = np.clip(y, 0, height-1)
        thickness = random.randint(1, 3)
        color_val = random.randint(0, 50)
        color = (color_val, color_val, color_val)
        cv2.line(crack_layer, (px, py), (x, y), color, thickness)
        if random.random() < 0.05:
            branch_x, branch_y = x, y
            for _ in range(random.randint(20, 50)):
                pbx, pby = branch_x, branch_y
                branch_x += random.randint(-1, 1) * random.randint(1,3)
                branch_y += random.randint(-1, 1) * random.randint(1,3)
                cv2.line(crack_layer, (pbx, pby), (branch_x, branch_y), color, 1)
    return crack_layer

def main():
    parser = argparse.ArgumentParser(description="Augmentation Tool with BiRefNet Masking")
    parser.add_argument('--dataset', type=str, required=True, choices=['mvtec', 'realiad', 'visa'], help="Target dataset: mvtec, realiad, or visa")
    parser.add_argument('--root', type=str, required=True, help="Root directory path of the dataset")
    args = parser.parse_args()

    image_root = args.root
    categories = DATASET_CONFIG[args.dataset]
    
    print(f"Selected Dataset: {args.dataset}")
    print(f"Dataset Root: {image_root}")
    print(f"Categories: {len(categories)}")

    model, device, transform = init_segmentation_model()

    augmentation_functions = [
        apply_scratches, apply_stains, apply_tape_residue,
        apply_splatter, apply_cracks,
    ]
    
    # Extensions to search for
    valid_extensions = ['*.png', '*.jpg', '*.JPG', '*.jpeg']

    for category in categories:
        print(f"\n--- Processing Category: {category} ---")
        image_test_dir = os.path.join(image_root, category, 'test')

        if not os.path.isdir(image_test_dir):
            print(f"  [WARNING] Path not found: {image_test_dir}, skipping.")
            continue
        
        image_paths = []
        for ext in valid_extensions:
            image_paths.extend(glob.glob(os.path.join(image_test_dir, '**', ext), recursive=True))
            
        if not image_paths:
            print(f"  [INFO] No valid images (.png, .jpg, .JPG, .jpeg) found in {image_test_dir} or subfolders.")
            continue

        for image_path in image_paths:
            relative_path = os.path.relpath(image_path, image_test_dir)
            
            print(f"  > Processing: {relative_path}")
            
            image = imread_safe(image_path)
            if image is None: continue

            mask = get_mask_from_model(image_path, model, device, transform)
            
            if mask is None:
                print(f"    [WARNING] Failed to generate mask for {relative_path}, skipping.")
                continue
                
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            foreground = cv2.bitwise_and(image, image, mask=mask)
            
            margin = 15
            kernel_size = 2 * margin + 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask_forbidden_zone = cv2.dilate(mask, kernel)
            mask_safe_zone = cv2.bitwise_not(mask_forbidden_zone)
            
            safe_coords = np.where(mask_safe_zone > 0)

            current_background = cv2.bitwise_and(image, image, mask=mask_inv)
            
            num_augmentations_to_apply = random.randint(0, 3)
            if num_augmentations_to_apply == 0:
                imwrite_safe(image_path, image)
                continue

            functions_to_apply = random.sample(augmentation_functions, num_augmentations_to_apply)
            
            for func in functions_to_apply:
                if func.__name__ == 'apply_splatter':
                    candidate_background = func(current_background)
                    safe_part = cv2.bitwise_and(candidate_background, candidate_background, mask=mask_safe_zone)
                    preserved_part = cv2.bitwise_and(current_background, current_background, mask=mask_forbidden_zone)
                    current_background = cv2.add(safe_part, preserved_part)
                else:
                    scale_factor = 1.0
                    retry_count = 0
                    max_retries = 500
                    last_candidate = None

                    while retry_count < max_retries:
                        candidate_background = func(current_background, scale_factor=scale_factor, safe_coords=safe_coords)
                        last_candidate = candidate_background.copy()
                        
                        diff = cv2.absdiff(candidate_background, current_background)
                        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                        _, pollution_mask = cv2.threshold(diff_gray, 1, 255, cv2.THRESH_BINARY)
                        collision = cv2.bitwise_and(pollution_mask, mask_forbidden_zone)
                        
                        if cv2.countNonZero(collision) == 0:
                            current_background = candidate_background
                            break
                        
                        retry_count += 1
                        scale_factor *= 0.95 
                    
                    if retry_count == max_retries:
                        safe_part = cv2.bitwise_and(last_candidate, last_candidate, mask=mask_safe_zone)
                        preserved_part = cv2.bitwise_and(current_background, current_background, mask=mask_forbidden_zone)
                        trimmed_background = cv2.add(safe_part, preserved_part)
                        current_background = trimmed_background
            
            final_background = cv2.bitwise_and(current_background, current_background, mask=mask_inv)
            final_image = cv2.add(foreground, final_background)
            
            imwrite_safe(image_path, final_image)

    print("\n--- All categories processed ---")

if __name__ == "__main__":
    main()

#python background_noise_injector.py --dataset visa --root ./VisA/1cls
#python background_noise_injector.py --dataset mvtec --root ./mvtec_anomaly_detection
#python background_noise_injector.py --dataset realiad --root ./Real-IAD