import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from typing import List

from models.mvanet import inf_MVANet

# Official MVANet weights (Model_80.pth) released by the authors:
# https://github.com/qianyu-dlut/MVANet
MVANET_WEIGHTS_GDRIVE_ID = "1_gabQXOF03MfXnf3EWDK1d_8wKiOemOv"
MVANET_WEIGHTS_NAME = "Model_80.pth"

def load_mvanet_model(device: torch.device):
    """
    Load the MVANet segmentation model using a specific local weights directory.

    The weights are expected at ./weights/MVANet/Model_80.pth. If the file is
    missing, an automatic download from the official Google Drive release is
    attempted via gdown.

    Note: MVANet follows the same output convention as the previously used
    BiRefNet-HRSOD model, i.e. a single-channel logit map where, after sigmoid,
    the salient object (foreground) has high values. The saved binary masks are
    therefore white (255) on the object and black (0) on the background.
    """
    # Get current working directory
    current_dir = os.getcwd()
    # Define target weights path: ./weights/MVANet
    weights_dir = os.path.join(current_dir, "weights", "MVANet")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, MVANET_WEIGHTS_NAME)

    print(f"[Info] Model weights path: {weights_path}")

    if not os.path.exists(weights_path):
        print("[Info] MVANet weights not found. Downloading from the official release ...")
        try:
            import gdown
            gdown.download(id=MVANET_WEIGHTS_GDRIVE_ID, output=weights_path, quiet=False)
        except Exception as e:
            print(f"[Error] Automatic download failed: {e}")
            print("[Error] Please download Model_80.pth manually from the official MVANet "
                  "repository (https://github.com/qianyu-dlut/MVANet) and place it at "
                  f"{weights_path}")
            return None

    print("[Info] Loading MVANet model ...")
    try:
        model = inf_MVANet()
        pretrained_dict = torch.load(weights_path, map_location="cpu")
        # Tolerant loading, identical to the official MVANet predict.py
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return None

def process_single_image_mvanet(model, image_path: str, output_path: str, device: torch.device, transform_image):
    """
    Process a single image using the MVANet model and save the binary mask
    (white foreground / black background, same convention as before).
    """
    try:
        # Load and convert image
        image = Image.open(image_path).convert('RGB')
        w, h = image.size

        # Preprocessing
        input_tensor = transform_image(image).unsqueeze(0).to(device)

        # Inference (inf_MVANet returns a single-channel logit map at input resolution)
        with torch.no_grad():
            preds = model(input_tensor)
            if isinstance(preds, (list, tuple)):
                pred_logits = preds[-1]
            else:
                pred_logits = preds

        # Post-processing
        pred_map = torch.sigmoid(pred_logits)
        # Resize back to original dimensions
        pred_map = F.interpolate(pred_map, size=(h, w), mode='bilinear', align_corners=False)
        pred_map = pred_map.squeeze().cpu()

        # Binarization (threshold 0.5)
        binary_mask = (pred_map > 0.5).float()

        # Save as image (Mode 'L' for grayscale)
        mask_image = transforms.ToPILImage()(binary_mask)
        mask_image.save(output_path)

    except Exception as e:
        print(f"[Error] Failed to process image {image_path}: {e}")

def process_single_image_white(image_path: str, output_path: str):
    """
    Generate a full white mask based on the original image size and save it.
    """
    try:
        image = Image.open(image_path).convert('RGB')
        w, h = image.size

        # Create a full white image (Mode 'L', value 255)
        white_mask = Image.new('L', (w, h), 255)
        white_mask.save(output_path)

    except Exception as e:
        print(f"[Error] Failed to generate white mask for {image_path}: {e}")

def preprocess_datasets(data_root: str, obj_list: List[str], texture_list: List[str]):
    """
    Main interface to generate masks for datasets.

    Args:
        data_root (str): Root directory of the dataset.
        obj_list (List[str]): List of object categories (uses MVANet model).
        texture_list (List[str]): List of texture categories (uses full white masks).
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] Device used: {device}")

    # --- Step 1: Process Object Categories (MVANet) ---
    print("-" * 50)
    print("[Step 1] Preparing MVANet model for object categories...")

    model = load_mvanet_model(device)

    if model is None:
        print("[Error] Model loading failed. Skipping List 1 processing.")
    else:
        # Define Transforms (MVANet inference size: 1024, ImageNet normalization,
        # identical to the official MVANet predict.py)
        transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for item in obj_list:
            source_dir = os.path.join(data_root, item, 'train', 'good')
            target_dir = os.path.join(data_root, item, 'train', 'mask')

            if not os.path.exists(source_dir):
                print(f"[Warning] Path not found: {source_dir}, skipping.")
                continue

            os.makedirs(target_dir, exist_ok=True)

            files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            print(f"Processing Category [Model]: {item} ({len(files)} images)")

            for file_name in tqdm(files, desc=f"Processing {item}"):
                input_path = os.path.join(source_dir, file_name)
                output_path = os.path.join(target_dir, file_name)

                # Overwrite by default
                process_single_image_mvanet(model, input_path, output_path, device, transform_image)

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # --- Step 2: Process Texture Categories (White Mask) ---
    print("\n" + "-" * 50)
    print("[Step 2] Processing texture categories (White Masks)...")

    for item in texture_list:
        source_dir = os.path.join(data_root, item, 'train', 'good')
        target_dir = os.path.join(data_root, item, 'train', 'mask')

        if not os.path.exists(source_dir):
            print(f"[Warning] Path not found: {source_dir}, skipping.")
            continue

        os.makedirs(target_dir, exist_ok=True)

        files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        print(f"Processing Category [White]: {item} ({len(files)} images)")

        for file_name in tqdm(files, desc=f"Processing {item}"):
            input_path = os.path.join(source_dir, file_name)
            output_path = os.path.join(target_dir, file_name)

            process_single_image_white(input_path, output_path)

    print("\n[Finished] All preprocessing tasks completed.")
