import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from torchvision import transforms
from PIL import Image

try:
    from perlin import rand_perlin_2d_np
except ImportError:
    print("Warning: 'perlin' module not found. Please ensure perlin.py is available.")

    def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
        print("Warning: Using dummy perlin noise function.")
        return np.random.rand(shape[0], shape[1])

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomAnomalyTrainDataset(Dataset):
    def __init__(self, root_dir, data_transform, gt_transform):
        self.root_dir = root_dir
        self.data_transform = data_transform
        self.gt_transform = gt_transform

        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*.png")))
        if not self.image_paths:
            self.image_paths.extend(sorted(glob.glob(os.path.join(root_dir, "*.jpg"))))
        if not self.image_paths:
            self.image_paths.extend(sorted(glob.glob(os.path.join(root_dir, "*.bmp"))))
        if not self.image_paths:
            self.image_paths.extend(sorted(glob.glob(os.path.join(root_dir, "*.JPG"))))

        if not self.image_paths:
            raise ValueError(f"No image files (png, jpg, bmp, JPG) found in {root_dir}")

        parent_of_good_dir = os.path.dirname(self.root_dir)
        item_dir = os.path.dirname(parent_of_good_dir)

        train_level_mask_dir = os.path.join(parent_of_good_dir, "mask")
        item_level_mask_dir = os.path.join(item_dir, "mask")

        mask_dir_found = None
        if os.path.isdir(train_level_mask_dir):
            mask_dir_found = train_level_mask_dir
            logger.info(f"Found mask directory (sibling to 'good'): {mask_dir_found}")
        elif os.path.isdir(item_level_mask_dir):
            mask_dir_found = item_level_mask_dir
            logger.info(f"Found mask directory (category level): {mask_dir_found}")
        else:
            raise ValueError(
                f"Mask directory not found at {train_level_mask_dir} or {item_level_mask_dir}. 'root_dir' is: {self.root_dir}")

        self.mask_paths = []
        missing_masks = 0
        temp_image_paths = []
        for img_path in self.image_paths:
            base_name = os.path.basename(img_path)
            name_no_ext = os.path.splitext(base_name)[0]
            
            candidate_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG']
            mask_path = None
            

            for ext in candidate_exts:
                candidate_path = os.path.join(mask_dir_found, name_no_ext + ext)
                if os.path.exists(candidate_path):
                    mask_path = candidate_path
                    break
            
            if mask_path is None:
                mask_path = os.path.join(mask_dir_found, name_no_ext + ".png")

            if os.path.exists(mask_path):
                temp_image_paths.append(img_path)
                self.mask_paths.append(mask_path)
            else:
                logger.warning(f"Mask file not found for image {img_path}. Expected path {mask_path}. Skipping this image.")
                missing_masks += 1

        self.image_paths = temp_image_paths
        if not self.image_paths:
            raise ValueError("No valid image/mask pairs found after filtering. Please check mask paths and filenames.")
        if missing_masks > 0:
            logger.warning(f"Removed {missing_masks} images due to missing masks. Currently using {len(self.image_paths)} images.")

        self.anomaly_source_dir = "./dtd/images/"
        self.anomaly_source_paths = sorted(glob.glob(os.path.join(self.anomaly_source_dir, "*", "*.jpg")))
        if not self.anomaly_source_paths:
            self.anomaly_source_paths.extend(sorted(glob.glob(os.path.join(self.anomaly_source_dir, "*.jpg"))))
        if not self.anomaly_source_paths:
            raise ValueError(f"No anomaly source files (*.jpg) found in {self.anomaly_source_dir} or its subdirectories")

        self.augmenters_texture = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                                   iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                                   iaa.pillike.EnhanceSharpness(),
                                   iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                                   iaa.Solarize(0.5, threshold=(32, 128)),
                                   iaa.Posterize(),
                                   iaa.Invert(),
                                   iaa.pillike.Autocontrast(),
                                   iaa.pillike.Equalize(),
                                   iaa.Affine(rotate=(-45, 45))]

        self.rot_perlin = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def __len__(self):
        return len(self.image_paths)

    def _randAugmenter_texture(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters_texture)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters_texture[aug_ind[0]],
                              self.augmenters_texture[aug_ind[1]],
                              self.augmenters_texture[aug_ind[2]]])
        return aug

    def _generate_perlin_noise(self, target_hw_shape):
        perlin_scale = 6
        min_perlin_scale = 0
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).item())
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).item())

        perlin_noise_np = rand_perlin_2d_np((target_hw_shape[0], target_hw_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise_np = self.rot_perlin(image=perlin_noise_np)

        threshold = 0.5
        perlin_thr_np = np.where(perlin_noise_np > threshold, 1.0, 0.0).astype(np.float32)
        return np.expand_dims(perlin_thr_np, axis=2)

    def _apply_anomaly_augmentation(self, image_np_hwc_01, object_mask_np_hw1_01, perlin_noise_mask_np_hw1,
                                    anomaly_texture_path, attempt_anomaly):
        target_h, target_w = image_np_hwc_01.shape[:2]

        texture_aug = self._randAugmenter_texture()
        try:
            anomaly_source_img_bgr = cv2.imread(anomaly_texture_path)
            if anomaly_source_img_bgr is None:
                raise IOError(f"Could not read anomaly source image: {anomaly_texture_path}")
            anomaly_source_img_rgb = cv2.cvtColor(anomaly_source_img_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Failed to read or convert anomaly source image {anomaly_texture_path}: {e}")
            return image_np_hwc_01, image_np_hwc_01, np.zeros_like(perlin_noise_mask_np_hw1,
                                                                   dtype=np.float32), np.array([0.0], dtype=np.float32)

        anomaly_source_img_resized = cv2.resize(anomaly_source_img_rgb, dsize=(target_w, target_h))
        anomaly_texture_augmented_np = texture_aug(image=anomaly_source_img_resized)
        anomaly_texture_01 = anomaly_texture_augmented_np.astype(np.float32) / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_base = image_np_hwc_01 * (1 - perlin_noise_mask_np_hw1) + \
                         (1 - beta) * anomaly_texture_01 * perlin_noise_mask_np_hw1 + \
                         beta * image_np_hwc_01 * perlin_noise_mask_np_hw1
        augmented_base = np.clip(augmented_base, 0, 1)

        perlin_squeezed_uint8 = (perlin_noise_mask_np_hw1.squeeze() * 255).astype(np.uint8)
        num_labels, labels_map = cv2.connectedComponents(perlin_squeezed_uint8)

        object_mask_squeezed_01 = object_mask_np_hw1_01.squeeze()
        if object_mask_squeezed_01.shape != labels_map.shape:
            raise ValueError(f"Shape mismatch between mask and label map: obj {object_mask_squeezed_01.shape}, map {labels_map.shape}")

        touching_labels = np.unique(labels_map[object_mask_squeezed_01 > 0.5])
        touching_labels = touching_labels[touching_labels != 0]

        noise_chunks_touching_object_np_hw1 = np.zeros_like(labels_map, dtype=np.float32)
        for label_val in touching_labels:
            noise_chunks_touching_object_np_hw1[labels_map == label_val] = 1.0
        noise_chunks_touching_object_np_hw1 = np.expand_dims(noise_chunks_touching_object_np_hw1, axis=2)

        has_anomaly_applied = np.array([0.0], dtype=np.float32)
        if attempt_anomaly and np.sum(noise_chunks_touching_object_np_hw1) > 0:
            background_noise_final_mask = perlin_noise_mask_np_hw1 * (1.0 - object_mask_np_hw1_01) * (
                    1.0 - noise_chunks_touching_object_np_hw1)
            image_out = image_np_hwc_01 * (
                    1.0 - background_noise_final_mask) + augmented_base * background_noise_final_mask

            augmented_image_out = augmented_base

            anomaly_mask_out = noise_chunks_touching_object_np_hw1
            has_anomaly_applied = np.array([1.0], dtype=np.float32)
        else:
            image_out = image_np_hwc_01
            augmented_image_out = image_np_hwc_01
            anomaly_mask_out = np.zeros_like(perlin_noise_mask_np_hw1, dtype=np.float32)

        return (np.clip(image_out, 0, 1).astype(np.float32),
                np.clip(augmented_image_out, 0, 1).astype(np.float32),
                anomaly_mask_out.astype(np.float32),
                has_anomaly_applied)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        try:
            img_pil = Image.open(image_path).convert("RGB")
            mask_pil = Image.open(mask_path).convert("L")
        except Exception as e:
            logger.error(f"Error loading image/mask: {image_path}, {mask_path}: {e}", exc_info=True)
            next_idx = (idx + 1) % len(self)
            if len(self) == 1: raise RuntimeError(f"The only sample in the dataset {image_path} failed to load.") from e
            logger.warning(f"Trying next index {next_idx}")
            return self.__getitem__(next_idx)

        if not (isinstance(self.data_transform, transforms.Compose) and
                isinstance(self.data_transform.transforms[-1], transforms.Normalize) and
                len(self.data_transform.transforms) > 1):
            raise ValueError(
                "The provided data_transform must be a Compose object, with the last transform being Normalize, and at least one transform prior.")

        pre_normalization_img_transforms = transforms.Compose(self.data_transform.transforms[:-1])
        normalization_img_transform = self.data_transform.transforms[-1]

        image_tensor_before_norm = pre_normalization_img_transforms(img_pil)
        mask_tensor_transformed = self.gt_transform(mask_pil)

        image_np_hwc_01 = image_tensor_before_norm.permute(1, 2, 0).numpy()
        object_mask_np_hw1_01 = mask_tensor_transformed.permute(1, 2, 0).numpy()

        crop_h, crop_w = image_np_hwc_01.shape[:2]
        perlin_noise_mask_np_hw1 = self._generate_perlin_noise((crop_h, crop_w))

        anomaly_source_img_path = self.anomaly_source_paths[
            torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        ]

        no_anomaly_roll = torch.rand(1).item()
        attempt_anomaly = no_anomaly_roll <= 0.5

        try:
            image_out_np, aug_image_out_np, anomaly_mask_np, has_anomaly = \
                self._apply_anomaly_augmentation(image_np_hwc_01,
                                                 object_mask_np_hw1_01,
                                                 perlin_noise_mask_np_hw1,
                                                 anomaly_source_img_path,
                                                 attempt_anomaly)
        except Exception as e:
            logger.error(f"Error processing image {image_path} in _apply_anomaly_augmentation: {e}", exc_info=True)
            next_idx = (idx + 1) % len(self)
            if len(self) == 1: raise RuntimeError(f"The only sample in the dataset {image_path} failed during augmentation.") from e
            logger.warning(f"Trying next index {next_idx}")
            return self.__getitem__(next_idx)

        image_out_tensor_before_norm = torch.from_numpy(image_out_np.transpose((2, 0, 1)))
        aug_image_out_tensor_before_norm = torch.from_numpy(aug_image_out_np.transpose((2, 0, 1)))
        anomaly_mask_tensor = torch.from_numpy(anomaly_mask_np.transpose((2, 0, 1)))

        final_image_out = normalization_img_transform(image_out_tensor_before_norm)
        final_aug_image_out = normalization_img_transform(aug_image_out_tensor_before_norm)

        return {
            'image': final_image_out,
            'augmented_image': final_aug_image_out,
            'anomaly_mask': anomaly_mask_tensor,
            'has_anomaly': torch.tensor(has_anomaly, dtype=torch.float32),
            'idx': torch.tensor(idx, dtype=torch.int64),
            'image_path': image_path
        }