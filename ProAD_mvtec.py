import torch
import torch.nn as nn
from dataset import get_data_transforms, MVTecDataset
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader, ConcatDataset
import torch.backends.cudnn as cudnn
import argparse
from torch.nn import functional as F
from functools import partial
import logging
import math
import warnings
import torch.optim as optim

from models.uad import EncoderFeatureExtractor, ViTillDecoder, ViTillCombined
from models import vit_encoder
from dinov1.utils import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention, LinearAttention2, ConvBlock, FeatureJitter
from utils import evaluation_batch, global_cosine, regional_cosine_hm_percent, global_cosine_hm_percent, WarmCosineScheduler, new_evaluation_batch
from optimizers import StableAdamW
from synthetic_dataset import CustomAnomalyTrainDataset
from models.preprocess_masks import preprocess_datasets
from models.seg_model import UNetSegmentationHead

warnings.filterwarnings("ignore")

def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    if logger.hasHandlers():
        logger.handlers.clear()

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fileHandler = logging.FileHandler(save_path)
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    logger.propagate = False
    return logger

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_student_model(args, item_list, device, print_fn):
    setup_seed(1)

    total_iters = 100000
    batch_size = 16
    image_size = 448
    crop_size = 392

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list = []
    test_data_list = []
    for i, item in enumerate(item_list):
        train_good_path = os.path.join(args.data_path, item, 'train', 'good')
        train_data_instance = CustomAnomalyTrainDataset(
            root_dir=train_good_path,
            data_transform=data_transform,
            gt_transform=gt_transform
        )

        test_path = os.path.join(args.data_path, item)

        test_data_instance = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform,
                                          phase="test")
        train_data_list.append(train_data_instance)
        test_data_list.append(test_data_instance)

    train_data_concat = ConcatDataset(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(train_data_concat, batch_size=batch_size, shuffle=True,
                                                   num_workers=4,
                                                   drop_last=True,
                                                   pin_memory=True
                                                   )

    encoder_name = 'dinov2reg_vit_large_14'
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1], [2, 3, 4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1], [2, 3, 4, 5, 6, 7]]
    mask_neighbor_size_cfg = 0

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError("Architecture not in small, base, large.")

    original_encoder_full_model = vit_encoder.load(encoder_name)
    original_encoder_full_model = original_encoder_full_model.to(device)
    original_encoder_full_model.eval()
    for param in original_encoder_full_model.parameters():
        param.requires_grad = False

    effective_remove_class_token_setting = False
    num_registers = getattr(original_encoder_full_model, 'num_register_tokens', 0)

    encoder_feature_extractor = EncoderFeatureExtractor(
        encoder_instance=original_encoder_full_model,
        target_layers=target_layers,
        fuse_layer_encoder=fuse_layer_encoder
    ).to(device)
    encoder_feature_extractor.eval()

    bottleneck_modules = []
    bottleneck_modules.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.25))
    bottleneck_modules = nn.ModuleList(bottleneck_modules)

    decoder_modules = []
    for _ in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                       attn=LinearAttention2)
        decoder_modules.append(blk)
    decoder_modules = nn.ModuleList(decoder_modules)

    trainable_model_part = ViTillDecoder(
        bottleneck=bottleneck_modules,
        decoder=decoder_modules,
        fuse_layer_decoder=fuse_layer_decoder,
        mask_neighbor_size=mask_neighbor_size_cfg,
        effective_remove_class_token_config=effective_remove_class_token_setting,
        num_register_tokens=num_registers
    ).to(device)

    init_modules = nn.ModuleList([trainable_model_part.bottleneck, trainable_model_part.decoder])
    for m in init_modules.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    optimizer = StableAdamW([{'params': trainable_model_part.parameters()}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=total_iters,
                                       warmup_iters=100)

    model_for_evaluation = ViTillCombined(
        encoder_feature_extractor=encoder_feature_extractor,
        vitill_decoder=trainable_model_part,
        effective_remove_class_token_config=effective_remove_class_token_setting
    ).to(device)

    model_params_save_dir = "checkpoints"
    os.makedirs(model_params_save_dir, exist_ok=True)

    it = 0

    for epoch in range(int(math.ceil(total_iters / len(train_dataloader)))):
        trainable_model_part.train()
        loss_list = []
        for data_batch in train_dataloader:
            if it >= total_iters:
                break

            img = data_batch['image'].to(device)
            augmented_image = data_batch['augmented_image'].to(device)

            x_for_bottleneck, _, side = encoder_feature_extractor(
                augmented_image,
                remove_tokens_for_bottleneck_input_flag=effective_remove_class_token_setting
            )

            _, en_for_loss, _ = encoder_feature_extractor(
                img,
                remove_tokens_for_bottleneck_input_flag=effective_remove_class_token_setting
            )

            de_for_loss = trainable_model_part(
                x_for_bottleneck,
                img.shape[0],
                side
            )

            p_final = 0.9
            p = min(p_final * it / 1000, p_final)
            loss = global_cosine_hm_percent(en_for_loss, de_for_loss, p=p, factor=0.1)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_model_part.parameters(), max_norm=0.1)
            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()

            if (it + 1) == total_iters:
                completed_iterations = it + 1
                model_save_path = os.path.join(model_params_save_dir, "mvtec_student.pth")

                torch.save(trainable_model_part.state_dict(), model_save_path)
                print_fn(f"Saved model parameters to {model_save_path} after {completed_iterations} iterations")

                trainable_model_part.eval()
                model_for_evaluation.eval()

                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for item_eval, test_data_eval in zip(item_list, test_data_list):
                    test_dataloader_eval = torch.utils.data.DataLoader(test_data_eval, batch_size=batch_size,
                                                                       shuffle=False, num_workers=4)
                    results = evaluation_batch(model_for_evaluation, test_dataloader_eval, device, max_ratio=0.01,
                                               resize_mask=256)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)
                    print_fn(
                        '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                            item_eval, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

                print_fn(
                    'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                        np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list), np.mean(auroc_px_list),
                        np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))

                trainable_model_part.train()

            it += 1
            if it >= total_iters:
                break

        avg_loss = np.mean(loss_list) if loss_list else 0.0
        print_fn('Epoch [{}/{}], Iteration [{}/{}], loss:{:.4f}'.format(epoch + 1, int(math.ceil(
            total_iters / len(train_dataloader))), it, total_iters, avg_loss))

        if it >= total_iters:
            print_fn(f"Total iterations ({total_iters}) reached. Training finished.")
            break

    return

def train_segmentation_model(args, item_list_selected, device, print_fn, mvtec_item_list_all_for_train):
    setup_seed(1)
    
    batch_size = 16
    image_size = 448
    crop_size = 392

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list = []
    print_fn(f"Preparing training data for AFM and upsampler using items: {mvtec_item_list_all_for_train}")
    for item in mvtec_item_list_all_for_train:
        train_good_path = os.path.join(args.data_path, item, 'train', 'good')
        if os.path.exists(train_good_path):
            train_data_instance = CustomAnomalyTrainDataset(
                root_dir=train_good_path,
                data_transform=data_transform,
                gt_transform=gt_transform,
            )
            train_data_list.append(train_data_instance)
        else:
            print_fn(f"Warning: Training data path not found for item {item} at {train_good_path}")

    if not train_data_list:
        print_fn("Error: No training data loaded. Cannot proceed with training the AFM and upsampler model.")
        return

    train_data_concat = ConcatDataset(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(
        train_data_concat,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )

    test_data_list = []
    print_fn(f"Preparing test data for items: {item_list_selected}")
    for item in item_list_selected:
        test_path = os.path.join(args.data_path, item)
        test_data_instance = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform,
                                          phase="test")
        test_data_list.append(test_data_instance)

    if not test_data_list:
        print_fn("Error: No test data loaded. Exiting.")
        return

    encoder_name = 'dinov2reg_vit_large_14'
    print_fn(f"Initializing model with encoder: {encoder_name}")

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError("Architecture not supported.")

    fuse_layer_encoder = [[0, 1], [2, 3, 4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1], [2, 3, 4, 5, 6, 7]]
    mask_neighbor_size_cfg = 0
    effective_remove_class_token_setting = False

    original_encoder_full_model = vit_encoder.load(encoder_name)
    original_encoder_full_model = original_encoder_full_model.to(device)
    original_encoder_full_model.eval()
    for param in original_encoder_full_model.parameters():
        param.requires_grad = False

    num_registers = getattr(original_encoder_full_model, 'num_register_tokens', 0)
    
    encoder_feature_extractor = EncoderFeatureExtractor(
        encoder_instance=original_encoder_full_model,
        target_layers=target_layers,
        fuse_layer_encoder=fuse_layer_encoder
    ).to(device)
    encoder_feature_extractor.eval()

    bottleneck_modules = nn.ModuleList(
        [bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.1)])

    decoder_modules = []
    num_decoder_blocks = 8
    for _ in range(num_decoder_blocks):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                       attn=LinearAttention2)
        decoder_modules.append(blk)
    decoder_modules = nn.ModuleList(decoder_modules)

    trainable_model_part = ViTillDecoder(
        bottleneck=bottleneck_modules,
        decoder=decoder_modules,
        fuse_layer_decoder=fuse_layer_decoder,
        mask_neighbor_size=mask_neighbor_size_cfg,
        effective_remove_class_token_config=effective_remove_class_token_setting,
        num_register_tokens=num_registers
    ).to(device)

    params_path = os.path.join("checkpoints", "mvtec_student.pth")

    if os.path.exists(params_path):
        print_fn(f"Loading model parameters from: {params_path}")
        try:
            trainable_model_part.load_state_dict(torch.load(params_path, map_location=device))
            print_fn(f"Successfully loaded model parameters into student model.")
        except Exception as e:
            print_fn(f"Error loading state_dict: {e}")
            return
    else:
        print_fn(f"Error: Model parameters not found at {params_path}")
        return

    trainable_model_part.eval()

    model_for_evaluation = ViTillCombined(
        encoder_feature_extractor=encoder_feature_extractor,
        vitill_decoder=trainable_model_part,
        effective_remove_class_token_config=effective_remove_class_token_setting
    ).to(device)
    model_for_evaluation.eval()

    seg_learning_rate = 3e-5
    seg_weight_decay = 5e-5

    seg_model_input_channels = 2050
    seg_model_output_channels = 1

    segmentation_model = UNetSegmentationHead(
        in_channels=seg_model_input_channels,
        n_classes=seg_model_output_channels,
        target_size=(crop_size, crop_size)
    ).to(device)

    criterion_seg = nn.BCELoss()

    optimizer_seg = optim.AdamW(
        segmentation_model.parameters(),
        lr=seg_learning_rate,
        weight_decay=seg_weight_decay
    )

    scheduler_seg = torch.optim.lr_scheduler.StepLR(optimizer_seg, step_size=100, gamma=0.5)

    print_fn(f"AFM and upsampler model initialized. Training for 100 epochs.")

    num_epochs_to_train = 100

    for epoch in range(num_epochs_to_train):
        segmentation_model.train()
        running_loss_seg = 0.0
        num_batches = 0
        for data_batch in train_dataloader:
            augmented_image = data_batch['augmented_image'].to(device)
            anomaly_mask = data_batch['anomaly_mask'].to(device)

            with torch.no_grad():
                output = model_for_evaluation(augmented_image)

            en, de = output[0], output[1]
            en_0 = en[0]
            de_0 = de[0]
            en_1 = en[1]
            de_1 = de[1]
            a_map_0 = 1 - F.cosine_similarity(en_0, de_0)
            a_map_0 = torch.unsqueeze(a_map_0, dim=1)
            a_map_1 = 1 - F.cosine_similarity(en_1, de_1)
            a_map_1 = torch.unsqueeze(a_map_1, dim=1)

            numerator = en_0 * de_0
            norm_en_0 = torch.norm(en_0, p=2, dim=1, keepdim=True)
            norm_de_0 = torch.norm(de_0, p=2, dim=1, keepdim=True)
            denominator = norm_en_0 * norm_de_0 + 1e-8
            X_1 = numerator / denominator
            numerator = en_1 * de_1
            norm_en_0 = torch.norm(en_1, p=2, dim=1, keepdim=True)
            norm_de_0 = torch.norm(de_1, p=2, dim=1, keepdim=True)
            denominator = norm_en_0 * norm_de_0 + 1e-8
            X_2 = numerator / denominator

            a_concat = torch.cat([a_map_0, a_map_1, X_1, X_2], dim=1)

            optimizer_seg.zero_grad()

            predicted_mask = segmentation_model(a_concat)

            loss_seg = criterion_seg(predicted_mask, anomaly_mask)

            loss_seg.backward()

            optimizer_seg.step()

            running_loss_seg += loss_seg.item()
            num_batches += 1

        if num_batches > 0:
            avg_epoch_loss_seg = running_loss_seg / num_batches
        else:
            avg_epoch_loss_seg = 0.0
            print_fn(f"Warning: Epoch [{epoch + 1}/{num_epochs_to_train}] had no batches processed.")

        current_lr = optimizer_seg.param_groups[0]['lr']
        print_fn(f"Epoch [{epoch + 1}/{num_epochs_to_train}], Segmentation Loss: {avg_epoch_loss_seg:.4f}, LR: {current_lr:.6f}")

        if scheduler_seg:
            scheduler_seg.step()

        if (epoch + 1) % 10 == 0:
            current_epoch_num = epoch + 1
            model_save_path = os.path.join("checkpoints", "mvtec_afm_upsampler.pth")

            torch.save(segmentation_model.state_dict(), model_save_path)
            print_fn(f"Saved AFM and upsampler model to {model_save_path}")

            segmentation_model.eval()
            auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
            auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []
            print_fn(f"Evaluating AFM and upsampler model on {len(item_list_selected)} categories...")
            
            with torch.no_grad():
                for item_eval, test_data_eval in zip(item_list_selected, test_data_list):
                    if not test_data_eval:
                        continue

                    test_dataloader_eval = torch.utils.data.DataLoader(test_data_eval, batch_size=batch_size,
                                                                       shuffle=False, num_workers=4,
                                                                       pin_memory=True)

                    results = new_evaluation_batch(model=model_for_evaluation, model_2=segmentation_model,
                                                   dataloader=test_dataloader_eval,
                                                   device=device,
                                                   max_ratio=0.01,
                                                   resize_mask=256)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)
                    print_fn(
                        '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                            item_eval, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

            mean_auroc_sp = np.mean(auroc_sp_list) if auroc_sp_list else 0
            mean_ap_sp = np.mean(ap_sp_list) if ap_sp_list else 0
            mean_f1_sp = np.mean(f1_sp_list) if f1_sp_list else 0
            mean_auroc_px = np.mean(auroc_px_list) if auroc_px_list else 0
            mean_ap_px = np.mean(ap_px_list) if ap_px_list else 0
            mean_f1_px = np.mean(f1_px_list) if f1_px_list else 0
            mean_aupro_px = np.mean(aupro_px_list) if aupro_px_list else 0

            print_fn(
                f'Mean Scores after Epoch {current_epoch_num}: '
                f'I-AUROC:{mean_auroc_sp:.4f}, I-AP:{mean_ap_sp:.4f}, I-F1:{mean_f1_sp:.4f}, '
                f'P-AUROC:{mean_auroc_px:.4f}, P-AP:{mean_ap_px:.4f}, P-F1:{mean_f1_px:.4f}, P-AUPRO:{mean_aupro_px:.4f}')

    print_fn("AFM and upsampler model training finished.")

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    logging.getLogger().setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='./mvtec_anomaly_detection')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='mvtec_anomaly_detection')
    args = parser.parse_args()

    item_list_all = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    obj_list = [
        'bottle', 'capsule', 'hazelnut', 'metal_nut',
        'pill', 'screw', 'toothbrush', 'zipper'
    ]
    texture_list = [
        'carpet', 'grid', 'leather', 'tile',
        'wood', 'cable', 'transistor'
    ]

    item_list_selected = item_list_all

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name, 'log.txt'))
    print_fn = logger.info

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print_fn(f"Using device: {device}")

    student_model_path = os.path.join("checkpoints", "mvtec_student.pth")
    
    if os.path.exists(student_model_path):
        print_fn(f"Found existing model at {student_model_path}. Skipping student model training.")
    else:
        print_fn("Starting data preprocessing (Mask Generation)...")
        preprocess_datasets(args.data_path, obj_list, texture_list)
        print_fn("Data preprocessing finished.")
        print_fn("Starting student model training...")
        train_student_model(args, item_list_selected, device, print_fn)
        print_fn("Student model training completed.")

    print_fn("Starting AFM and upsampler model training...")
    train_segmentation_model(args, item_list_selected, device, print_fn, item_list_all)
    #python ProAD_mvtec.py --data_path ./mvtec_anomaly_detection
