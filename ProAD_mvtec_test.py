import torch
import torch.nn as nn
from dataset import get_data_transforms, MVTecDataset
import numpy as np
import random
import os
from torch.utils.data import DataLoader
import argparse
from functools import partial
import warnings
from models.uad import EncoderFeatureExtractor, ViTillDecoder, ViTillCombined
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from utils import new_evaluation_batch
from models.seg_model import UNetSegmentationHead

warnings.filterwarnings("ignore")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='Testing Script')
    parser.add_argument('--data_path', type=str, default='./mvtec_anomaly_detection')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    setup_seed(1)

    batch_size = 16
    image_size = 448
    crop_size = 392
    
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']

    encoder_name = 'dinov2reg_vit_large_14'
    embed_dim, num_heads = 1024, 16
    target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    fuse_layer_encoder = [[0, 1], [2, 3, 4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1], [2, 3, 4, 5, 6, 7]]
    
    print(f"Initializing Encoder: {encoder_name}")
    original_encoder_full_model = vit_encoder.load(encoder_name)
    original_encoder_full_model = original_encoder_full_model.to(device)
    original_encoder_full_model.eval()
    
    encoder_feature_extractor = EncoderFeatureExtractor(
        encoder_instance=original_encoder_full_model,
        target_layers=target_layers,
        fuse_layer_encoder=fuse_layer_encoder
    ).to(device)
    encoder_feature_extractor.eval()

    bottleneck_modules = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.1)])
    
    decoder_modules = []
    for _ in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                       attn=LinearAttention2)
        decoder_modules.append(blk)
    decoder_modules = nn.ModuleList(decoder_modules)

    num_registers = getattr(original_encoder_full_model, 'num_register_tokens', 0)

    student_model = ViTillDecoder(
        bottleneck=bottleneck_modules,
        decoder=decoder_modules,
        fuse_layer_decoder=fuse_layer_decoder,
        mask_neighbor_size=0,
        effective_remove_class_token_config=False,
        num_register_tokens=num_registers
    ).to(device)

    student_ckpt_path = os.path.join(args.checkpoint_dir, "mvtec_student.pth")
    if os.path.exists(student_ckpt_path):
        print(f"Loading student checkpoint from {student_ckpt_path}")
        student_model.load_state_dict(torch.load(student_ckpt_path, map_location=device))
    else:
        print(f"Error: Student checkpoint not found at {student_ckpt_path}")
        return

    student_model.eval()

    model_for_evaluation = ViTillCombined(
        encoder_feature_extractor=encoder_feature_extractor,
        vitill_decoder=student_model,
        effective_remove_class_token_config=False
    ).to(device)
    model_for_evaluation.eval()

    seg_model_input_channels = 2050
    seg_model_output_channels = 1
    
    segmentation_model = UNetSegmentationHead(
        in_channels=seg_model_input_channels,
        n_classes=seg_model_output_channels,
        target_size=(crop_size, crop_size)
    ).to(device)

    afm_ckpt_path = os.path.join(args.checkpoint_dir, "mvtec_afm_upsampler.pth")
    if os.path.exists(afm_ckpt_path):
        print(f"Loading AFM/Upsampler checkpoint from {afm_ckpt_path}")
        segmentation_model.load_state_dict(torch.load(afm_ckpt_path, map_location=device))
    else:
        print(f"Error: AFM checkpoint not found at {afm_ckpt_path}")
        return

    segmentation_model.eval()

    auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
    auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

    print("Starting evaluation...")

    with torch.no_grad():
        for item_name in item_list:
            test_path = os.path.join(args.data_path, item_name)
            if not os.path.exists(test_path):
                print(f"Warning: Data path for {item_name} not found, skipping.")
                continue

            test_dataset = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

            results = new_evaluation_batch(
                model=model_for_evaluation,
                model_2=segmentation_model,
                dataloader=test_dataloader,
                device=device,
                max_ratio=0.01,
                resize_mask=256
            )
            
            auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

            auroc_sp_list.append(auroc_sp)
            ap_sp_list.append(ap_sp)
            f1_sp_list.append(f1_sp)
            auroc_px_list.append(auroc_px)
            ap_px_list.append(ap_px)
            f1_px_list.append(f1_px)
            aupro_px_list.append(aupro_px)

            print(f'{item_name}: I-AUROC:{auroc_sp:.4f}, I-AP:{ap_sp:.4f}, I-F1:{f1_sp:.4f}, P-AUROC:{auroc_px:.4f}, P-AP:{ap_px:.4f}, P-F1:{f1_px:.4f}, P-AUPRO:{aupro_px:.4f}')

    print("-" * 50)
    print(f'Mean I-AUROC: {np.mean(auroc_sp_list):.4f}')
    print(f'Mean I-AP:    {np.mean(ap_sp_list):.4f}')
    print(f'Mean I-F1:    {np.mean(f1_sp_list):.4f}')
    print(f'Mean P-AUROC: {np.mean(auroc_px_list):.4f}')
    print(f'Mean P-AP:    {np.mean(ap_px_list):.4f}')
    print(f'Mean P-F1:    {np.mean(f1_px_list):.4f}')
    print(f'Mean P-AUPRO: {np.mean(aupro_px_list):.4f}')
    print("-" * 50)

if __name__ == '__main__':
    main()
#python ProAD_mvtec_test.py --data_path ./mvtec_anomaly_detection --checkpoint_dir ./checkpoints
