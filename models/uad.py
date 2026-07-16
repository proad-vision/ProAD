import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from sklearn.cluster import KMeans
import math
import warnings


from functools import partial  # 需要确保导入


# 从您的 ViTill 类中提取的辅助方法，或者让它们成为新类的方法
def fuse_feature_static(feat_list):
    if not feat_list:
        return torch.empty(0)  # 或者根据需要处理空列表
    return torch.stack(feat_list, dim=1).mean(dim=1)


def generate_mask_static(feature_size, device, remove_class_token_flag, num_register_tokens, mask_neighbor_size):
    """
    Generate a square mask for the sequence.
    remove_class_token_flag: Indicates if the sequence for which mask is generated already has class token removed.
    """
    if mask_neighbor_size <= 0:
        return None

    h, w = feature_size, feature_size
    hm, wm = mask_neighbor_size, mask_neighbor_size
    mask = torch.ones(h, w, h, w, device=device)
    for idx_h1 in range(h):
        for idx_w1 in range(w):
            idx_h2_start = max(idx_h1 - hm // 2, 0)
            idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
            idx_w2_start = max(idx_w1 - wm // 2, 0)
            idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
            mask[
            idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
            ] = 0
    mask = mask.view(h * w, h * w)

    # The mask from original ViTill applies to the sequence *after* cls/reg tokens might be present or absent.
    # If remove_class_token_flag is True, it means the sequence (h*w) is purely spatial.
    # If remove_class_token_flag is False, it means the sequence (h*w) is accompanied by cls/reg tokens.
    if remove_class_token_flag:  # Mask is for spatial tokens only
        return mask
    else:  # Mask needs to account for class and register tokens if they are part of the sequence fed to decoder
        # The original ViTill's attn_mask is passed to decoder blocks.
        # If decoder sequence includes CLS/REG tokens, mask should be larger.
        # Based on original ViTill, if self.remove_class_token is False, the sequence includes these tokens.
        mask_all = torch.ones(h * w + 1 + num_register_tokens,
                              h * w + 1 + num_register_tokens, device=device)
        # Non-spatial tokens attend to all
        # Spatial tokens attend based on 'mask'
        mask_all[1 + num_register_tokens:, 1 + num_register_tokens:] = mask
        return mask_all


class EncoderFeatureExtractor(nn.Module):
    def __init__(self, encoder_instance, target_layers, fuse_layer_encoder,
                 # remove_class_token_for_en_output: Original ViTill always strips 'en' output.
                 # So this parameter is not strictly needed if we follow that fixed behavior.
                 # For clarity, let's assume 'en' output is always stripped.
                 encoder_require_grad_layer=[]  # From original ViTill, if specific layers need grad
                 ):
        super().__init__()
        self.encoder = encoder_instance  # This is the pre-trained encoder model/module
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        self.num_register_tokens = self.encoder.num_register_tokens
        self.encoder_require_grad_layer = encoder_require_grad_layer

    def forward(self, x_img, remove_tokens_for_bottleneck_input_flag):
        # This model's parameters are not trained.
        # Make sure computations here do not affect gradients for the main encoder,
        # if the main encoder is part of a larger trainable graph elsewhere (not the case here).
        # For this setup, original_encoder is frozen, so no_grad() context is implicitly managed.

        # --- Encoder Pass ---
        # Using with torch.no_grad() if self.encoder is truly frozen and not part of any backprop path
        # However, original ViTill selectively applied no_grad to encoder blocks.
        # We replicate that selective grad logic for `x_img` processing if layers are specified.
        # If encoder_require_grad_layer is empty, all encoder blocks run under current grad mode (e.g. no_grad if this whole module is in eval).
        # For this specific split, this EncoderFeatureExtractor is *never* trained.
        # So, its forward pass can effectively be under torch.no_grad().

        with torch.no_grad():  # Ensure this part does not contribute to gradients
            x_prepared = self.encoder.prepare_tokens(x_img)
            en_list_raw = []
            current_x = x_prepared
            for i, blk in enumerate(self.encoder.blocks):
                if i <= self.target_layers[-1]:
                    # Original ViTill had selective no_grad here. Since this model is not trained,
                    # and the original encoder is used by reference, we must ensure that if the
                    # original encoder *is* being trained elsewhere (not in this script's setup),
                    # its gradients are not affected unexpectedly.
                    # For this script's purpose (encoder is frozen), simple forward pass is fine.
                    # If encoder_require_grad_layer was meant for the original ViTill's training,
                    # it's not directly applicable here as this module isn't trained.
                    # We assume the encoder instance itself is globally frozen.
                    current_x = blk(current_x)
                else:
                    continue  # Optimization: stop iterating if past last target layer
                if i in self.target_layers:
                    en_list_raw.append(current_x)

        if not en_list_raw:
            raise ValueError("en_list_raw is empty. Check target_layers and encoder configuration.")

        side = int(math.sqrt(en_list_raw[0].shape[1] - 1 - self.num_register_tokens))

        # 1. Prepare `x_to_bottleneck` (input for the ViTillDecoder part)
        _en_list_for_bottleneck_processing = en_list_raw
        if remove_tokens_for_bottleneck_input_flag:  # This flag is ViTill's original self.remove_class_token
            _en_list_for_bottleneck_processing = [
                e[:, 1 + self.num_register_tokens:, :] for e in en_list_raw
            ]
        x_to_bottleneck = fuse_feature_static(_en_list_for_bottleneck_processing)

        # 2. Prepare `en_output_reshaped` (this is the `en` for loss/evaluation)
        #    Original ViTill:
        #    `en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]`
        #       (here en_list could be raw or stripped based on self.remove_class_token)
        #    `if not self.remove_class_token: en = [e[:, 1+num_reg:,:] ... for e in en]`
        #    This means 'en' output is ALWAYS stripped of class/register tokens before reshaping.

        #    To match: if remove_tokens_for_bottleneck_input_flag was True, en_list_raw was effectively stripped for 'en' calculation's input too.
        #    If remove_tokens_for_bottleneck_input_flag was False, en_list_raw was raw, then stripped.
        #    So, use en_list_raw for fusion, then always strip.

        en_fused_from_raw = [
            fuse_feature_static([en_list_raw[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder
        ]

        # Always strip tokens for the final 'en' output before reshaping
        en_output_fused_stripped = [
            e[:, 1 + self.num_register_tokens:, :] for e in en_fused_from_raw
        ]

        en_output_reshaped = [
            e.permute(0, 2, 1).reshape([x_img.shape[0], -1, side, side]).contiguous()
            for e in en_output_fused_stripped
        ]

        return x_to_bottleneck, en_output_reshaped, side


class ViTillDecoder(nn.Module):
    def __init__(
            self,
            bottleneck,
            decoder,
            fuse_layer_decoder,
            mask_neighbor_size,
            # This is the original ViTill's remove_class_token config value
            # It determines if the decoder operates on sequences with or without class/reg tokens
            # and how the final 'de' is processed.
            effective_remove_class_token_config,
            num_register_tokens  # From the encoder
    ) -> None:
        super().__init__()
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.mask_neighbor_size = mask_neighbor_size
        self.effective_remove_class_token_config = effective_remove_class_token_config
        self.num_register_tokens = num_register_tokens

    def forward(self, x_from_encoder, batch_size_for_reshape, side_for_reshape):
        # x_from_encoder has class/register tokens if effective_remove_class_token_config is False
        # and is stripped if effective_remove_class_token_config is True.

        x = x_from_encoder
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        attn_mask = None
        if self.mask_neighbor_size > 0:
            # The generate_mask_static's remove_class_token_flag should be True if 'x' (decoder input) is spatial only.
            # This happens if self.effective_remove_class_token_config is True.
            # If self.effective_remove_class_token_config is False, 'x' has tokens, so flag for generate_mask is False.
            mask_for_spatial_part_only = self.effective_remove_class_token_config
            attn_mask = generate_mask_static(
                feature_size=side_for_reshape,
                device=x.device,
                remove_class_token_flag=mask_for_spatial_part_only,
                # This tells mask generator about sequence structure
                num_register_tokens=self.num_register_tokens,
                mask_neighbor_size=self.mask_neighbor_size
            )
            # Correction: The original generate_mask in ViTill uses self.remove_class_token.
            # This flag dictates if the *mask itself* should be for a sequence with cls tokens (mask_all) or not.
            # The sequence `x` fed to decoder blocks will have tokens if `effective_remove_class_token_config` is False.
            # The mask's structure should correspond to `x`.
            # So `generate_mask_static`'s `remove_class_token_flag` is `self.effective_remove_class_token_config`.
            attn_mask = generate_mask_static(
                feature_size=side_for_reshape,
                device=x.device,
                remove_class_token_flag=self.effective_remove_class_token_config,
                # if True, x is spatial, mask is for spatial. If False, x has tokens, mask is mask_all.
                num_register_toke48ns=self.num_register_tokens,
                mask_neighbor_size=self.mask_neighbor_size
            )

        de_list = []
        for i, blk in enumerate(self.decoder):
            # Ensure decoder blocks (e.g. LinearAttention2) can handle attn_mask if provided
            # Some attention implementations might not take attn_mask, or take it with a different name
            try:
                x = blk(x, attn_mask=attn_mask)
            except TypeError:  # If the block doesn't accept attn_mask
                warnings.warn(f"Block {type(blk).__name__} at index {i} does not accept attn_mask. Passing without it.")
                x = blk(x)
            de_list.append(x)
        de_list = de_list[::-1]  # Original ViTill reverses decoder output before fusion

        de_fused = [fuse_feature_static([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        # Original ViTill 'de' output processing:
        # `if not self.remove_class_token: de = [d[:, 1+num_reg:,:] ... for d in de]`
        # This means 'de' output is ALWAYS stripped if self.remove_class_token is False.
        # If self.remove_class_token is True, 'de' (from fused, possibly already stripped list) is used as is.
        # This implies: if effective_remove_class_token_config is True, de_fused items are already from stripped sequences.
        #             if effective_remove_class_token_config is False, de_fused items may contain tokens and need stripping.
        # So, 'de' is ALWAYS STRIPPED before reshape, just like 'en'.

        de_output_fused_stripped = [
            d[:, 1 + self.num_register_tokens:, :] if d.shape[1] > (side_for_reshape * side_for_reshape) else d
            # ensure stripping only if tokens are there
            for d in de_fused
        ]
        # A more robust way to strip, assuming de_fused elements have tokens if not effective_remove_class_token_config:
        if not self.effective_remove_class_token_config:  # If tokens were kept through decoder
            de_final_processed = [d[:, 1 + self.num_register_tokens:, :] for d in de_fused]
        else:  # Tokens were already removed before bottleneck/decoder
            de_final_processed = de_fused

        # Given the logic for 'en' output (always stripped), 'de' should also always be stripped for consistency.
        # The `if not self.remove_class_token:` implies that if tokens *were* present (i.e., `remove_class_token` is false), they get stripped.
        # If tokens were *not* present (i.e., `remove_class_token` is true), this step is skipped, and `de_fused` is already stripped.
        # So, effectively, the output `de` is always stripped.

        de_final_payload = []
        for d_tensor in de_fused:  # Iterate over list of tensors
            # Check if stripping is necessary/possible
            # Stripping occurs if class/register tokens were part of the sequence processed by the decoder.
            # This is the case if self.effective_remove_class_token_config is False.
            if not self.effective_remove_class_token_config and d_tensor.shape[1] > side_for_reshape * side_for_reshape:
                de_final_payload.append(d_tensor[:, 1 + self.num_register_tokens:, :])
            else:  # Already stripped or never had them in a way that requires this specific stripping
                de_final_payload.append(d_tensor)

        de_output_reshaped = [
            d.permute(0, 2, 1).reshape([batch_size_for_reshape, -1, side_for_reshape, side_for_reshape]).contiguous()
            for d in de_final_payload  # Corrected: use de_final_payload
        ]
        return de_output_reshaped


class ViTillCombined(nn.Module):
    """
    A wrapper model for evaluation purposes to mimic the original ViTill(img) -> (en, de) output.
    This model itself is not trained.
    """

    def __init__(self, encoder_feature_extractor, vitill_decoder,
                 # This is the original ViTill's remove_class_token config
                 effective_remove_class_token_config):
        super().__init__()
        self.encoder_extractor = encoder_feature_extractor
        self.decoder_processor = vitill_decoder
        self.effective_remove_class_token_config = effective_remove_class_token_config

    def forward(self, img):
        # The remove_tokens_for_bottleneck_input_flag for encoder_extractor
        # should be the same as effective_remove_class_token_config
        x_for_bottleneck, en_output, side = self.encoder_extractor(
            img,
            remove_tokens_for_bottleneck_input_flag=self.effective_remove_class_token_config
        )

        de_output = self.decoder_processor(
            x_for_bottleneck,
            img.shape[0],  # batch_size
            side
        )
        return en_output, de_output


class ViTill(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            mask_neighbor_size=0,
            remove_class_token=False,
            encoder_require_grad_layer=[],
    ) -> None:
        super(ViTill, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        x = self.fuse_feature(en_list)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        if self.mask_neighbor_size > 0:
            attn_mask = self.generate_mask(side, x.device)
        else:
            attn_mask = None

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def generate_mask(self, feature_size, device='cuda'):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h, w = feature_size, feature_size
        hm, wm = self.mask_neighbor_size, self.mask_neighbor_size
        mask = torch.ones(h, w, h, w, device=device)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        if self.remove_class_token:
            return mask
        mask_all = torch.ones(h * w + 1 + self.encoder.num_register_tokens,
                              h * w + 1 + self.encoder.num_register_tokens, device=device)
        mask_all[1 + self.encoder.num_register_tokens:, 1 + self.encoder.num_register_tokens:] = mask
        return mask_all


class ViTillCat(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[1, 3, 5, 7],
            mask_neighbor_size=0,
            remove_class_token=False,
            encoder_require_grad_layer=[],
    ) -> None:
        super(ViTillCat, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        x = self.fuse_feature(en_list)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        for i, blk in enumerate(self.decoder):
            x = blk(x)

        en = [torch.cat([en_list[idx] for idx in self.fuse_layer_encoder], dim=2)]
        de = [x]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

class ViTAD(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 5, 8, 11],
            fuse_layer_encoder=[0, 1, 2],
            fuse_layer_decoder=[2, 5, 8],
            mask_neighbor_size=0,
            remove_class_token=False,
    ) -> None:
        super(ViTAD, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                with torch.no_grad():
                    x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]
            x = x[:, 1 + self.encoder.num_register_tokens:, :]

        # x = torch.cat(en_list, dim=2)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        if self.mask_neighbor_size > 0:
            attn_mask = self.generate_mask(side, x.device)
        else:
            attn_mask = None

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [en_list[idx] for idx in self.fuse_layer_encoder]
        de = [de_list[idx] for idx in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de


class ViTillv2(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7]
    ) -> None:
        super(ViTillv2, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                with torch.no_grad():
                    x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en.append(x)

        x = self.fuse_feature(en)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        de = []
        for i, blk in enumerate(self.decoder):
            x = blk(x)
            de.append(x)

        side = int(math.sqrt(x.shape[1]))

        en = [e[:, self.encoder.num_register_tokens + 1:, :] for e in en]
        de = [d[:, self.encoder.num_register_tokens + 1:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]

        return en[::-1], de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)


class ViTillv3(nn.Module):
    def __init__(
            self,
            teacher,
            student,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_dropout=0.,
    ) -> None:
        super(ViTillv3, self).__init__()
        self.teacher = teacher
        self.student = student
        if fuse_dropout > 0:
            self.fuse_dropout = nn.Dropout(fuse_dropout)
        else:
            self.fuse_dropout = nn.Identity()
        self.target_layers = target_layers
        if not hasattr(self.teacher, 'num_register_tokens'):
            self.teacher.num_register_tokens = 0

    def forward(self, x):
        with torch.no_grad():
            patch = self.teacher.prepare_tokens(x)
            x = patch
            en = []
            for i, blk in enumerate(self.teacher.blocks):
                if i <= self.target_layers[-1]:
                    x = blk(x)
                else:
                    continue
                if i in self.target_layers:
                    en.append(x)
            en = self.fuse_feature(en, fuse_dropout=False)

        x = patch
        de = []
        for i, blk in enumerate(self.student):
            x = blk(x)
            if i in self.target_layers:
                de.append(x)
        de = self.fuse_feature(de, fuse_dropout=False)

        en = en[:, 1 + self.teacher.num_register_tokens:, :]
        de = de[:, 1 + self.teacher.num_register_tokens:, :]
        side = int(math.sqrt(en.shape[1]))

        en = en.permute(0, 2, 1).reshape([x.shape[0], -1, side, side])
        de = de.permute(0, 2, 1).reshape([x.shape[0], -1, side, side])
        return [en.contiguous()], [de.contiguous()]

    def fuse_feature(self, feat_list, fuse_dropout=False):
        if fuse_dropout:
            feat = torch.stack(feat_list, dim=1)
            feat = self.fuse_dropout(feat).mean(dim=1)
            return feat
        else:
            return torch.stack(feat_list, dim=1).mean(dim=1)


class ReContrast(nn.Module):
    def __init__(
            self,
            encoder,
            encoder_freeze,
            bottleneck,
            decoder,
    ) -> None:
        super(ReContrast, self).__init__()
        self.encoder = encoder
        self.encoder.layer4 = None
        self.encoder.fc = None

        self.encoder_freeze = encoder_freeze
        self.encoder_freeze.layer4 = None
        self.encoder_freeze.fc = None

        self.bottleneck = bottleneck
        self.decoder = decoder

    def forward(self, x):
        en = self.encoder(x)
        with torch.no_grad():
            en_freeze = self.encoder_freeze(x)
        en_2 = [torch.cat([a, b], dim=0) for a, b in zip(en, en_freeze)]
        de = self.decoder(self.bottleneck(en_2))
        de = [a.chunk(dim=0, chunks=2) for a in de]
        de = [de[0][0], de[1][0], de[2][0], de[3][1], de[4][1], de[5][1]]
        return en_freeze + en, de

    def train(self, mode=True, encoder_bn_train=True):
        self.training = mode
        if mode is True:
            if encoder_bn_train:
                self.encoder.train(True)
            else:
                self.encoder.train(False)
            self.encoder_freeze.train(False)  # the frozen encoder is eval()
            self.bottleneck.train(True)
            self.decoder.train(True)
        else:
            self.encoder.train(False)
            self.encoder_freeze.train(False)
            self.bottleneck.train(False)
            self.decoder.train(False)
        return self


def update_moving_average(ma_model, current_model, momentum=0.99):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = update_average(old_weight, up_weight)

    for current_buffers, ma_buffers in zip(current_model.buffers(), ma_model.buffers()):
        old_buffer, up_buffer = ma_buffers.data, current_buffers.data
        ma_buffers.data = update_average(old_buffer, up_buffer, momentum)


def update_average(old, new, momentum=0.99):
    if old is None:
        return new
    return old * momentum + (1 - momentum) * new


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)
