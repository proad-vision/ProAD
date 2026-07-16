import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.se = SELayer(out_channels, reduction)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2, stride=2), 
            SEResidualBlock(in_channels, out_channels))
    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 != 0 or w % 2 != 0:
            pad_w = 1 if w % 2 != 0 else 0
            pad_h = 1 if h % 2 != 0 else 0
            x = F.pad(x, [0, pad_w, 0, pad_h])
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, ch_g, ch_x, ch_out):
        super().__init__()
        up_out_ch = ch_g // 2
        self.up = nn.ConvTranspose2d(ch_g, up_out_ch, kernel_size=2, stride=2)
        self.att = AttentionGate(F_g=up_out_ch, F_l=ch_x, F_int=ch_g // 4 if ch_g > 4 else 1)
        self.conv = SEResidualBlock(up_out_ch + ch_x, ch_out)
    def forward(self, g, x):
        g_up = self.up(g)
        if g_up.shape[2:] != x.shape[2:]:
            g_up = F.interpolate(g_up, size=x.shape[2:], mode='bilinear', align_corners=True)
        x_att = self.att(g=g_up, x=x)
        x_cat = torch.cat([g_up, x_att], dim=1)
        return self.conv(x_cat)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output
        x = x + self.ffn(self.norm2(x))
        return x

class TransformerBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, num_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.proj_in = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(dim=in_channels, num_heads=num_heads, mlp_dim=in_channels * 4)
            for _ in range(num_layers)
        ])
        
        self.pos_embed = nn.Parameter(torch.randn(1, 16, in_channels))
        
        self.proj_out = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_proj = self.proj_in(x)
        
        x_flat = x_proj.flatten(2).transpose(1, 2)
        
        pos_embed = self.pos_embed[:, :h*w, :]
        x_flat = x_flat + pos_embed
        
        for layer in self.transformer_layers:
            x_flat = layer(x_flat)
        
        x_reshaped = x_flat.transpose(1, 2).reshape(b, c, h, w)
        
        output = self.proj_out(x_reshaped)
        return output
        
class UNetSegmentationHead(nn.Module):
    def __init__(self, in_channels=2050, n_classes=1, target_size=(392, 392)):
        super(UNetSegmentationHead, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.target_size = target_size

        self.inc = SEResidualBlock(in_channels, 96)
        self.down1 = Down(96, 192)
        self.down2 = Down(192, 384)
        self.down3 = Down(384, 768)
        
        self.bottleneck = TransformerBottleneck(
            in_channels=768, 
            out_channels=1024, 
            num_layers=4,
            num_heads=16
        )

        #self.up1 = Up(ch_g=1024, ch_x=768, ch_out=512)
        self.up2 = Up(ch_g=1024, ch_x=384, ch_out=512)
        self.up3 = Up(ch_g=512, ch_x=192, ch_out=256)
        self.up4 = Up(ch_g=256, ch_x=96, ch_out=128)

        self.up_final_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            SEResidualBlock(128, 64))
        self.up_final_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            SEResidualBlock(64, 32))
        self.up_final_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            SEResidualBlock(32, 16))
        self.up_to_target_size = nn.Upsample(size=target_size, mode='bilinear', align_corners=True)

        self.outc = nn.Sequential(
            SEResidualBlock(16, 16),
            OutConv(16, n_classes),
            nn.Sigmoid())

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        bn = self.bottleneck(x4)
        
        #out_up1 = self.up1(g=bn, x=x4)
        out_up2 = self.up2(g=bn, x=x3)
        out_up3 = self.up3(g=out_up2, x=x2)
        out_up4 = self.up4(g=out_up3, x=x1)

        f1 = self.up_final_1(out_up4)
        f2 = self.up_final_2(f1)
        f3 = self.up_final_3(f2)
        
        f_target = self.up_to_target_size(f3)
        output_mask = self.outc(f_target)
        return output_mask