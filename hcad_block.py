import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class HCADBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):
        super(HCADBlock, self).__init__()
        
        # Upsampling layer
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        # Feature Fusion Convolution
        self.conv_fusion = nn.Sequential(
            nn.Conv3d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        # Core Novelty: Channel Attention
        self.att = ChannelAttention(out_channels)
        
        # Residual Connection Block
        self.conv_res = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, skip):
        """
        x: Lower resolution feature from decoder
        skip: Higher resolution feature from encoder (Skip Connection)
        """
        x_up = self.up(x)
        
        # Padding handling (if sizes don't match exactly due to pooling)
        if x_up.shape != skip.shape:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='trilinear', align_corners=True)
            
        # Concatenate & Fuse
        fused = torch.cat([x_up, skip], dim=1)
        fused = self.conv_fusion(fused)
        
        # Apply Attention
        att_feat = self.att(fused)
        
        # Residual Add
        out = self.conv_res(att_feat) + att_feat
        return out