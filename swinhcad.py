import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from .hcad_block import HCADBlock

class SwinHCAD(nn.Module):
    """
    SwinHCAD Architecture
    - Encoder: Swin Transformer (based on SwinUNETR)
    - Decoder: Hierarchical Channel-wise Attention Decoder (HCAD)
    """
    def __init__(self, img_size=(96, 96, 96), in_channels=4, out_channels=3, feature_size=48):
        super(SwinHCAD, self).__init__()
        
        # 1. Swin Transformer Encoder (Pre-trained backbone capable)
        self.swin_encoder = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=True,
            spatial_dims=3
        )
        
        # Feature dimensions at each stage (Standard SwinUNETR configs)
        # dims: [48, 96, 192, 384, 768] for feature_size=48
        dims = [feature_size * (2**i) for i in range(5)]

        # 2. HCAD Decoder Blocks
        # Bottleneck(Layer 4) -> Layer 3
        self.hcad4 = HCADBlock(dims[4], dims[3], dims[3])
        # Layer 3 -> Layer 2
        self.hcad3 = HCADBlock(dims[3], dims[2], dims[2])
        # Layer 2 -> Layer 1
        self.hcad2 = HCADBlock(dims[2], dims[1], dims[1])
        # Layer 1 -> Layer 0
        self.hcad1 = HCADBlock(dims[1], dims[0], dims[0])
        
        # 3. Final Segmentation Head
        self.final_conv = nn.Conv3d(dims[0], out_channels, kernel_size=1)

    def forward(self, x):
        # --- Encoder Path (Swin Transformer) ---
        # Extract hierarchical features using MONAI's SwinViT
        hidden_states = self.swin_encoder.swinViT(x, self.swin_encoder.normalize)
        
        # Reshape hidden states to 3D feature maps
        enc0 = self.swin_encoder.encoder1(x)                # Stage 0
        enc1 = self.swin_encoder.encoder2(hidden_states[0]) # Stage 1
        enc2 = self.swin_encoder.encoder3(hidden_states[1]) # Stage 2
        enc3 = self.swin_encoder.encoder4(hidden_states[2]) # Stage 3
        enc4 = self.swin_encoder.encoder10(hidden_states[4])# Bottleneck
        
        # --- Decoder Path (HCAD) ---
        dec3 = self.hcad4(enc4, enc3) # Fuse Bottleneck & Stage 3
        dec2 = self.hcad3(dec3, enc2) # Fuse & Stage 2
        dec1 = self.hcad2(dec2, enc1) # Fuse & Stage 1
        dec0 = self.hcad1(dec1, enc0) # Fuse & Stage 0
        
        # --- Output ---
        logits = self.final_conv(dec0)
        return logits