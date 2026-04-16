# src/segmentation/model_unet.py
# FedMedSeg Phase 2 — MobileNetV2-UNet Architecture
#
# ARCHITECTURE OVERVIEW:
#
#   Input (3, 224, 224)
#       │
#   ┌───┴──────────────────────────── ENCODER (MobileNetV2) ──────┐
#   │  Block 1  → Skip 1  (16ch,  112×112)                        │
#   │  Block 3  → Skip 2  (24ch,   56×56)                         │
#   │  Block 6  → Skip 3  (32ch,   28×28)                         │
#   │  Block 13 → Skip 4  (96ch,   14×14)                         │
#   │  Block 17 → Bottleneck (320ch, 7×7)                          │
#   └──────────────────────────────────────────────────────────────┘
#       │  (bottleneck features 7×7)
#   ┌───┴──────────────────────────── DECODER ────────────────────┐
#   │  Up-block 1: 7×7   → 14×14  + Skip 4  (96ch)  → 256ch      │
#   │  Up-block 2: 14×14 → 28×28  + Skip 3  (32ch)  → 128ch      │
#   │  Up-block 3: 28×28 → 56×56  + Skip 2  (24ch)  →  64ch      │
#   │  Up-block 4: 56×56 → 112×112+ Skip 1  (16ch)  →  32ch      │
#   │  Up-block 5: 112×112→224×224 (no skip)          →  16ch     │
#   └──────────────────────────────────────────────────────────────┘
#       │
#   1×1 Conv → Sigmoid → Binary Mask (1, 224, 224)
#
# Skip connections preserve fine spatial details lost during downsampling.
# Transposed Convolutions (ConvTranspose2d) perform learned upsampling.

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List


# ── Decoder Building Block ────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    A single upsampling block in the Decoder.

    Steps:
      1. Transposed Convolution  →  doubles the spatial resolution (×2 upsample)
      2. Concatenate with skip connection feature map
      3. Two Conv-BN-ReLU layers to refine the combined features

    Args:
        in_channels  (int): Channels coming from the previous decoder layer.
        skip_channels (int): Channels from the encoder skip connection.
        out_channels (int): Output channels after this block.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super(DecoderBlock, self).__init__()

        # Transposed convolution: halves channels, doubles spatial size
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

        # After concatenation, input channels = out_channels + skip_channels
        combined = out_channels + skip_channels

        self.refine = nn.Sequential(
            nn.Conv2d(combined, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = self.upsample(x)

        if skip is not None:
            # Handle potential size mismatch due to rounding in pooling
            if x.shape != skip.shape:
                x = nn.functional.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)   # Concatenate along channel dimension

        x = self.refine(x)
        return x


# ── Final Upsampling Block (no skip connection) ───────────────────────────────

class FinalBlock(nn.Module):
    """Last decoder block — upsamples to 224×224 with no skip connection."""

    def __init__(self, in_channels: int, out_channels: int):
        super(FinalBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Main Model ─────────────────────────────────────────────────────────────────

class MobileNetV2UNet(nn.Module):
    """
    MobileNetV2-UNet for binary pneumonia segmentation.

    Encoder: Pre-trained MobileNetV2 (ImageNet weights).
    Decoder: 5 upsampling blocks with skip connections.
    Output:  Binary mask of shape (B, 1, H, W) with values in [0, 1].

    Args:
        pretrained (bool): If True, loads ImageNet pre-trained MobileNetV2 encoder.
                           Default: True.
        freeze_encoder (bool): If True, freezes encoder weights initially.
                               Set to False for fine-tuning. Default: True.
    """

    # MobileNetV2 feature block indices and their output channels
    # These were determined by inspecting the MobileNetV2 architecture
    _SKIP_INDICES = {
        "skip1": (1,  16),   # After first inverted residual block — 112×112
        "skip2": (3,  24),   # After block 3 — 56×56
        "skip3": (6,  32),   # After block 6 — 28×28
        "skip4": (13, 96),   # After block 13 — 14×14
        "bottleneck": (17, 320),  # After block 17 — 7×7
    }

    def __init__(self, pretrained: bool = True, freeze_encoder: bool = True):
        super(MobileNetV2UNet, self).__init__()

        # ── Encoder: MobileNetV2 ──────────────────────────────────────────────
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)

        # We only need the 'features' part (not the classifier head)
        self.encoder_blocks = backbone.features   # nn.Sequential of 19 blocks

        if freeze_encoder:
            for param in self.encoder_blocks.parameters():
                param.requires_grad = False

        # ── Decoder ───────────────────────────────────────────────────────────
        # Each DecoderBlock(in_ch, skip_ch, out_ch)
        # in_ch   = channels from the previous decoder level
        # skip_ch = channels from the corresponding encoder skip connection
        # out_ch  = channels we want after this block

        self.decoder4 = DecoderBlock(320, 96,  256)   # 7×7   → 14×14
        self.decoder3 = DecoderBlock(256, 32,  128)   # 14×14 → 28×28
        self.decoder2 = DecoderBlock(128, 24,   64)   # 28×28 → 56×56
        self.decoder1 = DecoderBlock( 64, 16,   32)   # 56×56 → 112×112
        self.decoder0 = FinalBlock  ( 32,       16)   # 112×112→ 224×224

        # ── Segmentation Head ────────────────────────────────────────────────
        # 1×1 Conv maps 16 channels → 1 channel (binary mask probability)
        self.seg_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Input image batch. Shape: (B, 3, 224, 224)

        Returns:
            Tensor: Predicted mask probabilities. Shape: (B, 1, 224, 224)
        """
        # ── Encoder: extract feature maps at 5 resolutions ───────────────────
        skip1 = self.encoder_blocks[:2](x)    # (B,  16, 112, 112)
        skip2 = self.encoder_blocks[:4](x)    # (B,  24,  56,  56)
        skip3 = self.encoder_blocks[:7](x)    # (B,  32,  28,  28)
        skip4 = self.encoder_blocks[:14](x)   # (B,  96,  14,  14)
        bottleneck = self.encoder_blocks(x)   # (B, 320,   7,   7)  — full encoder
        # Note: encoder_blocks has 19 blocks (index 0-18), [:19] == full pass

        # ── Decoder: upsample + skip connections ──────────────────────────────
        d4 = self.decoder4(bottleneck, skip4)  # (B, 256, 14, 14)
        d3 = self.decoder3(d4, skip3)          # (B, 128, 28, 28)
        d2 = self.decoder2(d3, skip2)          # (B,  64, 56, 56)
        d1 = self.decoder1(d2, skip1)          # (B,  32, 112, 112)
        d0 = self.decoder0(d1)                 # (B,  16, 224, 224)

        # ── Segmentation head + sigmoid ───────────────────────────────────────
        logits = self.seg_head(d0)             # (B,   1, 224, 224)
        return torch.sigmoid(logits)            # probabilities in [0, 1]

    def unfreeze_encoder(self, num_blocks: int = 5):
        """
        Unfreeze the last `num_blocks` encoder blocks for fine-tuning.

        Call this after the initial training phase when the decoder has stabilized.
        Unfreezing allows the encoder to also adapt to the segmentation task.

        Args:
            num_blocks (int): Number of MobileNetV2 blocks (from the end) to unfreeze.
        """
        all_blocks = list(self.encoder_blocks.children())
        blocks_to_unfreeze = all_blocks[-num_blocks:]

        for block in blocks_to_unfreeze:
            for param in block.parameters():
                param.requires_grad = True

        total   = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Encoder] Unfroze last {num_blocks} blocks.")
        print(f"[Params]  Trainable: {trainable:,} / Total: {total:,}")


# ── Custom CNN Baselines ───────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Basic convolutional block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class CustomCNNUnet(nn.Module):
    """
    A customizable U-Net used for the "from scratch" baseline models.
    Supports depth=2 (Model 1) and depth=3 (Model 2).
    """
    def __init__(self, depth: int = 2):
        super().__init__()
        self.depth = depth
        
        # ── Encoder ─────────────────────────────
        self.enc1 = ConvBlock(3, 16)
        self.enc2 = ConvBlock(16, 32)
        if self.depth == 3:
            self.enc3 = ConvBlock(32, 64)
            self.bottleneck = ConvBlock(64, 128)
        else:
            self.bottleneck = ConvBlock(32, 64)
            
        self.pool = nn.MaxPool2d(2)
        
        # ── Decoder ─────────────────────────────
        if self.depth == 3:
            self.dec3 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
            self.dec2 = DecoderBlock(in_channels=64,  skip_channels=32, out_channels=32)
            self.dec1 = DecoderBlock(in_channels=32,  skip_channels=16, out_channels=16)
        else:
            self.dec2 = DecoderBlock(in_channels=64,  skip_channels=32, out_channels=32)
            self.dec1 = DecoderBlock(in_channels=32,  skip_channels=16, out_channels=16)
            
        self.seg_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.enc1(x)                # skip1
        p1 = self.pool(x1)
        
        x2 = self.enc2(p1)               # skip2
        p2 = self.pool(x2)
        
        if self.depth == 3:
            x3 = self.enc3(p2)           # skip3
            p3 = self.pool(x3)
            bot = self.bottleneck(p3)
            
            # Decoder
            d3 = self.dec3(bot, x3)
            d2 = self.dec2(d3, x2)
            d1 = self.dec1(d2, x1)
        else:
            bot = self.bottleneck(p2)
            
            # Decoder
            d2 = self.dec2(bot, x2)
            d1 = self.dec1(d2, x1)
            
        logits = self.seg_head(d1)
        return torch.sigmoid(logits)

    def unfreeze_encoder(self, num_blocks: int = 5):
        # This function exists purely for compatibility with the training loop
        pass



# ── Shape Assertion Test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    from segmentation.device_utils import get_device
    device = get_device("auto")

    model = MobileNetV2UNet(pretrained=True, freeze_encoder=True).to(device)

    # Count parameters
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Parameters:     {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"Frozen Parameters:    {total - trainable:,}  (MobileNetV2 encoder)")

    # ── SHAPE ASSERTION ───────────────────────────────────────────────────────
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\nInput  shape:  {dummy_input.shape}")
    print(f"Output shape:  {output.shape}")

    assert output.shape == (2, 1, 224, 224), \
        f"FAIL: Expected (2,1,224,224) but got {output.shape}"
    assert output.min() >= 0.0 and output.max() <= 1.0, \
        "FAIL: Output should be probabilities in [0, 1]"

    print("\n✓ Shape assertion passed:  (B, 1, 224, 224)")
    print("\n✓ Value range assertion:   [0, 1]")
    print("✓ MobileNetV2-UNet ready for training!")

    # Test Custom Models
    model1 = CustomCNNUnet(depth=2).to(device)
    model2 = CustomCNNUnet(depth=3).to(device)
    out1 = model1(dummy_input)
    out2 = model2(dummy_input)
    assert out1.shape == (2, 1, 224, 224)
    assert out2.shape == (2, 1, 224, 224)
    print("\n✓ CustomCNNUnet baselines ready for tracking!")
