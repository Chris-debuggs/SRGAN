"""
SRGAN Generator Architecture

The generator uses residual learning with skip connections to upscale images.
Architecture: Conv → Residual Blocks → Upsampling → Conv
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers.
    Uses skip connection to ease gradient flow.
    """
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # Skip connection
        out = out + residual
        return out


class UpsampleBlock(nn.Module):
    """
    Upsampling block using PixelShuffle (sub-pixel convolution).
    Upsamples by factor of 2. Stack multiple for 4× upsampling.
    """
    def __init__(self, in_channels, upscale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (upscale_factor ** 2), 
                             kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.prelu = nn.PReLU()
        
    def forward(self, x):
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.prelu(out)
        return out


class Generator(nn.Module):
    """
    SRGAN Generator Network
    
    Takes low-resolution image and outputs high-resolution image.
    Uses residual blocks for feature extraction and PixelShuffle for upsampling.
    
    Args:
        scale_factor (int): Upsampling factor (2 or 4)
        num_residual_blocks (int): Number of residual blocks (default: 16)
        num_channels (int): Number of feature channels (default: 64)
    """
    def __init__(self, scale_factor=4, num_residual_blocks=16, num_channels=64):
        super(Generator, self).__init__()
        assert scale_factor in [2, 4], "Scale factor must be 2 or 4"
        
        self.scale_factor = scale_factor
        
        # Initial feature extraction
        self.conv_input = nn.Conv2d(3, num_channels, kernel_size=9, padding=4)
        self.prelu_input = nn.PReLU()
        
        # Residual blocks
        residual_blocks = []
        for _ in range(num_residual_blocks):
            residual_blocks.append(ResidualBlock(num_channels))
        self.residual_blocks = nn.Sequential(*residual_blocks)
        
        # Post-residual conv
        self.conv_mid = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(num_channels)
        
        # Upsampling blocks
        upsample_blocks = []
        if scale_factor == 2:
            upsample_blocks.append(UpsampleBlock(num_channels, 2))
        elif scale_factor == 4:
            upsample_blocks.append(UpsampleBlock(num_channels, 2))
            upsample_blocks.append(UpsampleBlock(num_channels, 2))
        self.upsample_blocks = nn.Sequential(*upsample_blocks)
        
        # Output layer
        self.conv_output = nn.Conv2d(num_channels, 3, kernel_size=9, padding=4)
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Low-resolution image tensor [B, 3, H, W]
        
        Returns:
            High-resolution image tensor [B, 3, H*scale, W*scale]
        """
        # Initial feature extraction
        out = self.prelu_input(self.conv_input(x))
        residual = out
        
        # Pass through residual blocks
        out = self.residual_blocks(out)
        
        # Post-residual processing
        out = self.bn_mid(self.conv_mid(out))
        
        # Add skip connection from initial features
        out = out + residual
        
        # Upsampling
        out = self.upsample_blocks(out)
        
        # Final output
        out = self.conv_output(out)
        
        return out
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    # Test the generator
    print("Testing SRGAN Generator...")
    
    # Create model
    model = Generator(scale_factor=4, num_residual_blocks=16)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    
    # Test with dummy input
    batch_size = 4
    lr_size = 24  # Low-res image size
    dummy_input = torch.randn(batch_size, 3, lr_size, lr_size)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, 3, {lr_size * 4}, {lr_size * 4})")
    
    # Verify output size
    assert output.shape == (batch_size, 3, lr_size * 4, lr_size * 4), "Output shape mismatch!"
    print("✓ Generator test passed!")
