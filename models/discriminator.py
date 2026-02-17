"""
SRGAN Discriminator Architecture

The discriminator is a binary classifier that distinguishes real HR images from generated ones.
Architecture: VGG-style with strided convolutions for downsampling.
"""
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    SRGAN Discriminator Network
    
    Binary classifier that outputs probability of input being a real HR image.
    Uses VGG-style architecture with strided convolutions.
    
    Args:
        input_shape (tuple): Input image shape (height, width)
    """
    def __init__(self, input_shape=(96, 96)):
        super(Discriminator, self).__init__()
        
        self.input_shape = input_shape
        
        def discriminator_block(in_channels, out_channels, stride=1, bn=True):
            """Helper to create conv block"""
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        # Convolutional layers
        layers = []
        in_channels = 3
        
        # Block 1: 64 filters
        layers.extend(discriminator_block(in_channels, 64, stride=1, bn=False))
        layers.extend(discriminator_block(64, 64, stride=2, bn=True))
        
        # Block 2: 128 filters
        layers.extend(discriminator_block(64, 128, stride=1, bn=True))
        layers.extend(discriminator_block(128, 128, stride=2, bn=True))
        
        # Block 3: 256 filters
        layers.extend(discriminator_block(128, 256, stride=1, bn=True))
        layers.extend(discriminator_block(256, 256, stride=2, bn=True))
        
        # Block 4: 512 filters
        layers.extend(discriminator_block(256, 512, stride=1, bn=True))
        layers.extend(discriminator_block(512, 512, stride=2, bn=True))
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Calculate the size after convolutions
        # Each stride=2 reduces size by 2, we have 4 such layers
        # So: input_size / (2^4) = input_size / 16
        conv_output_size = input_shape[0] // 16
        
        # Adaptive pooling to handle various input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Image tensor [B, 3, H, W]
        
        Returns:
            Probability tensor [B, 1] - probability of being real
        """
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Adaptive pooling
        pooled = self.adaptive_pool(features)
        
        # Flatten
        flattened = pooled.view(pooled.size(0), -1)
        
        # Classification
        output = self.classifier(flattened)
        
        return output
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    # Test the discriminator
    print("Testing SRGAN Discriminator...")
    
    # Create model
    model = Discriminator(input_shape=(96, 96))
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    
    # Test with dummy input (HR images)
    batch_size = 4
    hr_size = 96
    dummy_input = torch.randn(batch_size, 3, hr_size, hr_size)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values (probabilities): {output.squeeze()}")
    
    # Verify output
    assert output.shape == (batch_size, 1), "Output shape mismatch!"
    assert (output >= 0).all() and (output <= 1).all(), "Output not in [0, 1] range!"
    
    print("✓ Discriminator test passed!")
