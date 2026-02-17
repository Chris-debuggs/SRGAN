"""
Loss Functions for SRGAN

Implements the three key losses:
1. Content Loss (L1/MSE) - Pixel-wise accuracy
2. Perceptual Loss (VGG) - High-level feature similarity
3. Adversarial Loss (GAN) - Realism from discriminator feedback
"""
import torch
import torch.nn as nn
import torchvision.models as models


class ContentLoss(nn.Module):
    """
    Content loss using L1 or MSE.
    Measures pixel-wise difference between generated and ground truth.
    """
    def __init__(self, loss_type='l1'):
        super(ContentLoss, self).__init__()
        if loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif loss_type == 'mse':
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
    def forward(self, generated, target):
        """
        Args:
            generated: Generated HR image [B, 3, H, W]
            target: Ground truth HR image [B, 3, H, W]
        
        Returns:
            Content loss scalar
        """
        return self.loss(generated, target)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features.
    Compares high-level features instead of raw pixels.
    Uses pre-trained VGG19 network (frozen).
    """
    def __init__(self, layer='relu5_4'):
        super(PerceptualLoss, self).__init__()
        
        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True).features
        
        # Map layer names to indices
        layer_map = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_4': 17,
            'relu4_4': 26,
            'relu5_4': 35,
        }
        
        if layer not in layer_map:
            raise ValueError(f"Unknown layer: {layer}. Choose from {list(layer_map.keys())}")
        
        # Extract features up to specified layer
        layer_idx = layer_map[layer]
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:layer_idx + 1])
        
        # Freeze VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Move to eval mode
        self.feature_extractor.eval()
        
        # Loss function
        self.mse_loss = nn.MSELoss()
        
        # VGG normalization (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def normalize(self, x):
        """Normalize image for VGG (ImageNet normalization)"""
        # Assume input is in [0, 1] range
        x = (x - self.mean) / self.std
        return x
    
    def forward(self, generated, target):
        """
        Args:
            generated: Generated HR image [B, 3, H, W] in [0, 1]
            target: Ground truth HR image [B, 3, H, W] in [0, 1]
        
        Returns:
            Perceptual loss scalar
        """
        # Normalize inputs
        generated_norm = self.normalize(generated)
        target_norm = self.normalize(target)
        
        # Extract features
        generated_features = self.feature_extractor(generated_norm)
        target_features = self.feature_extractor(target_norm)
        
        # Compute MSE between features
        loss = self.mse_loss(generated_features, target_features)
        
        return loss


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for GAN training.
    Uses Binary Cross-Entropy.
    """
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, discriminator_output, is_real):
        """
        Args:
            discriminator_output: Discriminator predictions [B, 1]
            is_real: Boolean, whether labels should be real (True) or fake (False)
        
        Returns:
            Adversarial loss scalar
        """
        batch_size = discriminator_output.size(0)
        device = discriminator_output.device
        
        # Create labels
        if is_real:
            # Real labels (1.0)
            labels = torch.ones(batch_size, 1, device=device)
        else:
            # Fake labels (0.0)
            labels = torch.zeros(batch_size, 1, device=device)
        
        # Compute BCE loss
        loss = self.bce_loss(discriminator_output, labels)
        
        return loss


class SRGANLoss(nn.Module):
    """
    Combined SRGAN loss.
    L_total = λ_content * L_content + λ_perceptual * L_perceptual + λ_adv * L_adversarial
    """
    def __init__(self, 
                 lambda_content=1.0, 
                 lambda_perceptual=0.006, 
                 lambda_adversarial=0.001,
                 content_loss_type='l1',
                 vgg_layer='relu5_4'):
        super(SRGANLoss, self).__init__()
        
        self.lambda_content = lambda_content
        self.lambda_perceptual = lambda_perceptual
        self.lambda_adversarial = lambda_adversarial
        
        # Initialize loss components
        self.content_loss = ContentLoss(content_loss_type)
        self.perceptual_loss = PerceptualLoss(vgg_layer)
        self.adversarial_loss = AdversarialLoss()
        
    def forward(self, generated, target, discriminator_output_fake):
        """
        Compute combined generator loss
        
        Args:
            generated: Generated HR image [B, 3, H, W]
            target: Ground truth HR image [B, 3, H, W]
            discriminator_output_fake: Discriminator predictions on generated images
        
        Returns:
            tuple: (total_loss, loss_dict)
        """
        # Content loss
        l_content = self.content_loss(generated, target)
        
        # Perceptual loss
        # Ensure images are in [0, 1] range
        generated_clipped = torch.clamp(generated, 0, 1)
        target_clipped = torch.clamp(target, 0, 1)
        l_perceptual = self.perceptual_loss(generated_clipped, target_clipped)
        
        # Adversarial loss (fool discriminator)
        l_adversarial = self.adversarial_loss(discriminator_output_fake, is_real=True)
        
        # Combined loss
        total_loss = (self.lambda_content * l_content + 
                     self.lambda_perceptual * l_perceptual + 
                     self.lambda_adversarial * l_adversarial)
        
        # Return loss breakdown for logging
        loss_dict = {
            'total': total_loss.item(),
            'content': l_content.item(),
            'perceptual': l_perceptual.item(),
            'adversarial': l_adversarial.item(),
        }
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # Test loss functions
    print("Testing SRGAN Loss Functions...")
    
    # Dummy data
    batch_size = 4
    hr_size = 96
    generated = torch.rand(batch_size, 3, hr_size, hr_size)
    target = torch.rand(batch_size, 3, hr_size, hr_size)
    disc_output = torch.rand(batch_size, 1)
    
    # Test content loss
    content_loss = ContentLoss('l1')
    l_content = content_loss(generated, target)
    print(f"✓ Content Loss: {l_content.item():.6f}")
    
    # Test perceptual loss
    print("Loading VGG19 for perceptual loss...")
    perceptual_loss = PerceptualLoss('relu5_4')
    l_perceptual = perceptual_loss(generated, target)
    print(f"✓ Perceptual Loss: {l_perceptual.item():.6f}")
    
    # Test adversarial loss
    adversarial_loss = AdversarialLoss()
    l_adv_real = adversarial_loss(disc_output, is_real=True)
    l_adv_fake = adversarial_loss(disc_output, is_real=False)
    print(f"✓ Adversarial Loss (real): {l_adv_real.item():.6f}")
    print(f"✓ Adversarial Loss (fake): {l_adv_fake.item():.6f}")
    
    # Test combined loss
    combined_loss = SRGANLoss()
    total_loss, loss_dict = combined_loss(generated, target, disc_output)
    print(f"✓ Combined Loss: {loss_dict}")
    
    print("\nAll loss tests passed!")
