"""
Image utility functions for converting between tensors and images.
"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a PIL Image.
    
    Args:
        tensor: Image tensor [3, H, W] or [B, 3, H, W] in [0, 1] range
    
    Returns:
        PIL Image or list of PIL Images
    """
    # Handle batch
    if tensor.dim() == 4:
        return [tensor_to_image(t) for t in tensor]
    
    # Single image [3, H, W]
    tensor = tensor.detach().cpu().clamp(0, 1)
    
    # Convert to PIL
    return TF.to_pil_image(tensor)


def image_to_tensor(image):
    """
    Convert a PIL Image to a PyTorch tensor.
    
    Args:
        image: PIL Image or list of PIL Images
    
    Returns:
        Image tensor [3, H, W] or [B, 3, H, W] in [0, 1] range
    """
    # Handle list
    if isinstance(image, list):
        tensors = [image_to_tensor(img) for img in image]
        return torch.stack(tensors)
    
    # Single image
    return TF.to_tensor(image)


def save_image(tensor, path):
    """
    Save a tensor as an image file.
    
    Args:
        tensor: Image tensor [3, H, W] in [0, 1] range
        path: Save path
    """
    img = tensor_to_image(tensor)
    img.save(path)


def load_image(path):
    """
    Load an image file as a tensor.
    
    Args:
        path: Image file path
    
    Returns:
        Image tensor [3, H, W] in [0, 1] range
    """
    img = Image.open(path).convert('RGB')
    return image_to_tensor(img)


def make_grid(images, nrow=4):
    """
    Create a grid of images.
    
    Args:
        images: List of tensors or batched tensor [B, 3, H, W]
        nrow: Number of images per row
    
    Returns:
        Grid tensor [3, H_grid, W_grid]
    """
    import torchvision.utils as vutils
    
    if isinstance(images, list):
        images = torch.stack(images)
    
    grid = vutils.make_grid(images, nrow=nrow, normalize=True, padding=2)
    return grid


if __name__ == "__main__":
    # Test utilities
    print("Testing image utilities...")
    
    # Create test tensor
    tensor = torch.rand(3, 64, 64)
    
    # Convert to image and back
    img = tensor_to_image(tensor)
    print(f"PIL Image size: {img.size}")
    
    tensor_back = image_to_tensor(img)
    print(f"Tensor shape: {tensor_back.shape}")
    
    # Test batch
    batch = torch.rand(4, 3, 64, 64)
    images = tensor_to_image(batch)
    print(f"Batch converted to {len(images)} images")
    
    # Test grid
    grid = make_grid(batch, nrow=2)
    print(f"Grid shape: {grid.shape}")
    
    print("\n✓ Image utilities test passed!")
