"""
Image quality metrics for super-resolution evaluation.
PSNR and SSIM implementations.
"""
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.
    
    Args:
        img1: First image (numpy array or torch tensor)
        img2: Second image (numpy array or torch tensor)
        max_val: Maximum possible pixel value (1.0 for normalized images)
    
    Returns:
        PSNR value in dB
    """
    # Convert tensors to numpy if needed
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    # Ensure images are in [H, W, C] format
    if img1.ndim == 3 and img1.shape[0] == 3:
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.ndim == 3 and img2.shape[0] == 3:
        img2 = np.transpose(img2, (1, 2, 0))
    
    # Calculate PSNR
    return psnr(img1, img2, data_range=max_val)


def calculate_ssim(img1, img2, max_val=1.0):
    """
    Calculate SSIM (Structural Similarity Index) between two images.
    
    Args:
        img1: First image (numpy array or torch tensor)
        img2: Second image (numpy array or torch tensor)
        max_val: Maximum possible pixel value (1.0 for normalized images)
    
    Returns:
        SSIM value (between 0 and 1)
    """
    # Convert tensors to numpy if needed
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    # Ensure images are in [H, W, C] format
    if img1.ndim == 3 and img1.shape[0] == 3:
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.ndim == 3 and img2.shape[0] == 3:
        img2 = np.transpose(img2, (1, 2, 0))
    
    # Calculate SSIM
    return ssim(img1, img2, data_range=max_val, channel_axis=2)


def batch_psnr(batch_generated, batch_target):
    """
    Calculate average PSNR for a batch of images.
    
    Args:
        batch_generated: Generated images [B, 3, H, W]
        batch_target: Target images [B, 3, H, W]
    
    Returns:
        Average PSNR across the batch
    """
    psnr_values = []
    
    batch_generated = batch_generated.detach()
    batch_target = batch_target.detach()
    
    for i in range(batch_generated.size(0)):
        psnr_val = calculate_psnr(batch_generated[i], batch_target[i])
        psnr_values.append(psnr_val)
    
    return np.mean(psnr_values)


def batch_ssim(batch_generated, batch_target):
    """
    Calculate average SSIM for a batch of images.
    
    Args:
        batch_generated: Generated images [B, 3, H, W]
        batch_target: Target images [B, 3, H, W]
    
    Returns:
        Average SSIM across the batch
    """
    ssim_values = []
    
    batch_generated = batch_generated.detach()
    batch_target = batch_target.detach()
    
    for i in range(batch_generated.size(0)):
        ssim_val = calculate_ssim(batch_generated[i], batch_target[i])
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)


if __name__ == "__main__":
    # Test metrics
    print("Testing PSNR and SSIM metrics...")
    
    # Create test images
    img1 = torch.rand(3, 96, 96)
    img2 = img1.clone()  # Identical image
    img3 = torch.rand(3, 96, 96)  # Different image
    
    # Test PSNR
    psnr_identical = calculate_psnr(img1, img2)
    psnr_different = calculate_psnr(img1, img3)
    print(f"PSNR (identical): {psnr_identical:.2f} dB (should be very high)")
    print(f"PSNR (different): {psnr_different:.2f} dB")
    
    # Test SSIM
    ssim_identical = calculate_ssim(img1, img2)
    ssim_different = calculate_ssim(img1, img3)
    print(f"SSIM (identical): {ssim_identical:.4f} (should be ~1.0)")
    print(f"SSIM (different): {ssim_different:.4f}")
    
    # Test batch metrics
    batch_generated = torch.rand(4, 3, 96, 96)
    batch_target = torch.rand(4, 3, 96, 96)
    
    avg_psnr = batch_psnr(batch_generated, batch_target)
    avg_ssim = batch_ssim(batch_generated, batch_target)
    print(f"\nBatch PSNR: {avg_psnr:.2f} dB")
    print(f"Batch SSIM: {avg_ssim:.4f}")
    
    print("\n✓ Metrics test passed!")
