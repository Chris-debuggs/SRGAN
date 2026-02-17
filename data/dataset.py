"""
Dataset class for Super-Resolution training.
Loads HR images and generates LR pairs on-the-fly.
"""
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class SRDataset(Dataset):
    """
    Super-Resolution Dataset
    
    Loads high-resolution images and generates low-resolution pairs through downsampling.
    Applies data augmentation during training.
    
    Args:
        hr_dir (str): Directory containing HR images
        scale_factor (int): Downsampling factor (2 or 4)
        crop_size (int): HR patch size for random cropping
        augment (bool): Whether to apply data augmentation
        mode (str): 'train' or 'val'
    """
    def __init__(self, hr_dir, scale_factor=4, crop_size=96, augment=True, mode='train'):
        super(SRDataset, self).__init__()
        
        self.hr_dir = hr_dir
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.augment = augment and (mode == 'train')
        self.mode = mode
        
        # Get all image files
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        self.image_files = [
            f for f in os.listdir(hr_dir)
            if f.lower().endswith(valid_extensions)
        ]
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {hr_dir}")
        
        print(f"Loaded {len(self.image_files)} images from {hr_dir}")
        
        # LR size
        self.lr_crop_size = crop_size // scale_factor
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict: {'lr': LR image tensor, 'hr': HR image tensor}
        """
        # Load HR image
        img_path = os.path.join(self.hr_dir, self.image_files[idx])
        hr_img = Image.open(img_path).convert('RGB')
        
        # Random crop
        if self.mode == 'train':
            hr_img = self._random_crop(hr_img, self.crop_size)
        else:
            # For validation, use center crop
            hr_img = TF.center_crop(hr_img, min(hr_img.size))
            hr_img = TF.resize(hr_img, (self.crop_size, self.crop_size))
        
        # Data augmentation
        if self.augment:
            hr_img = self._augment(hr_img)
        
        # Generate LR image by downsampling
        lr_img = TF.resize(hr_img, (self.lr_crop_size, self.lr_crop_size), 
                          interpolation=Image.BICUBIC)
        
        # Convert to tensors [0, 1]
        hr_tensor = TF.to_tensor(hr_img)
        lr_tensor = TF.to_tensor(lr_img)
        
        return {
            'lr': lr_tensor,  # [3, H/scale, W/scale]
            'hr': hr_tensor,  # [3, H, W]
        }
    
    def _random_crop(self, img, crop_size):
        """Random crop to specified size"""
        width, height = img.size
        
        if width < crop_size or height < crop_size:
            # If image smaller than crop, resize first
            scale = max(crop_size / width, crop_size / height)
            new_width = int(width * scale) + 1
            new_height = int(height * scale) + 1
            img = TF.resize(img, (new_height, new_width))
            width, height = img.size
        
        # Random crop
        left = random.randint(0, width - crop_size)
        top = random.randint(0, height - crop_size)
        img = TF.crop(img, top, left, crop_size, crop_size)
        
        return img
    
    def _augment(self, img):
        """Apply data augmentation"""
        # Random horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
        
        # Random rotation (0, 90, 180, 270 degrees)
        if random.random() > 0.5:
            angle = random.choice([0, 90, 180, 270])
            img = TF.rotate(img, angle)
        
        return img


if __name__ == "__main__":
    # Test dataset
    print("Testing SRDataset...")
    
    # Note: This will fail if data directory doesn't exist
    # Create dummy data directory for testing
    test_dir = "data/test"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a dummy image
    dummy_img = Image.new('RGB', (256, 256), color='red')
    dummy_img.save(os.path.join(test_dir, "test.png"))
    
    try:
        dataset = SRDataset(
            hr_dir=test_dir,
            scale_factor=4,
            crop_size=96,
            augment=True,
            mode='train'
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Get a sample
        sample = dataset[0]
        lr = sample['lr']
        hr = sample['hr']
        
        print(f"LR shape: {lr.shape}")
        print(f"HR shape: {hr.shape}")
        
        assert lr.shape == (3, 24, 24), "LR shape mismatch!"
        assert hr.shape == (3, 96, 96), "HR shape mismatch!"
        
        print("✓ Dataset test passed!")
        
    finally:
        # Cleanup
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
