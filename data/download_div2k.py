"""
Download DIV2K dataset for SRGAN training.

DIV2K is a high-quality dataset of 2K resolution images.
- Train: 800 images
- Validation: 100 images
"""
import os
import urllib.request
import zipfile
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """Download a file with progress bar"""
    print(f"Downloading {url}...")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_zip(zip_path, extract_to):
    """Extract a zip file"""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✓ Extracted to {extract_to}")


def download_div2k():
    """
    Download DIV2K dataset.
    
    Note: DIV2K HR images are very large (multiple GB).
    You can also manually download from: https://data.vision.ee.ethz.ch/cvl/DIV2K/
    """
    # Create data directories
    os.makedirs('data/train/HR', exist_ok=True)
    os.makedirs('data/val/HR', exist_ok=True)
    
    print("=" * 60)
    print("DIV2K Dataset Download")
    print("=" * 60)
    
    # DIV2K download URLs
    # Note: These are example URLs - you may need to update them
    train_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    val_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
    
    # Download training set
    if not os.path.exists('data/DIV2K_train_HR.zip'):
        try:
            download_file(train_url, 'data/DIV2K_train_HR.zip')
        except Exception as e:
            print(f"Error downloading training set: {e}")
            print("\nPlease download manually from:")
            print("https://data.vision.ee.ethz.ch/cvl/DIV2K/")
            return
    else:
        print("Training set already downloaded")
    
    # Extract training set
    if not os.path.exists('data/DIV2K_train_HR'):
        extract_zip('data/DIV2K_train_HR.zip', 'data/')
        
        # Move images to train/HR
        import shutil
        src_dir = 'data/DIV2K_train_HR'
        dst_dir = 'data/train/HR'
        
        if os.path.exists(src_dir):
            for img in os.listdir(src_dir):
                shutil.move(os.path.join(src_dir, img), dst_dir)
            os.rmdir(src_dir)
            print(f"✓ Moved training images to {dst_dir}")
    
    # Download validation set
    if not os.path.exists('data/DIV2K_valid_HR.zip'):
        try:
            download_file(val_url, 'data/DIV2K_valid_HR.zip')
        except Exception as e:
            print(f"Error downloading validation set: {e}")
            print("\nPlease download manually from:")
            print("https://data.vision.ee.ethz.ch/cvl/DIV2K/")
            return
    else:
        print("Validation set already downloaded")
    
    # Extract validation set
    if not os.path.exists('data/DIV2K_valid_HR'):
        extract_zip('data/DIV2K_valid_HR.zip', 'data/')
        
        # Move images to val/HR
        import shutil
        src_dir = 'data/DIV2K_valid_HR'
        dst_dir = 'data/val/HR'
        
        if os.path.exists(src_dir):
            for img in os.listdir(src_dir):
                shutil.move(os.path.join(src_dir, img), dst_dir)
            os.rmdir(src_dir)
            print(f"✓ Moved validation images to {dst_dir}")
    
    print("\n" + "=" * 60)
    print("✓ DIV2K Dataset Ready!")
    print("=" * 60)
    print(f"Training images: {len(os.listdir('data/train/HR'))}")
    print(f"Validation images: {len(os.listdir('data/val/HR'))}")


def create_dummy_dataset(num_train=50, num_val=10, size=256):
    """
    Create a small dummy dataset for testing.
    Useful for quick experiments without downloading full DIV2K.
    
    Args:
        num_train: Number of training images
        num_val: Number of validation images
        size: Image size
    """
    from PIL import Image
    import random
    
    print(f"Creating dummy dataset ({num_train} train, {num_val} val)...")
    
    os.makedirs('data/train/HR', exist_ok=True)
    os.makedirs('data/val/HR', exist_ok=True)
    
    # Generate training images
    for i in range(num_train):
        # Random colored image
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img = Image.new('RGB', (size, size), color)
        img.save(f'data/train/HR/dummy_{i:04d}.png')
    
    # Generate validation images
    for i in range(num_val):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img = Image.new('RGB', (size, size), color)
        img.save(f'data/val/HR/dummy_{i:04d}.png')
    
    print("✓ Dummy dataset created!")
    print(f"Training images: {num_train}")
    print(f"Validation images: {num_val}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download DIV2K dataset')
    parser.add_argument('--dummy', action='store_true', 
                       help='Create dummy dataset for testing instead')
    parser.add_argument('--num_train', type=int, default=50,
                       help='Number of dummy training images')
    parser.add_argument('--num_val', type=int, default=10,
                       help='Number of dummy validation images')
    
    args = parser.parse_args()
    
    if args.dummy:
        create_dummy_dataset(args.num_train, args.num_val)
    else:
        download_div2k()
