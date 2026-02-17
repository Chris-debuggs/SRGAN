"""
SRGAN Inference Script

Test the trained model on images and generate super-resolution results.
"""
import os
import argparse
import time
import torch
from PIL import Image

from models import Generator
from utils import load_image, save_image, tensor_to_image
from config import Config


def super_resolve(image_path, model_path, output_path, device='cuda'):
    """
    Perform super-resolution on a single image.
    
    Args:
        image_path: Path to input LR image
        model_path: Path to trained generator checkpoint
        output_path: Path to save SR image
        device: Device to run inference on
    """
    # Load model
    print(f"Loading model from {model_path}...")
    generator = Generator(
        scale_factor=Config.SCALE_FACTOR,
        num_residual_blocks=Config.NUM_RESIDUAL_BLOCKS
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    # Load image
    print(f"Loading image from {image_path}...")
    lr_img = Image.open(image_path).convert('RGB')
    
    # Convert to tensor
    import torchvision.transforms.functional as TF
    lr_tensor = TF.to_tensor(lr_img).unsqueeze(0).to(device)
    
    # Pad image to be divisible by scale factor
    _, _, h, w = lr_tensor.shape
    pad_h = (Config.SCALE_FACTOR - h % Config.SCALE_FACTOR) % Config.SCALE_FACTOR
    pad_w = (Config.SCALE_FACTOR - w % Config.SCALE_FACTOR) % Config.SCALE_FACTOR
    
    if pad_h > 0 or pad_w > 0:
        lr_tensor = torch.nn.functional.pad(lr_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    
    # Generate SR image
    print("Generating super-resolution image...")
    start_time = time.time()
    
    with torch.no_grad():
        sr_tensor = generator(lr_tensor)
    
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.3f} seconds")
    
    # Remove padding from output
    if pad_h > 0 or pad_w > 0:
        sr_tensor = sr_tensor[:, :, :h * Config.SCALE_FACTOR, :w * Config.SCALE_FACTOR]
    
    # Clamp and convert to image
    sr_tensor = sr_tensor.clamp(0, 1).squeeze(0)
    
    # Save result
    print(f"Saving result to {output_path}...")
    save_image(sr_tensor, output_path)
    
    # Print info
    sr_img = tensor_to_image(sr_tensor)
    print(f"Input size: {lr_img.size}")
    print(f"Output size: {sr_img.size}")
    print(f"✓ Super-resolution completed!")
    
    return sr_img


def batch_super_resolve(input_dir, model_path, output_dir, device='cuda'):
    """
    Perform super-resolution on all images in a directory.
    
    Args:
        input_dir: Directory containing LR images
        model_path: Path to trained generator checkpoint
        output_dir: Directory to save SR images
        device: Device to run inference on
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}...")
    generator = Generator(
        scale_factor=Config.SCALE_FACTOR,
        num_residual_blocks=Config.NUM_RESIDUAL_BLOCKS
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    # Get all image files
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(valid_extensions)
    ]
    
    print(f"Found {len(image_files)} images")
    
    # Process each image
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing {image_file}...")
        
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, f'sr_{image_file}')
        
        try:
            super_resolve(input_path, model_path, output_path, device)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue
    
    print(f"\n✓ Batch processing completed! Results saved to {output_dir}")


def compare_with_bicubic(image_path, model_path, output_dir, device='cuda'):
    """
    Compare SRGAN with bicubic upsampling.
    
    Args:
        image_path: Path to input LR image
        model_path: Path to trained generator checkpoint
        output_dir: Directory to save comparison
        device: Device to run inference on
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    lr_img = Image.open(image_path).convert('RGB')
    
    # SRGAN super-resolution
    sr_path = os.path.join(output_dir, 'srgan_output.png')
    sr_img = super_resolve(image_path, model_path, sr_path, device)
    
    # Bicubic upsampling
    bicubic_img = lr_img.resize(
        (lr_img.width * Config.SCALE_FACTOR, lr_img.height * Config.SCALE_FACTOR),
        Image.BICUBIC
    )
    bicubic_path = os.path.join(output_dir, 'bicubic_output.png')
    bicubic_img.save(bicubic_path)
    
    # Save LR for reference
    lr_path = os.path.join(output_dir, 'input_lr.png')
    lr_img.save(lr_path)
    
    print(f"\n✓ Comparison saved to {output_dir}")
    print(f"  - LR input: {lr_path}")
    print(f"  - Bicubic: {bicubic_path}")
    print(f"  - SRGAN: {sr_path}")


def main():
    parser = argparse.ArgumentParser(description='SRGAN Inference')
    parser.add_argument('--image', type=str, help='Path to input LR image')
    parser.add_argument('--input_dir', type=str, help='Directory of LR images (batch mode)')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth', 
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, help='Output path or directory')
    parser.add_argument('--compare', action='store_true', 
                       help='Compare with bicubic upsampling')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Single image mode
    if args.image:
        output_path = args.output or 'output_sr.png'
        
        if args.compare:
            output_dir = args.output or 'comparison'
            compare_with_bicubic(args.image, args.model, output_dir, args.device)
        else:
            super_resolve(args.image, args.model, output_path, args.device)
    
    # Batch mode
    elif args.input_dir:
        output_dir = args.output or 'output_sr'
        batch_super_resolve(args.input_dir, args.model, output_dir, args.device)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
