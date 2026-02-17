"""
Checkpoint utilities for saving and loading model states.
"""
import os
import torch


def save_checkpoint(state, checkpoint_dir, filename):
    """
    Save model checkpoint.
    
    Args:
        state: Dictionary containing model states and metadata
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath, generator, discriminator=None, 
                   optimizer_g=None, optimizer_d=None, device='cpu'):
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        generator: Generator model
        discriminator: Discriminator model (optional)
        optimizer_g: Generator optimizer (optional)
        optimizer_d: Discriminator optimizer (optional)
        device: Device to load checkpoint on
    
    Returns:
        Dictionary containing checkpoint metadata
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    print(f"Loading checkpoint: {filepath}")
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load generator
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    # Load discriminator if provided
    if discriminator is not None and 'discriminator_state_dict' in checkpoint:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    # Load optimizers if provided
    if optimizer_g is not None and 'optimizer_g_state_dict' in checkpoint:
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    
    if optimizer_d is not None and 'optimizer_d_state_dict' in checkpoint:
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    
    # Return metadata
    metadata = {
        'iteration': checkpoint.get('iteration', 0),
        'epoch': checkpoint.get('epoch', 0),
        'best_psnr': checkpoint.get('best_psnr', 0.0),
    }
    
    print(f"Checkpoint loaded: iteration={metadata['iteration']}")
    
    return metadata


def save_best_model(generator, checkpoint_dir, psnr):
    """
    Save the best model based on PSNR.
    
    Args:
        generator: Generator model
        checkpoint_dir: Directory to save checkpoint
        psnr: Current PSNR value
    """
    state = {
        'generator_state_dict': generator.state_dict(),
        'psnr': psnr,
    }
    save_checkpoint(state, checkpoint_dir, 'best_model.pth')


if __name__ == "__main__":
    # Test checkpoint utilities
    print("Testing checkpoint utilities...")
    
    from models.generator import Generator
    from models.discriminator import Discriminator
    
    # Create models
    generator = Generator()
    discriminator = Discriminator()
    
    # Create optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    
    # Test save
    state = {
        'iteration': 1000,
        'epoch': 10,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'best_psnr': 28.5,
    }
    
    test_dir = 'test_checkpoints'
    save_checkpoint(state, test_dir, 'test_checkpoint.pth')
    
    # Test load
    metadata = load_checkpoint(
        os.path.join(test_dir, 'test_checkpoint.pth'),
        generator, discriminator, optimizer_g, optimizer_d
    )
    
    print(f"Loaded metadata: {metadata}")
    
    # Cleanup
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    print("\n✓ Checkpoint utilities test passed!")
