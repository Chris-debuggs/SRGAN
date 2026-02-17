"""
Configuration and hyperparameters for SRGAN training.
All training settings in one place for easy experimentation.
"""
import torch

class Config:
    """SRGAN Configuration"""
    
    # ============ Model Architecture ============
    SCALE_FACTOR = 4  # Upsampling factor (2 or 4)
    NUM_RESIDUAL_BLOCKS = 16  # Number of residual blocks in Generator
    
    # ============ Training ============
    # Two-stage training strategy
    PRETRAIN_ITERATIONS = 100000  # Stage 1: Content loss only
    FINETUNE_ITERATIONS = 300000  # Stage 2: Full GAN training
    
    # Learning rates
    LR_GENERATOR = 1e-4
    LR_DISCRIMINATOR = 1e-4
    LR_PRETRAIN = 2e-4  # Higher LR for pretraining
    
    # Learning rate decay
    LR_DECAY_STEP = 100000  # Decay every N iterations
    LR_DECAY_GAMMA = 0.5  # Multiply LR by this factor
    
    # Optimization
    BETA1 = 0.9  # Adam beta1
    BETA2 = 0.999  # Adam beta2
    BATCH_SIZE = 16
    GRADIENT_CLIP = 10.0  # Gradient clipping max norm
    
    # ============ Loss Weights ============
    LAMBDA_CONTENT = 1.0  # Content loss weight
    LAMBDA_PERCEPTUAL = 0.006  # Perceptual loss weight (VGG)
    LAMBDA_ADVERSARIAL = 0.001  # Adversarial loss weight
    
    # VGG layer for perceptual loss
    VGG_LAYER = 'relu5_4'  # Which VGG19 layer to use
    
    # ============ Data ============
    # Image crop size during training
    HR_CROP_SIZE = 96  # High-res patch size
    LR_CROP_SIZE = HR_CROP_SIZE // SCALE_FACTOR  # Low-res patch size (24 for 4x)
    
    # Data paths
    TRAIN_HR_PATH = 'data/train/HR'
    VAL_HR_PATH = 'data/val/HR'
    
    # Data augmentation
    USE_AUGMENTATION = True
    HORIZONTAL_FLIP = True
    ROTATION = True  # Random 90° rotations
    
    # ============ Checkpointing & Logging ============
    CHECKPOINT_DIR = 'checkpoints'
    LOG_DIR = 'logs'
    SAMPLE_DIR = 'samples'
    
    # Save checkpoints every N iterations
    CHECKPOINT_INTERVAL = 5000
    
    # Log to tensorboard every N iterations
    LOG_INTERVAL = 100
    
    # Save sample images every N iterations
    SAMPLE_INTERVAL = 1000
    
    # ============ Hardware ============
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 4  # DataLoader workers
    PIN_MEMORY = True if torch.cuda.is_available() else False
    
    # ============ Evaluation ============
    # Metrics to compute
    COMPUTE_PSNR = True
    COMPUTE_SSIM = True
    
    @classmethod
    def display(cls):
        """Print all configuration values"""
        print("=" * 60)
        print("SRGAN Configuration")
        print("=" * 60)
        for key, value in vars(cls).items():
            if not key.startswith('_') and not callable(value):
                print(f"{key:30s}: {value}")
        print("=" * 60)


if __name__ == "__main__":
    # Display config when run directly
    Config.display()
