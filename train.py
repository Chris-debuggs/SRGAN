"""
SRGAN Training Script

Two-stage training:
1. Pre-training with content loss only (100K iterations)
2. Fine-tuning with full GAN loss (300K iterations)
"""
import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import Generator, Discriminator, SRGANLoss, AdversarialLoss
from data import SRDataset
from utils import (
    calculate_psnr, calculate_ssim,
    batch_psnr, batch_ssim,
    save_checkpoint, load_checkpoint,
    tensor_to_image, save_image, make_grid
)
from config import Config


def train():
    """Main training function"""
    
    # Display configuration
    Config.display()
    
    # Create directories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    os.makedirs(Config.SAMPLE_DIR, exist_ok=True)
    
    # Initialize tensorboard
    writer = SummaryWriter(Config.LOG_DIR)
    
    # Create models
    print("\nInitializing models...")
    generator = Generator(
        scale_factor=Config.SCALE_FACTOR,
        num_residual_blocks=Config.NUM_RESIDUAL_BLOCKS
    ).to(Config.DEVICE)
    
    discriminator = Discriminator(
        input_shape=(Config.HR_CROP_SIZE, Config.HR_CROP_SIZE)
    ).to(Config.DEVICE)
    
    # Count parameters
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = SRDataset(
        hr_dir=Config.TRAIN_HR_PATH,
        scale_factor=Config.SCALE_FACTOR,
        crop_size=Config.HR_CROP_SIZE,
        augment=Config.USE_AUGMENTATION,
        mode='train'
    )
    
    val_dataset = SRDataset(
        hr_dir=Config.VAL_HR_PATH,
        scale_factor=Config.SCALE_FACTOR,
        crop_size=Config.HR_CROP_SIZE,
        augment=False,
        mode='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize optimizers
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=Config.LR_PRETRAIN,  # Start with pretrain LR
        betas=(Config.BETA1, Config.BETA2)
    )
    
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=Config.LR_DISCRIMINATOR,
        betas=(Config.BETA1, Config.BETA2)
    )
    
    # Learning rate schedulers
    scheduler_g = torch.optim.lr_scheduler.StepLR(
        optimizer_g,
        step_size=Config.LR_DECAY_STEP,
        gamma=Config.LR_DECAY_GAMMA
    )
    
    scheduler_d = torch.optim.lr_scheduler.StepLR(
        optimizer_d,
        step_size=Config.LR_DECAY_STEP,
        gamma=Config.LR_DECAY_GAMMA
    )
    
    # Loss functions
    content_loss_fn = nn.L1Loss().to(Config.DEVICE)
    srgan_loss_fn = SRGANLoss(
        lambda_content=Config.LAMBDA_CONTENT,
        lambda_perceptual=Config.LAMBDA_PERCEPTUAL,
        lambda_adversarial=Config.LAMBDA_ADVERSARIAL
    ).to(Config.DEVICE)
    adversarial_loss_fn = AdversarialLoss().to(Config.DEVICE)
    
    # Training state
    iteration = 0
    best_psnr = 0.0
    
    # ============================================
    # STAGE 1: Pre-training (Content Loss Only)
    # ============================================
    print("\n" + "=" * 60)
    print("STAGE 1: Pre-training with Content Loss")
    print("=" * 60)
    
    pretrain_iterations = Config.PRETRAIN_ITERATIONS
    generator.train()
    
    pbar = tqdm(total=pretrain_iterations, desc="Pre-training")
    
    while iteration < pretrain_iterations:
        for batch in train_loader:
            if iteration >= pretrain_iterations:
                break
            
            lr_imgs = batch['lr'].to(Config.DEVICE)
            hr_imgs = batch['hr'].to(Config.DEVICE)
            
            # Generate SR images
            sr_imgs = generator(lr_imgs)
            
            # Content loss
            loss_content = content_loss_fn(sr_imgs, hr_imgs)
            
            # Optimize generator
            optimizer_g.zero_grad()
            loss_content.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), Config.GRADIENT_CLIP)
            optimizer_g.step()
            
            # Update learning rate
            scheduler_g.step()
            
            iteration += 1
            pbar.update(1)
            
            # Logging
            if iteration % Config.LOG_INTERVAL == 0:
                writer.add_scalar('Pretrain/Loss', loss_content.item(), iteration)
                writer.add_scalar('Pretrain/LR', optimizer_g.param_groups[0]['lr'], iteration)
                pbar.set_postfix({'Loss': f'{loss_content.item():.4f}'})
            
            # Save samples
            if iteration % Config.SAMPLE_INTERVAL == 0:
                with torch.no_grad():
                    # Save comparison grid
                    comparison = torch.cat([
                        nn.functional.interpolate(lr_imgs[:4], scale_factor=Config.SCALE_FACTOR, mode='bicubic'),
                        sr_imgs[:4].clamp(0, 1),
                        hr_imgs[:4]
                    ], dim=0)
                    grid = make_grid(comparison, nrow=4)
                    save_image(grid, os.path.join(Config.SAMPLE_DIR, f'pretrain_{iteration:06d}.png'))
                    writer.add_image('Pretrain/Comparison', grid, iteration)
            
            # Save checkpoint
            if iteration % Config.CHECKPOINT_INTERVAL == 0:
                state = {
                    'iteration': iteration,
                    'generator_state_dict': generator.state_dict(),
                    'optimizer_g_state_dict': optimizer_g.state_dict(),
                    'best_psnr': best_psnr,
                }
                save_checkpoint(state, Config.CHECKPOINT_DIR, f'pretrain_{iteration:06d}.pth')
    
    pbar.close()
    print("✓ Pre-training completed!")
    
    # Save pretrained model
    state = {
        'iteration': iteration,
        'generator_state_dict': generator.state_dict(),
    }
    save_checkpoint(state, Config.CHECKPOINT_DIR, 'pretrained_generator.pth')
    
    # ============================================
    # STAGE 2: Fine-tuning (Full GAN Training)
    # ============================================
    print("\n" + "=" * 60)
    print("STAGE 2: Fine-tuning with Full GAN Loss")
    print("=" * 60)
    
    # Reset learning rates
    for param_group in optimizer_g.param_groups:
        param_group['lr'] = Config.LR_GENERATOR
    
    total_iterations = pretrain_iterations + Config.FINETUNE_ITERATIONS
    generator.train()
    discriminator.train()
    
    pbar = tqdm(total=Config.FINETUNE_ITERATIONS, desc="Fine-tuning")
    
    while iteration < total_iterations:
        for batch in train_loader:
            if iteration >= total_iterations:
                break
            
            lr_imgs = batch['lr'].to(Config.DEVICE)
            hr_imgs = batch['hr'].to(Config.DEVICE)
            
            # ==========================================
            # Train Discriminator
            # ==========================================
            optimizer_d.zero_grad()
            
            # Real images
            real_preds = discriminator(hr_imgs)
            loss_d_real = adversarial_loss_fn(real_preds, is_real=True)
            
            # Fake images
            with torch.no_grad():
                sr_imgs = generator(lr_imgs)
            fake_preds = discriminator(sr_imgs.detach())
            loss_d_fake = adversarial_loss_fn(fake_preds, is_real=False)
            
            # Total discriminator loss
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), Config.GRADIENT_CLIP)
            optimizer_d.step()
            
            # ==========================================
            # Train Generator
            # ==========================================
            optimizer_g.zero_grad()
            
            # Generate SR images
            sr_imgs = generator(lr_imgs)
            
            # Get discriminator predictions on generated images
            fake_preds = discriminator(sr_imgs)
            
            # Combined loss
            loss_g, loss_dict = srgan_loss_fn(sr_imgs, hr_imgs, fake_preds)
            
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), Config.GRADIENT_CLIP)
            optimizer_g.step()
            
            # Update learning rates
            scheduler_g.step()
            scheduler_d.step()
            
            iteration += 1
            pbar.update(1)
            
            # Logging
            if iteration % Config.LOG_INTERVAL == 0:
                writer.add_scalar('Train/G_Loss', loss_dict['total'], iteration)
                writer.add_scalar('Train/G_Content', loss_dict['content'], iteration)
                writer.add_scalar('Train/G_Perceptual', loss_dict['perceptual'], iteration)
                writer.add_scalar('Train/G_Adversarial', loss_dict['adversarial'], iteration)
                writer.add_scalar('Train/D_Loss', loss_d.item(), iteration)
                writer.add_scalar('Train/LR_G', optimizer_g.param_groups[0]['lr'], iteration)
                
                pbar.set_postfix({
                    'G': f'{loss_dict["total"]:.4f}',
                    'D': f'{loss_d.item():.4f}'
                })
            
            # Validation and samples
            if iteration % Config.SAMPLE_INTERVAL == 0:
                # Evaluate on validation set
                avg_psnr, avg_ssim = evaluate(generator, val_loader, Config.DEVICE)
                writer.add_scalar('Val/PSNR', avg_psnr, iteration)
                writer.add_scalar('Val/SSIM', avg_ssim, iteration)
                
                print(f"\nValidation - PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")
                
                # Save best model
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    state = {
                        'iteration': iteration,
                        'generator_state_dict': generator.state_dict(),
                        'psnr': best_psnr,
                    }
                    save_checkpoint(state, Config.CHECKPOINT_DIR, 'best_model.pth')
                    print(f"✓ New best model! PSNR: {best_psnr:.2f} dB")
                
                # Save sample images
                with torch.no_grad():
                    comparison = torch.cat([
                        nn.functional.interpolate(lr_imgs[:4], scale_factor=Config.SCALE_FACTOR, mode='bicubic'),
                        sr_imgs[:4].clamp(0, 1),
                        hr_imgs[:4]
                    ], dim=0)
                    grid = make_grid(comparison, nrow=4)
                    save_image(grid, os.path.join(Config.SAMPLE_DIR, f'finetune_{iteration:06d}.png'))
                    writer.add_image('Train/Comparison', grid, iteration)
                
                generator.train()
            
            # Save checkpoint
            if iteration % Config.CHECKPOINT_INTERVAL == 0:
                state = {
                    'iteration': iteration,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizer_g_state_dict': optimizer_g.state_dict(),
                    'optimizer_d_state_dict': optimizer_d.state_dict(),
                    'best_psnr': best_psnr,
                }
                save_checkpoint(state, Config.CHECKPOINT_DIR, f'checkpoint_{iteration:06d}.pth')
    
    pbar.close()
    print("\n✓ Training completed!")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    
    writer.close()


def evaluate(generator, val_loader, device):
    """Evaluate generator on validation set"""
    generator.eval()
    
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for batch in val_loader:
            lr_imgs = batch['lr'].to(device)
            hr_imgs = batch['hr'].to(device)
            
            # Generate SR images
            sr_imgs = generator(lr_imgs).clamp(0, 1)
            
            # Calculate metrics
            psnr = batch_psnr(sr_imgs, hr_imgs)
            ssim = batch_ssim(sr_imgs, hr_imgs)
            
            psnr_values.append(psnr)
            ssim_values.append(ssim)
    
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    
    return avg_psnr, avg_ssim


if __name__ == "__main__":
    train()
