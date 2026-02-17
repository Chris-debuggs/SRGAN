# Super-Resolution GAN (SRGAN)

A PyTorch implementation of **Super-Resolution GAN** for learning-based image upscaling. Train a GAN from scratch to enhance low-resolution images into photo-realistic high-resolution versions.

## 🎯 What This Does

- **Upscales images by 4×** (configurable to 2× or 4×)
- **Generates realistic textures** using adversarial training
- **Combines three loss functions**: Content (L1), Perceptual (VGG19), and Adversarial (GAN)
- **Two-stage training**: Pre-training with content loss → Fine-tuning with full GAN

## 🏗️ Architecture

### Generator
- 16 residual blocks with skip connections
- PixelShuffle for learnable upsampling
- ~1.5M parameters

### Discriminator  
- VGG-style CNN classifier
- Strided convolutions for downsampling
- ~3M parameters

### Loss Functions
```
L_total = λ_content * L1(SR, HR) + λ_perceptual * VGG(SR, HR) + λ_adversarial * BCE(D(SR), real)
```

## 📁 Project Structure

```
SRGAN/
├── models/
│   ├── generator.py          # Generator architecture
│   ├── discriminator.py      # Discriminator architecture
│   └── losses.py              # Loss functions
├── data/
│   └── dataset.py             # Dataset class
├── utils/
│   ├── metrics.py             # PSNR & SSIM
│   ├── image_utils.py         # Image conversion utilities
│   └── checkpoint.py          # Model checkpointing
├── train.py                   # Training script
├── test.py                    # Inference script
├── config.py                  # Configuration
└── requirements.txt
```

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+ with CUDA (recommended)
- ~8GB GPU VRAM for training

### 2. Prepare Dataset

Download a dataset like **DIV2K**:
- [DIV2K Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

Organize as:
```
data/
├── train/HR/     # Training high-res images
└── val/HR/       # Validation high-res images
```

Low-resolution pairs are generated automatically during training.

### 3. Configure Training

Edit `config.py` to adjust hyperparameters:

```python
SCALE_FACTOR = 4              # Upsampling factor
NUM_RESIDUAL_BLOCKS = 16       # Generator depth
BATCH_SIZE = 16                # Batch size
PRETRAIN_ITERATIONS = 100000   # Stage 1 iterations
FINETUNE_ITERATIONS = 300000   # Stage 2 iterations
```

### 4. Train the Model

```bash
python train.py
```

**Training Progress:**
- **Stage 1 (100K iters)**: Pre-training with content loss (~10-12 hours on RTX 3080)
- **Stage 2 (300K iters)**: Fine-tuning with GAN loss (~30-36 hours on RTX 3080)

Monitor with TensorBoard:
```bash
tensorboard --logdir logs
```

### 5. Test on Images

**Single image:**
```bash
python test.py --image path/to/low_res.png --output sr_output.png
```

**Batch processing:**
```bash
python test.py --input_dir path/to/lr_images/ --output output_dir/
```

**Compare with bicubic:**
```bash
python test.py --image lr_image.png --compare --output comparison/
```

## 📊 Evaluation Metrics

- **PSNR** (Peak Signal-to-Noise Ratio): Measures pixel accuracy
- **SSIM** (Structural Similarity): Measures perceptual quality

Expected results on DIV2K:
- PSNR: ~28-30 dB (vs. ~26 dB for bicubic)
- SSIM: ~0.80-0.85 (vs. ~0.70 for bicubic)

## 🎓 Training Tips

### If Discriminator Overpowers Generator
- Reduce discriminator learning rate
- Train generator more frequently (2:1 ratio)
- Increase `LAMBDA_ADVERSARIAL`

### If Generator Doesn't Improve
- Increase `LAMBDA_PERCEPTUAL` for better texture
- Try longer pre-training
- Check that VGG19 weights downloaded correctly

### To Speed Up Training
- Use mixed precision (`torch.cuda.amp`)
- Reduce `NUM_RESIDUAL_BLOCKS` to 8-12
- Use smaller crop size (e.g., 64×64)

## 🔬 Experiment Ideas

1. **Different upscaling factors**: Change `SCALE_FACTOR` to 2×, 8×
2. **Loss weight tuning**: Adjust λ values in `config.py`
3. **Architecture variants**: Try deeper networks, different residual blocks
4. **Domain-specific training**: Train on faces (CelebA-HQ) or anime images
5. **Progressive growing**: Start with 2× then fine-tune to 4×

## 📚 References

- **SRGAN Paper**: [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
- **ESRGAN**: [Enhanced SRGAN](https://arxiv.org/abs/1809.00219) (improved architecture)

## 🤝 Acknowledgments

This implementation was built for educational purposes to understand super-resolution GANs from the ground up.

## 📝 License

MIT License - Feel free to experiment and learn!

---

**Happy Super-Resolving! 🚀**