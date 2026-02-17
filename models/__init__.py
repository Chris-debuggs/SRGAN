"""SRGAN Models"""
from .generator import Generator
from .discriminator import Discriminator
from .losses import ContentLoss, PerceptualLoss, AdversarialLoss

__all__ = [
    'Generator',
    'Discriminator',
    'ContentLoss',
    'PerceptualLoss',
    'AdversarialLoss',
]
