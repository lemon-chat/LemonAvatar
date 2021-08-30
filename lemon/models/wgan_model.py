
import argparse
import os
import numpy as np
import math
from torch._C import device

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl

from PIL import Image



class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim: int):
        super(Generator, self).__init__()

        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


class WGAN(pl.LightningModule):
    def __init__(self, channels:int, img_size:int,
            latent_dim: int = 100,
            lr: float = 0.0002,
            **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.clip_value = 0.01
        self.n_critic = 5

        # Initialize generator and discriminator
        self.generator = Generator(img_shape=(channels, img_size, img_size), latent_dim=latent_dim)
        self.discriminator = Discriminator(img_shape=(channels, img_size, img_size))

    
    def configure_optimizers(self):
        g_optim = torch.optim.RMSprop(self.generator.parameters(), lr=self.hparams.lr)
        d_optim = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.hparams.lr)
        return (d_optim, g_optim)
            

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        generated_image = self.generator(x)
        return generated_image

    def training_step(self, batch, batch_idx, optimizer_idx):
        # training_step defined the train loop.
        # It is independent of forward
        imgs, _ = batch
        device = imgs.device
        batch_size = imgs.shape[0]

        real_imgs = imgs

        # Sample noise as generator input
        z = torch.tensor(np.random.normal(0, 1, (batch_size, self.hparams.latent_dim)), device=device).float()
        # train Discriminator
        if optimizer_idx == 0:
            # Generate a batch of images
            fake_imgs = self(z).detach()
            # Adversarial loss
            loss_D = -torch.mean(self.discriminator(real_imgs)) + torch.mean(self.discriminator(fake_imgs))
            self.log("d_loss", loss_D)
            return loss_D

        if optimizer_idx == 1:
            # Clip weights of discriminator
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.clip_value, self.clip_value)
            if self.global_step % self.n_critic == 0:
                # Generate a batch of images
                gen_imgs = self.generator(z)
                # Adversarial loss
                loss_G = -torch.mean(self.discriminator(gen_imgs))
                self.log("g_loss", loss_G)
                return loss_G
            else:
                return None
    
    def training_epoch_end(self, outputs):
        device = self.device
        z = torch.randn(8, self.hparams.latent_dim, requires_grad=False).to(device)

        generated_imgs = self(z)
        img_tensor = generated_imgs[0]
        img = (255*img_tensor.permute(1, 2, 0)).detach().cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img, 'RGB')
        img.save(f"images/{self.current_epoch}.jpg")
        # img.show()
