
import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl


class Generator(nn.Module):
    def __init__(self, img_size=32, latent_dim=100, channels=1):
        super(Generator, self).__init__()

        self.img_size = img_size
        self.latent_dim = latent_dim
        self.channels = channels

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_size=32, latent_dim=100, channels=1):
        super(Discriminator, self).__init__()

        self.img_size = img_size
        self.latent_dim = latent_dim
        self.channels = channels

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class DCGAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.latent_dim = 100
        # Initialize generator and discriminator
        self.generator = Generator()
        self.discriminator = Discriminator()
        # Loss function
        self.adversarial_loss = torch.nn.BCELoss()
        
        # Initialize weights
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)
    
    def configure_optimizers(self):
        g_optim = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        return [g_optim, d_optim], []
            

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        generated_image = self.generator(x)
        return generated_image

    def training_step(self, batch, batch_idx, optimizer_idx):
        # training_step defined the train loop.
        # It is independent of forward
        img = batch
        batch_size = img.shape[0]

        # Adversarial ground truths
        valid = torch.tensor(batch_size, 1, requires_grad=False).fill_(1.0)
        fake = torch.tensor(batch_size, 1, requires_grad=False).fill_(0.0)

        # Configure input
        real_imgs = batch

        # -----------------
        #  Train Generator
        # -----------------
        # train generator
        if optimizer_idx == 0:
            # Sample noise as generator input
            z = torch.tensor(np.random.normal(0, 1, (batch_size, self.latent_dim)))

            # Generate a batch of images
            self.gen_imgs = self.forward(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = self.adversarial_loss(self.discriminator(self.gen_imgs), valid)

            # Logging to TensorBoard by default
            self.log("g_loss", g_loss)
            return g_loss
        # ---------------------
        #  Train Discriminator
        # ---------------------
        # train generator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
            real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
            fake_loss = self.adversarial_loss(self.discriminator(self.gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            # Logging to TensorBoard by default
            self.log("d_loss", d_loss)
            return d_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer