from lemon.models.dcgan_model import DCGAN
from lemon.models.wgan_model import WGAN
from lemon.models.gan_model import GAN
import os
import argparse
import torch
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from PIL import Image

from lemon.datasets.pixiv_dataset import PixivDataset

def main():
    os.makedirs("images", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate") # 0.0002
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    opt = parser.parse_args()
    print(opt)

    dataset = PixivDataset("./data", size=opt.img_size, lazy=True)

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=16,
    )

    # init model
    gan = WGAN(channels=3, img_size=opt.img_size, latent_dim=opt.latent_dim, lr=opt.lr)
    # gan = DCGAN(channels=3, img_size=opt.img_size)
    # gan = GAN(3, opt.img_size, opt.img_size, latent_dim=opt.latent_dim, lr=opt.lr, b1=opt.b1, b2=opt.b2)
    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    # trainer = pl.Trainer(gpus=8) (if you have GPUs)
    trainer = pl.Trainer(
        gpus=[1]
    )
    trainer.fit(gan, train_dataloader)


if __name__ == "__main__":
    main()