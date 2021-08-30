import pytorch_lightning as pl
import torch
import tqdm
import json
import os
import re
import numpy as np
import pickle
from typing import Sequence, Union
from torchvision import transforms
import glob
from PIL import Image


class PixivDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str, size: int=256, lazy=False):
        self.lazy = lazy
        self.transforms = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]),
        ])

        self.files = list(glob.glob(os.path.join(dataset_path, "./**/*.jpg"), recursive=True))
        print(f"file number: {len(self.files)}")

        if not lazy:
            print('lazy data loading')
            self.lazy_data = []
            for each_file in tqdm.tqdm(self.files):
                self.lazy_data.append(self.load_file(each_file))

    def tensor_to_img(self, img_tensor):
        return (255*img_tensor.permute(1, 2, 0)).detach().numpy().astype(np.uint8)

    def load_file(self, filename: str):
        image = Image.open(filename)
        img_tensor = self.transforms(image)

        
        #有的图像四通道，全部变三通道
        if img_tensor.shape[0] == 4:
            img_tensor = img_tensor[:3, :, :]
        elif img_tensor.shape[0] == 2:
            img_tensor = img_tensor[0, :, :].repeat(3, 1, 1)
        elif img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)

        # img = Image.fromarray(self.tensor_to_img(img_tensor), 'RGB')
        # img.save(f'raw_images/{os.path.basename(filename)}')

        return img_tensor, 0

    def __getitem__(self, i):
        if not self.lazy:
            return self.lazy_data[i]
        else:
            img_tensor, label = self.load_file(self.files[i])
            return img_tensor, 0

    def __len__(self) -> int:
        return len(self.files)