import os
from pathlib import Path

import cv2
import numpy as np
import torchvision
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
import pandas as pd


class Dataset(data.Dataset):

    def __init__(self, root, phase='train', input_shape=(128, 128)):
        self.phase = phase
        self.input_shape = input_shape

        root = Path(root)

        self.df = pd.read_csv(root / 'mapping.csv', header=None)

        self.img_paths = [str((root / 'images') / fn) for fn in self.df[0]]
        self.labels = self.df[1].to_list()

        normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])

        if self.phase == 'train':
            self.transforms = T.Compose([
                T.RandomCrop(self.input_shape),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.CenterCrop(self.input_shape),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        data = Image.open(img_path)
        data = self.transforms(data)
        label = np.int32(self.labels[index])
        return data.float(), label

    def __len__(self):
        return len(self.img_paths)
