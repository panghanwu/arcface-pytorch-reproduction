from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def pad_to_square(image: np.ndarray, fill: int = 128) -> np.ndarray:
    h, w, c = image.shape
    if h > w:
        length = h - w
        padding = fill * np.ones((h, length, c), dtype=np.uint8)
        left, right = padding[:, :length//2], padding[:, length//2:]
        image = np.concatenate([left, image, right], axis=1)
    elif h < w:
        length = w - h
        padding = fill * np.ones((length, w, c), dtype=np.uint8)
        top, bottom = padding[:length//2], padding[length//2:]
        image = np.concatenate([top, image, bottom], axis=0)
    return image


class FaceRecognitionData(Dataset):
    """
    Directory structure:
        dataset/
        ├── images/
        │   ├── 001.jpg
        │   ├── 002.jpg
        │   └── ...
        └── mapping.csv

    Structure of mapping.txt
        image_file_name, id
    """
    def __init__(
        self, 
        dataset_dir: str, 
        image_size: int = 256, 
        augmentation: bool = False
    ) -> None:
        self.root = Path(dataset_dir)
        self.img_sz = (image_size, image_size)
        self.aug = augmentation

        df = pd.read_csv(self.root / 'mapping.csv', header=None)
        self.num_classes = df[1].max() + 1
        self.image_list = df[0].tolist()
        self.labels = df[1].tolist()
        self.to_tensor = transforms.ToTensor()

        if augmentation:
            crop_padding = int(0.3*image_size)
            self.transforms = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
                transforms.RandomRotation(degrees=20),
                transforms.RandomCrop(size=image_size, padding=crop_padding)
            ])

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index) -> tuple[Tensor, int]:
        fn = self.image_list[index]
        label = self.labels[index]
        image_path = (self.root / 'images') / fn
        img = cv2.imread(str(image_path))[..., ::-1]
        img = pad_to_square(img, fill=0)
        img = cv2.resize(img, self.img_sz)
        img = self.to_tensor(img)
        if self.aug:
            img = self.transforms(img)

        return img, label


def create_face_recognition_dataloader(
    root_dir: str, 
    image_size: int = 256,
    augmentation: bool = False,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    dataset = FaceRecognitionData(root_dir, image_size, augmentation)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )