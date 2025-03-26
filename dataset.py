from torch.utils.data import Dataset
from torchvision import datasets

import os


class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.data = datasets.ImageFolder(root, transform=transform)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        return img, label


class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.image_paths = [
            os.path.join(root, img) for img in os.listdir(root)
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        return img_path
