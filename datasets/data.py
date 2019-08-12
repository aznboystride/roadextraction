import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from datasets.augmentation import default_loader
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, train_dir, image_filenames):
        self.image_filenames = image_filenames
        self.train_dir = train_dir
    
    def __getitem__(self, index):
        image, mask = default_loader(index, self.train_dir, self.image_filenames)
        return image, mask
    
    def __len__(self):
        return len(self.image_filenames)
