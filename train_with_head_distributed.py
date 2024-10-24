from glob import glob
import os
import clip
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
import wandb
import matplotlib.pyplot as plt

# Logging setup
logging.basicConfig(filename="model_info1.log", format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# WandB setup
device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "/home/jayant/neurodiscovery/data_mri/mri_image/Train"
num_classes = 2

# Configurations
CONFIG = dict(
    clip_type='ViT-L/14@336px',
    epochs=100,
    max_lr=3e-5,
    pct_start=0.2,
    anneal_strategy='linear',
    weight_decay=0.0002,
    batch_size=32,
    dropout=0.5,
    hid_dim=512,
    activation='relu'
)

run = wandb.init(project="ventricle_classifier", config=CONFIG)
CONFIG = wandb.config

# Activation functions
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

get_activation = {
    'q_gelu': QuickGELU,
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'leaky_relu': nn.LeakyReLU
}

# Helper class to track metrics
class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Dataset class with Albumentations
class ImageDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.im_paths = glob(os.path.join(root_dir, "*", "*"))
        self.augment = augment

        # Albumentations pipelines
        self.transform = A.Compose([
            A.RandomResizedCrop(336, 336, scale=(0.8, 1.0)),
            A.Rotate(limit=30, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        self.val_transform = A.Compose([
            A.Resize(336, 336),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        # Create label map based on folder names
        self.classes = sorted({os.path.basename(os.path.dirname(p)) for p in self.im_paths})
        self.label_dict = {name: idx for idx, name in enumerate(self.classes)}

        logger.info(f"Label mapping: {self.label_dict}")

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        im_path = self.im_paths[idx]
        folder_name = os.path.basename(os.path.dirname(im_path))
        label = self.label_dict[folder_name]

        img = cv2.imread(im_path)
        if self.augment:
            img = self.transform(image=img)['image']
        else:
            img = self.val_transform(image=img)['image']

        return img, label

# Train/test split
def load_split_train_test(datadir, valid_size=0.2):
    dataset = ImageDataset(datadir, augment=True)

    indices = list(range(len(dataset)))
    np.random.shuffle(indices)

    split = int(np.floor(valid_size * len(dataset)))
    train_idx, test_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = DataLoader(dataset, sampler=train_sampler, batch_size=CONFIG['batch_size'],
                             pin_memory=True, drop_last=True, num_workers=8)
    testloader = DataLoader(dataset, sampler=test_sampler, batch_size=CONFIG['batch_size'],
                            pin_memory=True, drop_last=False, num_workers=8)

    return trainloader, testloader

# Freeze layers
def freeze_layer(module, unfreezed_layer=4):
    for param in module.parame