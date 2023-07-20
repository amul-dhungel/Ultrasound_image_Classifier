"""
    Author: Mauro Mendez.
    Date: 22/11/2021.

    This file creates the dataset object to read the dataset and create dataloaders.
"""

import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from hyperparameters import parameters as params


# This class defines our custom dataset
class CustomDataset(Dataset):
    def __init__(self, ids, labels, transf):
        super().__init__()

        # Images IDS amd Labels
        self.ids = ids
        self.labels = labels

        # Transforms
        self.transforms = transf

    def __getitem__(self, index):
        # Get an ID of a specific image
        print("wow")
        id_img = self.ids[index]
        # Open Image
        img = Image.open(id_img).convert("L")
        print("hi")
        img = self.transforms(img)
        img = img.type(torch.float32)

        # Get Label
        label = torch.as_tensor(self.labels[index], dtype=torch.float16)

        return img, label

    def __len__(self):
        return len(self.ids)


# This class defines our custom data module used by pytorch lightning
class CustomDataModule(pl.LightningDataModule):


    def __init__(self, aug):
        super().__init__()
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']

        # Transformations
        self.train_transform = transforms.Compose([
            transforms.Resize(params['img_size'], transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-90, 90)),
            #transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.ToTensor(),
            transforms.Normalize(params['data_mean'], params['data_std'])])

        self.test_transform = transforms.Compose([
            transforms.Resize(params['img_size'], transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(params['data_mean'], params['data_std'])])
        
        self.do_aug = aug
        print(self.do_aug)
        # self.aug = True


    # Creates the datasets
    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == 'fit' or stage is None:
            images = pd.read_csv(params['train_data'])
            X_train, X_test, y_train, y_test = train_test_split(images['ID_IMG'].tolist(), images['LABEL'].tolist(), test_size=0.2, random_state=123, stratify=images['LABEL'].tolist(), shuffle=True)
            transfs = self.train_transform if self.do_aug else self.train_transform
            self.custom_train = CustomDataset(ids=X_train, labels=y_train, transf=transfs)
            self.custom_val = CustomDataset(ids=X_test, labels=y_test, transf=self.test_transform)

        if stage == 'test' or stage is None:
            images = pd.read_csv(params['test_data'])
            self.custom_test = CustomDataset(ids=images['ID_IMG'].tolist(),
                                        labels=images['LABEL'].tolist(), transf=self.test_transform)
    # Creates the dataloaders
    def train_dataloader(self):
        custom_train = DataLoader(self.custom_train, batch_size=self.batch_size,
                                num_workers=self.num_workers, pin_memory=True, shuffle=True)
        return custom_train

    def val_dataloader(self):
        custom_val = DataLoader(self.custom_val, batch_size=self.batch_size,
                                num_workers=self.num_workers, pin_memory=True, shuffle=False)
        return custom_val

    def test_dataloader(self):
        custom_test = DataLoader(self.custom_test, batch_size=self.batch_size,
                                num_workers=self.num_workers, pin_memory=True, shuffle=False)
        return custom_test

a = CustomDataModule(aug=True)
a.setup() 