import os
import cv2
import pandas as pd
import numpy as np
import easydict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader

class MaskDataset(Dataset):
    
    def __init__(self, df, train, transform=None):
        
        self.df = df
        self.train = train
        
        if self.train:
            self.img_path = df["img_path"]
            self.label = df["label"]
        else:
            self.img_path = df["img_path"]
            
        self.transform = transform
    
    def __getitem__(self, idx):
        
        image = Image.open(self.img_path[idx])
        image = np.asarray(image)
        
        if self.transform:
            image = self.transform(image=image)["image"]
        
        if self.train:
            label = self.label[idx]
            return image, label
        else:
            return image
    
    def __len__(self):
        return len(self.df)
