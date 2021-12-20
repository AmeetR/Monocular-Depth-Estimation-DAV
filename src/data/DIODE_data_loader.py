import torch
import pandas as pd
from torch.utils.data import Dataset
import os
from data.DIODE_processing import DIODE_processing
import cv2
import numpy as np


class DIODEDataset(Dataset):
    def __init__(self, data_path, transform = None, train = True):
        self.data_path = data_path
        self.transform = transform
        
        if train:
            if not os.path.exists(os.path.join(data_path, 'train.csv')):
                diode_processor = DIODE_processing(data_path)
                diode_processor.generate_csv(diode_processor.train_dir, 'train.csv')
            self.data = pd.read_csv(os.path.join(self.data_path, 'train.csv'))
        else:
            if not os.path.exists(os.path.join(data_path, 'val.csv')):
                diode_processor = DIODE_processing(data_path)
                diode_processor.generate_csv(diode_processor.val_dir, 'val.csv')

            self.data = pd.read_csv(os.path.join(self.data_path, 'val.csv'))
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data['img_path'].iloc[idx]
        depth_path = self.data['depth_path'].iloc[idx]
        img = cv2.imread(img_path)
        if self.transform is not None:
            img = self.transform(img)
        depth = np.load(depth_path)
        return img, depth
            
        

        