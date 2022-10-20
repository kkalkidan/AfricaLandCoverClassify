import os
from unicodedata import name
import pandas as pd
from pathlib import Path
from afrimap.data_preparation.constants import CLIMATE_CHIP_NAME_MAP,  LCN_IMAGES
from afrimap.data_preparation.cropper import lcn_crop
from afrimap.data_preparation.dataset_dataloader import CropDataLoaderLCN

from skimage import io
import numpy as np
import torch


def mask_label(label, scl):
       
    label = label.numpy()
    scl = scl.numpy()

    scl_mask = ((scl >= 3) & (scl < 8)) | (scl > 9)

    scl_mask = np.broadcast_to(scl_mask, label.shape)
    label[~scl_mask] = 0
    return torch.Tensor(label)


def train_prep(image_path, label_path):
    files = os.listdir(image_path)
    columns=['path', 'chip_name', 'date', 'valid_pixels_count']
    if(os.path.exists('afrimap/data_preparation/all_chips.csv')):
        df = pd.read_csv('afrimap/data_preparation/all_chips.csv')
    else:
        scl_values = []
        for f in files:
            try:
                file_path = Path(image_path, f, 'SCL.tif')
                scl = io.imread(Path(image_path, f, 'SCL.tif'))
                scl = ((scl >= 3) & (scl < 8)) | (scl > 9)
                f2 = f.replace('ref_landcovernet_v1_source_', '')
                chip_name = f2[:8]
                date = f2[9:]
                scl_values.append([f, chip_name, pd.to_datetime(date), scl.sum()])
            except:
                print('SCL.tiff file not found >>> skipping')
        df = pd.DataFrame(scl_values, columns=columns) 
        df.to_csv('afrimap/data_preparation/all_chips.csv') 
    
    month_to_season = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4, 12: 1}
    df['season'] = pd.to_datetime(df['date']).dt.month.map(month_to_season) 
    df = df.sort_values(['valid_pixels_count'], ascending=False).groupby(['chip_name', 'season']).first().reset_index()
    dataloader = CropDataLoaderLCN(image_path, label_path, df)
    # writes cropped dataset to afrimap/data_preparation/lcn32_dataset_images/labels folder
    lcn_crop(data_loader=dataloader)
    
    

      
