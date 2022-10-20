
from pathlib import Path
from afrimap.data_preparation.constants import BANDS, MEAN, VARIANCE
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np 
import torch
from skimage import io
from torchvision import transforms
import random


def mask_label(label, scl):
       
    label = label.numpy()
    scl = scl.numpy()

    # also masks thin cirrus 
    scl_mask = ((scl >= 3) & (scl < 8)) | (scl > 9)

    scl_mask = np.broadcast_to(scl_mask, label.shape)
    # # print('here',scl_mask.shape, label.shape)
    label[~scl_mask] = 0
    return torch.Tensor(label)    

class LCNDataset(Dataset):
    def __init__(self, images_dir, label_dir, df, bands=BANDS):
        self.images_dir = images_dir
        self.label_dir = label_dir
        self.df = df
        self.bands = bands
        

    def __len__(self):
        return len(self.df)
    
    def open_image(self, source_path):
        imgs = []
        for band in self.bands:
            img = io.imread(Path(source_path, band + '.tif'))
            
            imgs.append(img)
        return np.stack(imgs, axis=2) 
    
    def __getitem__(self, idx):
       
        chip_path = str(self.df.loc[idx, 'path'])
        tile_date = str(self.df.loc[idx, 'date'])
        chip_name = str(self.df.loc[idx, 'chip_name'])
        season = str(self.df.loc[idx, 'season'])
        label_path = Path(self.label_dir,  'ref_landcovernet_v1_labels_'+chip_name, 'labels.tif')
        source_path = Path(self.images_dir, chip_path)

        image = torch.from_numpy(self.open_image(source_path).transpose((2, 0, 1)).astype(np.float32)).float()

        label = torch.from_numpy(io.imread(label_path).transpose((2, 0, 1)).astype(np.int32)).long()
    
        return image, label, chip_name, tile_date, str(label_path), season

class LandCoverNet32(LCNDataset):
    def __init__(self, images_dir, label_dir, df, bands=BANDS, transform=None, target_transform=None):
        super().__init__(images_dir, label_dir, df, bands=BANDS)
        self.target_transform = target_transform
        self.transform = transform
    
    def open_image(self, path):
        return io.imread(path)

    def __getitem__(self, idx):
           
        source_path = Path(self.images_dir, self.df.iloc[idx, 0])
        if(len(self.df.columns) > 1):
            label_path = Path(self.label_dir, self.df.iloc[idx, 1])
            label = torch.from_numpy(io.imread(label_path).transpose((2, 0, 1)).astype(np.int32)).long()
            unmasked_target = label[0].clone()
           
        else:
            label = -1
            target = -1
            unmasked_target = -1
            score = -1
        image = torch.from_numpy(self.open_image(source_path).transpose((2, 0, 1)).astype(np.float32)).float()
        if(type(label) != int ): label = mask_label(label, image[-1])
        #exclude the scene classification layer
        image = image[:-1, ]
        if self.transform:
            image = self.transform(image)

        if(type(label) != int ):  
            target = label[0]
            score = label[1]
            if(self.target_transform is not None):
                seed = random.randint(1, 1000)
                torch.manual_seed(seed)
                random.seed(seed)
                image = self.target_transform(image)
                random.seed(seed)
                torch.manual_seed(seed)
                label = self.target_transform(label)
        # only pixels with consensus score == 100 have a class label
        # pixels with less than 65 percent consensus score are labeled as ambigous(8)
        # pixels with score == 0 are masked out as invalid based on scl layer
            target[(score < 65) & (score != 0)] = 0
            target = target.long()
        return image, target, unmasked_target, score, str(source_path)


class CropDataLoaderLCN(DataLoader):
    
    def __init__(self, images_dir, label_dir, df, bands=None, batch_size=1, shuffle=True, validation_split=0.0, num_workers=1, transform = None):
        self.dataset = LCNDataset(images_dir, label_dir, df)
        super().__init__(self.dataset, batch_size, num_workers)

class LandCoverNetLoader32(DataLoader):
    
    def __init__(self, images_dir, label_dir, df, bands=None, batch_size=2, shuffle=True, validation_split=0.0, num_workers=1, transform = None):
        if(transform):
            transform = transforms.Compose([transforms.Normalize(MEAN, VARIANCE)])
        self.dataset = LandCoverNet32(images_dir, label_dir, df, transform=transform)
        super().__init__(self.dataset, batch_size, num_workers)

class LandCoverNetLoader32Aug(DataLoader):
    
    def __init__(self, images_dir, label_dir, df, bands=None, batch_size=2, shuffle=True, validation_split=0.0, num_workers=1, transform = None):
        transform = transforms.Compose([transforms.Normalize(MEAN, VARIANCE), transforms.GaussianBlur(kernel_size=(5, 5))])
        target_transform = torch.nn.Sequential(
                    transforms.RandomRotation(degrees=(0, 180)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    )
        target_transform = torch.jit.script(target_transform)
        self.dataset = LandCoverNet32(images_dir, label_dir, df, transform=transform, target_transform=target_transform)
        super().__init__(self.dataset, batch_size, num_workers)