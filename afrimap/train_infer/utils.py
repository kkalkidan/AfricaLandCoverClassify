
from afrimap.data_preparation.dataset_dataloader import LandCoverNetLoader32, LandCoverNetLoader32Aug
from afrimap.train_infer.model import MAnet
from afrimap.data_preparation.constants import CLIMATE_CHIP_NAME_MAP

import torch
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_class_distribution(data_loader, num_classes=8, cover=None):
   
    bins_all = np.zeros(num_classes)
    for data in tqdm(data_loader):
        i = data[0]
        j = data[1]
        bins_all += np.bincount(j.flatten(), minlength=num_classes)
     
    return 1/np.log(bins_all)

def sample_train_val(images_path):
    images_df = pd.DataFrame({'image_path': os.listdir(images_path)})
    images_df['chip_name'] = images_df['image_path'].map(lambda x: x[:8])
    climate_zone_df = pd.read_csv(CLIMATE_CHIP_NAME_MAP)
    df = images_df.merge(climate_zone_df, how='inner',on='chip_name')
    df['label_path'] = df['image_path'].map(lambda x: x[:8] + x[10:])
    train = df.groupby('climate').sample(frac=0.75)[['image_path', 'label_path']]
    val = df[~df.index.isin(train.index)][['image_path', 'label_path']]
    train.to_csv('afrimap/train_infer/train.csv', index=False)
    val.to_csv('afrimap/train_infer/val.csv', index=False)

def get_train_val(images_path, label_path=None):
    
    if(label_path is None):
        infer_df = pd.read_csv("afrimap/train_infer/infer.csv")
        infer_loader = LandCoverNetLoader32(images_path, label_path, df=infer_df, batch_size=1, transform=True)
        return infer_loader
    sample_train_val(images_path)
    train_df = pd.read_csv("afrimap/train_infer/train.csv")
    val_df = pd.read_csv("afrimap/train_infer/val.csv")

    train_loader = LandCoverNetLoader32Aug(images_path, label_path, df=train_df, batch_size=4, transform=True)
    val_loader = LandCoverNetLoader32(images_path, label_path, df=val_df, batch_size=4, transform=True)
    print('++++++++Calculating weights+++++++')
    weights = get_class_distribution(train_loader)
    weights[0] = 0
    print('weights', weights)
    return train_loader, val_loader, weights

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def cross_entropy(weight):
    device, device_ids = prepare_device(1)
    return  torch.nn.CrossEntropyLoss(weight=torch.Tensor(weight).to(device), ignore_index=0)

# config model and other variables based on given learning rate 
def create_configs(lr=0.02, class_weights=None):
    # manually configed class weights, snow and 
    # build model architecture, then print to console
    model =  MAnet(encoder_weights="imagenet", in_channels=13, classes=8, decoder_use_batchnorm=True, activation=None)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(1)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    criterion = None
    if(class_weights is not None):
        # get function handles of loss and metrics
        criterion = cross_entropy(class_weights)
    
  
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=2e-4, momentum=0.9)

    # calculate gamma using last_epochs_lr = inital_lr * (gamma)^nb_epochs 
    gamma_manet = 0.95
    lr_scheduler =  torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma_manet)
    return model, device, criterion, optimizer, lr_scheduler


