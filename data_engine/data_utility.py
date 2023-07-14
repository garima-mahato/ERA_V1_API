from sklearn.utils import shuffle
import zipfile
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from PIL import Image
from io import BytesIO
import random
import os
from pathlib import Path

# from .mask_dataset import *
# from .mask_depth_dataset import *
#import .mask_dataset as md
from ..data_engine import mask_dataset as md, mask_depth_dataset as mdd

def loadZipToMem(bg_dir_path, depth_zip_file_name=None, mask_zip_file_name=None, is_depth=False, is_mask=False):
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    bg_path = Path(bg_dir_path)
    bg_paths = list(bg_path.glob("*.jpg"))
    bg_imgs = {}
    for bg in bg_paths:
      bg_imgs[str(str(bg).split('/')[-1].split('.')[0])] = bg
    # from zipfile import ZipFile
    if depth_zip_file_name is not None and is_depth:
      with zipfile.ZipFile(depth_zip_file_name,'r') as zf:
        depth_data = {name: zf.read(name) for name in zf.namelist()}
        depth_label = list((row.split('\t') for row in (depth_data['depth_label.txt']).decode("utf-8").split('\n') if len(row) > 0))
      zf.close()

    if mask_zip_file_name is not None and is_mask:
      with zipfile.ZipFile(mask_zip_file_name,'r') as input_zip:
        mask_data = {name: input_zip.read(name) for name in input_zip.namelist()}
        mask_label = list((row.split('\t') for row in (mask_data['data_label.txt']).decode("utf-8").split('\n') if len(row) > 0))
      input_zip.close()

    # label merging
    if depth_zip_file_name is not None and is_depth and mask_zip_file_name is not None and is_mask:
      mask_label = pd.DataFrame(mask_label, columns=['bg', 'fg', 'fg_bg', 'fg_bg_mask', 'x', 'y', 'width', 'height'])
      mask_label = mask_label.drop(columns=['fg', 'x', 'y', 'width', 'height'])
      depth_label = pd.DataFrame(depth_label, columns=['batch', 'fg_bg', 'fg_bg_depth'])
      depth_label = depth_label.drop(columns=['batch'])
      labels = mask_label.merge(depth_label, left_on='fg_bg', right_on='fg_bg', how='inner')

      # Data merging
      df1 = pd.DataFrame(depth_data.items(), columns=['name', 'data'])
      df2 = pd.DataFrame(mask_data.items(), columns=['name', 'data'])
      data = pd.concat([df1, df2], ignore_index=True)
      
      labels = labels.values.tolist()
      data = dict(zip(data.name, data.data))

      # from sklearn.utils import shuffle
      labels = shuffle(labels, random_state=0)
    
      return data, labels, bg_imgs
    elif depth_zip_file_name is None and not is_depth:
      # from sklearn.utils import shuffle
      mask_label = shuffle(mask_label, random_state=0)
      return mask_data, mask_label, bg_imgs
    else:
	  # from sklearn.utils import shuffle
      depth_label = shuffle(depth_label, random_state=0)
      return depth_data, depth_label, bg_imgs

def generate_subset(data, label, i, subset_size=10000):
    # label = shuffle(label, random_state=0)
    label = label[(subset_size*i) : ((subset_size*i)+subset_size)] #[:subset_size]
    df1 = pd.DataFrame(label, columns=['bg', 'fg', 'fg_bg', 'fg_bg_mask', 'x', 'y', 'width', 'height'])
    df2 = pd.DataFrame(data.items(), columns=['name', 'data'])
    df3 = df1.merge(df2, left_on='fg_bg', right_on='name', how='inner').loc[:,['name', 'data']]
    df4 = df1.merge(df2, left_on='fg_bg_mask', right_on='name', how='inner').loc[:,['name', 'data']]
    data = pd.concat([df3, df4], ignore_index=True)
    data = dict(zip(data.name, data.data))

    return data, label

def getTrainingTestingData(dataset_name, dataset_params, batch_size, bg_dir_path, depth_zip_file_name=None, mask_zip_file_name=None, is_depth=False, is_mask=False, i=0, subset=False, subset_size=10000, val_percent=0.3, num_workers=4):
    data, label, bg_imgs = loadZipToMem(bg_dir_path, depth_zip_file_name, mask_zip_file_name, is_depth, is_mask)
    datasets = {'maskDatasetMemory': {'data': md.maskDatasetMemory, 'transform': md}, 'maskDepthDatasetMemory': {'data': mdd.maskDepthDatasetMemory, 'transform': mdd}}
    if subset:
      data, label = generate_subset(data, label, i, subset_size=subset_size)
    
    print('Loaded ({0}).'.format(len(label)))
    # transformed_data = maskDepthDatasetMemory(data, label, transform=getDefaultTrainTransform())
    n_val = int(len(label) * val_percent)
    n_train = len(label) - n_val
    # train, val = random_split(transformed_data, [n_train, n_val])

    transformed_training = datasets[dataset_name]['data'](data, label[:n_train], bg_imgs, **dataset_params, transform=datasets[dataset_name]['transform'].getDefaultTrainTransform())
    transformed_testing = datasets[dataset_name]['data'](data, label[n_train:], bg_imgs, **dataset_params, transform=datasets[dataset_name]['transform'].getNoTransform())

    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True) if torch.cuda.is_available() else dict(shuffle=True, batch_size=8)

    return DataLoader(transformed_training, **dataloader_args), DataLoader(transformed_testing, **dataloader_args)
    # return DataLoader(train, **dataloader_args), DataLoader(val, **dataloader_args)