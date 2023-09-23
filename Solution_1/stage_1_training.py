import os
import sys
import gc
import ast
import cv2
import time
import timm
import pickle
import random
import pydicom
import argparse
import warnings
import numpy as np
import pandas as pd
from glob import glob
import nibabel as nib
from PIL import Image
from tqdm import tqdm
import albumentations
from pylab import rcParams
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold, StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.data import DataLoader , Dataset

from monai.transforms import Resize
import monai.transforms as transforms

rcParams["figure.figsize"] = 20 , 8
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

import config

#Uses Monai
transforms_train = transforms.Compose([transforms.RandFlipd(keys=["image","mask"],prob=0.5, spatial_axis=1),
                                       transforms.RandFlipd(keys=["image","mask"], prob=0.5 , spatial_axis=2),
                                       transforms.RandAffined(keys=["image","mask"],translate_range=[int(x*y) for x , y in zip(config.image_sizes,[0.3,0.3,0.3])],
                                                             padding_mode="zeros", prob=0.7),
                                       transforms.RandGridDistortiond(keys=("image","mask"),prob=0.5,distort_limit=(-0.01,0.01),mode="nearest"),])
transforms_valid = transforms.Compose([])


#Uses Pandas
df_train = pd.read_csv(os.path.join(config.data_dir,"train.csv"))
mask_files = os.listdir(f"{config.data_dir}/segmentations")
df_mask = pd.DataFrame({
    "mask_file" : mask_files
})

df_mask["StudyInstanceUID"] = df_mask["mask_file"].apply(lambda x : x[:-4])
df_mask["mask_file"] = df_mask["mask_file"].apply(lambda x: os.path.join(config.data_dir,"segmentations",x))
df = df_train.merge(df_mask,on="StudyInstanceUID",how="left")
df["image_folder"] = df["StudyInstanceUID"].apply(lambda x: os.path.join(config.data_dir,"train_images",x))
df["mask_file"].fillna('',inplace=True)
df_seg = df.query('mask_file!= ""').reset_index(drop=True)


# K-Fold Stratified
kf = KFold(5)
df_seg["fold"] = -1
for fold , (train_idx,valid_idx) in enumerate(kf.split(df_seg,df_seg)) :
    df_seg.loc[valid_idx,"fold"] = fold

#df_seg.to_csv("df_seg.csv")

