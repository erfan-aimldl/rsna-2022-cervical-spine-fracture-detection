import os
from monai.transforms import Resize


debug = False

kernel_type = "timm3d_res18d_unet4b_128_128_128_dsv2_flip12_shift333p7_gd1p5_bs4_lr3e4_20x50ep"
load_kernel = True
load_last = True
n_block = 4
n_folds = 5
backbone = "restnet18d"

image_sizes = [128 , 128 , 128]

R = Resize(image_sizes)

init_lr = 3e-3
batch_size = 4
drop_rate = 0
drop_path_rate = 0
loss_weights = [1,1]
p_mixup = 0.1

data_dir = "/media/erfan/Extreme SSD/rsna-2022-cervical-spine-fracture-detection"
use_amp = True
num_workers = 4
out_dim = 7
n_epochs = 1000

log_dir = "./logs"
model_dir = "./models"
os.makedirs(log_dir,exist_ok=True)
os.makedirs(model_dir,exist_ok=True)

