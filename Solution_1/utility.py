import config
import os
from glob import glob
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import cv2

VISUALIZE = True
def load_dicom(path) :
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = cv2.resize(data ,(config.image_sizes[0] , config.image_sizes[1]) , interpolation = cv2.INTER_LINEAR)
    if VISUALIZE :
        plt.imshow(data,cmap="gray")
        plt.colorbar()
        plt.title("DICOM Image")
        plt.show()
    return data

def load_dicom_line_par(path) :
    t_paths = sorted(glob(os.path.join(path,"*")),key = lambda x : int(x.split("/")[-1].split(".")[0]))
    n_scans = len(t_paths)
    indices = np.quantile(list(range(n_scans)),np.linspace(0.,1.,config.image_sizes[2])).round().astype(int)
    t_paths= [t_paths[i] for i in indices]
    images = []
    for filename in t_paths :
        images.append(load_dicom(filename))

    images = np.stack(images,-1)

    images = images - np.min(images)
    images = images / (np.max(images) + 1e-4)
    images = (images * 255).astype(np.uint8)

    return images

def load_sample(row , has_mask=True) :
    image = load_dicom_line_par(row.image_folder)
    if image.ndim < 4 :
        image = np.expand_dims(image , 0).repeat(3,0)

    if has_mask :
        mask_org = nib.load(row.mask_file).get_fdata()
        shape = mask_org.shape
        mask_org = mask_org.transpose(1,0,2)[::-1,:,::-1]
        mask = np.zeros((7,shape[0],shape[1],shape[2]))
        for cid in range(7) :
            mask[cid] = (mask_org==(cid+1))
            mask = mask.astype(np.uint8) * 255
            mask = R(mask).numpy()
            return image ,mask
    else :
        return image



if __name__=="__main__" :

    test_image = os.path.join(config.data_dir,"train_images","1.2.826.0.1.3680043.14")
    load_dicom_line_par(test_image)