import os
import numpy as np
from PIL import Image

def png_folder_to_array(names, img_dir, mask_dir):
    imgs, masks = [], []
    for name in names:
        img_path, mask_path = os.path.join(img_dir, name), os.path.join(mask_dir, name)
        
        img, mask = np.array(Image.open(img_path)), np.array(Image.open(mask_path))
        imgs.append(img), masks.append(mask)

    return imgs, masks
