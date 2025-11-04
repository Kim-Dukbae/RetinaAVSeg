from torch.utils.data import TensorDataset

class SegmentDataset:
  def __init__(self, imgs_path, masks_path):
    self.imgs_path = imgs_path
    self.masks_path = masks_path
    self.images = self.images_to_numpy( self.imgs_path )
    self.masks = self.images_to_numpy( self.masks_path )

  def images_to_numpy(self, path):
    image_list = []

    filenames = sorted(os.listdir( path ))
    for filename in filenames:
        image_path = os.path.join( path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = np.expand_dims(image, axis=-1)
        image_list.append(image)

    images_array = np.stack(image_list)
    return images_array

  def split(self, train_ratio= 0.75, val_ratio= 0.25, seed= 42):
    np.random.seed(seed)

    t, _, _, _ = self.images.shape

    idx_list = np.arange( t )
    train_idx = np.random.choice(idx_list, size= int( t * train_ratio), replace=False)
    val_idx = idx_list[~np.isin(idx_list, train_idx)]

    train_imgs, val_imgs = self.images[train_idx], self.images[val_idx]
    train_masks, val_masks = self.masks[train_idx], self.masks[val_idx]

    return train_imgs, val_imgs, train_masks, val_masks

  def encoding(self, mask):
    total_classes = np.unique(mask)

    mask_encoded = np.zeros_like(mask, dtype=np.int64)
    for idx, class_value in enumerate(total_classes):
        mask_encoded[mask == class_value] = idx

    return mask_encoded

  def image_crop(self, imgs, crop_size=128, stride=128):
    crops = []

    n, h, w, c = imgs.shape
    for idx in range(n):
        img = imgs[idx]
        for i in range(0, h - crop_size + 1, stride):
            for j in range(0, w - crop_size + 1, stride):
                img_crop = img[i:i+crop_size, j:j+crop_size]
                crops.append(img_crop)

    return np.stack(crops)

  def torch_dataset(self, imgs, masks):
    imgs = imgs / 255.0

    imgs = torch.from_numpy(imgs.transpose(0, 3, 1, 2)).float()
    masks = torch.from_numpy(masks.transpose(0, 3, 1, 2)).long()

    return  TensorDataset(imgs, masks)

  def test_mode(self, mask= False):
     if mask:
        return self.images, self.masks
     else:
        return self.images
     
  def flip(self, x):
    flip = np.flip(x, axis=2)         
    return np.concatenate([x, flip], axis=0) 
   
  def flop(self, x):
    flop = np.flip(x, axis=1)     
    return np.concatenate([x, flop], axis=0) 
