import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

class HLEDataset(Dataset):
    def __init__(self, base_path, keys, transform=None, is_train=True, train_test_split=0.9, pad_to=150):
        self.image_dirs = [os.path.join(base_path, key, "png/") for key in keys]
        self.mask_dirs = [os.path.join(base_path, key, "mask/") for key in keys]
        self.glottal_mask_dirs = [os.path.join(base_path, key, "glottal_mask/") for key in keys]
        self.laserpoints_dirs = [os.path.join(base_path, key, "points2d/") for key in keys]
        self.transform = transform

        self.is_train = is_train
        self.train_test_split = 0.9

        self.images = self.make_list(self.image_dirs)
        self.masks = self.make_list(self.mask_dirs)
        self.glottal_masks = self.make_list(self.glottal_mask_dirs)
        self.laserpoints = self.make_list(self.laserpoints_dirs)
        self.pad_to = pad_to

    def make_list(self, dirs):
        list = []
        for dir in dirs:
            file_list = sorted(os.listdir(dir))

            if self.is_train:
                file_list = file_list[:int(len(file_list) * self.train_test_split)]
            else:
                file_list = file_list[int(len(file_list) * self.train_test_split):]

            for file_path in file_list:
                list.append(dir + file_path)

        return list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.masks[index]
        glottal_mask_path = self.glottal_masks[index]
        point2D_path = self.laserpoints[index]

        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
        
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0
        glottal_mask = np.array(Image.open(glottal_mask_path).convert("L"), dtype=np.float32)
        glottal_mask[glottal_mask == 255.0] = 2.0

        points2D = np.load(point2D_path)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask, glottal_mask=glottal_mask, keypoints=points2D)
            image = augmentations["image"]
            mask = augmentations["mask"]
            glottal_mask = augmentations["glottal_mask"]
            keypoints = augmentations["keypoints"]

        # Set class labels
        x = np.zeros(glottal_mask.shape, dtype=np.float32)
        x[mask == 1.0] = 1
        x[glottal_mask == 1.0] = 2
        
        # Pad keypoints, such that tensor have all the same size
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        to_pad = self.pad_to - keypoints.shape[0]
        keypoints = torch.concat([keypoints, torch.zeros((to_pad, 2))], dim=0)


        return image, x, keypoints