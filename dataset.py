import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import albumentations as A
import random
from albumentations.pytorch import ToTensorV2


class HLEPlusPlus(Dataset):
    def __init__(self, base_path, keys, pad_keypoints = 200, transform=None):
        self.image_dirs = [os.path.join(base_path, key, "png/") for key in keys]
        self.mask_dirs = [os.path.join(base_path, key, "mask/") for key in keys]
        self.glottal_mask_dirs = [os.path.join(base_path, key, "glottal_mask/") for key in keys]
        self.laserpoints_dirs = [os.path.join(base_path, key, "points2d/") for key in keys]
        self.vocalfold_mask_dirs = [os.path.join(base_path, key, "vf_mask/") for key in keys]
        
        self.transform = A.ReplayCompose(transform)
        self.train_test_split = 0.9
        self.pad_keypoints = pad_keypoints

        self.images = self.make_list(self.image_dirs)
        self.laserpoint_masks = self.make_list(self.mask_dirs)
        self.glottal_masks = self.make_list(self.glottal_mask_dirs)
        self.vocalfold_masks = self.make_list(self.vocalfold_mask_dirs)
        self.laserpoints = self.make_list(self.laserpoints_dirs)
        self.replay = None

    def make_list(self, dirs):
        list = []
        for dir in dirs:
            file_list = sorted(os.listdir(dir))

            for file_path in file_list:
                list.append(dir + file_path)

        return list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.laserpoint_masks[index]
        glottal_mask_path = self.glottal_masks[index]
        vocalfold_mask_path = self.vocalfold_masks[index]
        point2D_path = self.laserpoints[index]

        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32)

        laserpoints = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        laserpoints[laserpoints == 255.0] = 1.0

        glottal_mask = np.array(Image.open(glottal_mask_path).convert("L"), dtype=np.float32)
        glottal_mask[glottal_mask == 255.0] = 2.0

        vocalfold_mask = np.array(Image.open(vocalfold_mask_path).convert("L"), dtype=np.float32)
        vocalfold_mask[vocalfold_mask == 255.0] = 3.0

        keypoints = np.load(point2D_path)
        keypoints[~(keypoints == 0).any(axis=1)]
        
        # Set class labels
        seg = np.zeros(glottal_mask.shape, dtype=np.float32)
        seg[vocalfold_mask == 3.0] = 2
        seg[laserpoints == 1.0] = 3
        seg[glottal_mask == 2.0] = 1

        transformed_vf_mask = torch.zeros(seg.shape, dtype=torch.int)

        if self.transform is not None:
            if index % self.batch_size == 0:
                augmentations = self.transform(image=image, masks=[seg, vocalfold_mask], keypoints=keypoints)
                self.replay = augmentations['replay']
            else:
                augmentations = A.ReplayCompose.replay(self.replay, image=image, masks=[seg, vocalfold_mask], keypoints=keypoints)
            
            
            image = augmentations["image"]
            seg = augmentations["masks"][0]
            transformed_vf_mask = augmentations["masks"][1]
            keypoints = augmentations["keypoints"]
        
        # Pad keypoints, such that tensor have all the same size
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        if keypoints.nelement() != 0:
            keypoints[transformed_vf_mask[keypoints[:, 1].long(), keypoints[:, 0].long()] == 0] = torch.nan
        to_pad = self.pad_keypoints - keypoints.shape[0]
        keypoints = torch.concat([keypoints, torch.zeros((to_pad, 2))], dim=0)
        keypoints[(keypoints == 0.0)] = torch.nan

        return image, seg, keypoints


# HLEPlusPlus optimized for sequences of batch-size
class SBHLEPlusPlus(HLEPlusPlus):
    def __init__(self, base_path, keys, batch_size, do_shuffle=True, pad_keypoints = 200, transform=None):
        self.batch_size = batch_size
        self.do_shuffle = do_shuffle
        super().__init__(base_path, keys, pad_keypoints, transform)

        if do_shuffle:
            self.images = self.shuffle(self.images, self.batch_size)
            self.laserpoint_masks = self.shuffle(self.laserpoint_masks, self.batch_size)
            self.glottal_masks = self.shuffle(self.glottal_masks, self.batch_size)
            self.vocalfold_masks = self.shuffle(self.vocalfold_masks, self.batch_size)
            self.laserpoints = self.shuffle(self.laserpoints, self.batch_size)

    def reduce_list_to_fit_batchsize(self, list, batch_size):
        return list[:-(len(list) % batch_size)]


    def make_list(self, dirs):
        list = []
        for dir in dirs:
            file_list = sorted(os.listdir(dir))
            file_list = self.reduce_list_to_fit_batchsize(file_list, self.batch_size)

            for file_path in file_list:
                list.append(dir + file_path)

        return list

    def shuffle(self, list_to_shuffle, batch_size):
        assert len(list_to_shuffle) % batch_size == 0


        new_list = []
        for i in range(len(list_to_shuffle) // batch_size):
            new_list.append(list_to_shuffle[i*batch_size:(i+1)*batch_size])

        random.shuffle(new_list)
        return [item for sublist in new_list for item in sublist]


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

        keypoints = np.load(point2D_path)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask, glottal_mask=glottal_mask, keypoints=keypoints)
            image = augmentations["image"]
            mask = augmentations["mask"]
            glottal_mask = augmentations["glottal_mask"]
            keypoints = augmentations["keypoints"]

        # Set class labels
        seg = np.zeros(glottal_mask.shape, dtype=np.float32)
        seg[mask == 1.0] = 1
        seg[glottal_mask == 2.0] = 2
        
        # Pad keypoints, such that tensor have all the same size
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        to_pad = self.pad_to - keypoints.shape[0]
        keypoints = torch.concat([keypoints, torch.zeros((to_pad, 2))], dim=0)


        return image, seg, keypoints


class JonathanDataset(Dataset):
    def __init__(self, base_path, transform=None, is_train=True):
        self.train_image_dir = os.path.join(base_path, "train_images")
        self.train_label_dir = os.path.join(base_path, "train_masks", "all_4")
        self.test_image_dir = os.path.join(base_path, "val_images")
        self.test_label_dir = os.path.join(base_path, "val_masks", "all_4")
        self.transform = transform

        self.is_train = is_train

        self.images = self.make_list(self.train_image_dir) if self.is_train else self.make_list(self.test_image_dir)
        self.masks = self.make_list(self.train_label_dir) if self.is_train else self.make_list(self.test_label_dir)

    def make_list(self, directory):
        return [os.path.join(directory, file) for file in sorted(os.listdir(directory))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.masks[index]

        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
        seg = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=seg)
            image = augmentations["image"]
            seg = augmentations["mask"]

        return image, seg


class VideoDataset(Dataset):
    def __init__(self, path, videoname):
        self.image_dir = os.path.join(path, "png/")
        self.video_file = os.path.join(path, videoname)

        if not os.path.isdir(self.image_dir):
            if not os.path.isfile(self.video_file):
                assert FileNotFoundError("File: {0} does not exist.".format(self.video_file))

                self.create_image_data(self.image_dir, self.video_file)


        self.images = self.make_list(self.image_dir)


        self.transform = A.Compose(
        [
            A.Resize(height=512, width=256),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
        )

    def make_list(self, directory):
        return [os.path.join(directory, file) for file in sorted(os.listdir(directory))]

    def create_image_data(self, image_dir, video_file):
        print("ffmpeg -i {0} {1}/%05d.png".format(video_file, image_dir))
        os.system("ffmpeg -i {0} {1}%05d.png".format(video_file, image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
        
        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]
        
        return image



class ReinhardDataset(Dataset):
    def __init__(self, base_path, transform=None, is_train=True, train_test_split=0.9):
        self.image_dir = os.path.join(base_path, "png/")
        self.laserpoints_dir = os.path.join(base_path, "laserpoints/")
        self.transform = transform

        self.is_train = is_train
        self.train_test_split = 0.9

        self.images = self.make_list(self.image_dir)
        self.laserpoints = self.make_list(self.laserpoints_dir)

    def make_list(self, directory):
        file_list = [os.path.join(directory, file) for file in sorted(os.listdir(directory))]

        if self.is_train:
            file_list = file_list[:int(len(file_list) * self.train_test_split)]
        else:
            file_list = file_list[int(len(file_list) * self.train_test_split):]

        return file_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.laserpoints[index]

        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        # Set class labels
        x = np.zeros(mask.shape, dtype=np.float32)
        x[mask == 1.0] = 1

        return image, x