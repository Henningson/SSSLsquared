import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import albumentations as A
import random
from albumentations.pytorch import ToTensorV2
import imgaug


class SharanHLE(Dataset):
    def __init__(self, config, is_train=True, transform=None):
        base_path = config['dataset_path']
        keys = config['train_keys'].split(",") if is_train else config['val_keys'].split(",")

        self.image_dirs = [os.path.join(base_path, key, "png/") for key in keys]
        self.mask_dirs = [os.path.join(base_path, key, "mask/") for key in keys]
        self.heatmap_dirs = [os.path.join(base_path, key, "heatmap/") for key in keys]
        self.laserpoints_dirs = [os.path.join(base_path, key, "points2d/") for key in keys]
        self.vocalfold_mask_dirs = [os.path.join(base_path, key, "vf_mask/") for key in keys]
        
        self.transform = transform
        self.train_test_split = 0.9
        self.pad_keypoints = 200
        
        self.images = self.make_list(self.image_dirs)
        self.laserpoint_masks = self.make_list(self.mask_dirs)
        self.laserpoint_hm = self.make_list(self.heatmap_dirs)
        self.laserpoints = self.make_list(self.laserpoints_dirs)
        self.vocalfold_masks = self.make_list(self.vocalfold_mask_dirs)


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
        laserpoint_hm_path = self.laserpoint_hm[index]
        vocalfold_mask_path = self.vocalfold_masks[index]
        point2D_path = self.laserpoints[index]

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)

        binseg = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        binseg[binseg == 255.0] = 1.0

        heatmap = np.array(Image.open(laserpoint_hm_path).convert("L"), dtype=np.float32)
        heatmap /= 255.0

        vocalfold_mask = np.array(Image.open(vocalfold_mask_path).convert("L"), dtype=np.float32)


        keypoints = np.load(point2D_path)
        keypoints[~(keypoints == 0).any(axis=1)]

        if self.transform is not None:
            augmentations = self.transform(image=image, masks=[binseg, heatmap, vocalfold_mask], keypoints=keypoints)
            
            image = augmentations["image"]
            seg = augmentations["masks"][0]
            heatmap = augmentations["masks"][1]
            transformed_vf_mask = augmentations["masks"][2]
            keypoints = augmentations["keypoints"]
        
        # Pad keypoints, such that tensor have all the same size
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        if keypoints.nelement() != 0:
            keypoints[transformed_vf_mask[keypoints[:, 1].long(), keypoints[:, 0].long()] == 0] = torch.nan
        to_pad = self.pad_keypoints - keypoints.shape[0]
        keypoints = torch.concat([keypoints, torch.zeros((to_pad, 2))], dim=0)
        keypoints[(keypoints == 0.0)] = torch.nan
        keypoints -= np.array([[1.0, 1.0]])
        
        return image, seg, heatmap, keypoints


class SequenceHLEPlusPlus(Dataset):
    def __init__(self, config, is_train=True, transform=None):
        base_path = config['dataset_path']
        keys = config['train_keys'].split(",") if is_train else config['val_keys'].split(",")
        pad_keypoints = config['pad_keypoints']

        self.image_dirs = [os.path.join(base_path, key, "png/") for key in keys]
        self.mask_dirs = [os.path.join(base_path, key, "mask/") for key in keys]
        self.glottal_mask_dirs = [os.path.join(base_path, key, "glottal_mask/") for key in keys]
        self.laserpoints_dirs = [os.path.join(base_path, key, "points2d/") for key in keys]
        self.vocalfold_mask_dirs = [os.path.join(base_path, key, "vf_mask/") for key in keys]


        self.sequence_length = config['sequence_length']
        self.transform = transform
        self.train_test_split = 0.9
        self.pad_keypoints = pad_keypoints

        self.images = self.make_list(self.image_dirs)
        self.laserpoint_masks = self.make_list(self.mask_dirs)
        self.glottal_masks = self.make_list(self.glottal_mask_dirs)
        self.vocalfold_masks = self.make_list(self.vocalfold_mask_dirs)
        self.laserpoints = self.make_list(self.laserpoints_dirs)

    '''def build_targets_dict(self, sequence_length):
        additional_targets_dict = {}
        keypoint_targets = {}
        for i in range(1, sequence_length, 1):
            additional_targets_dict['image' + str(i)] = 'image'
            additional_targets_dict['keypoints' + str(i)] = 'keypoints'
            additional_targets_dict['mask' + str(i)] = 'mask'

            targets_dict['image' + str(i)] = 'image'
            targets_dict['keypoints' + str(i)] = 'keypoints'
            targets_dict['mask' + str(i)] = 'mask'

        return targets_dict, additional_targets_dict'''

    '''def fill_targets_dict(self, image_list, seg_list, keypoint_list):
        targets_dict = {}
        for count, (image, seg, keypoint) in enumerate(zip(image_list, seg_list, keypoint_list)):
            if count == 0:
                targets_dict['image'] = image
                targets_dict['keypoints'] = keypoint
                targets_dict['mask'] = seg
            else:
                targets_dict['image' + str(count)] = image
                targets_dict['keypoints' + str(count)] = keypoint
                targets_dict['mask' + str(count)] = seg

        return targets_dict'''

    '''def targets_to_tensor(self, targets):
        images = []
        segmentation = []
        keypoints = []
        for i in range(self.sequence_length):
            if i == 0:
                images.append(targets['image'])
                segmentation.append(targets['mask'])
                keypoints.append(targets['keypoints'])
            else:
                images.append(targets['image' + str(i)])
                segmentation.append(targets['mask' + str(i)])
                keypoints.append(targets['keypoints' + str(i)])

        return images, segmentation, keypoints'''


    def make_list(self, dirs):
        list = []
        for dir in dirs:
            file_list = sorted(os.listdir(dir))

            for file_path in file_list:
                list.append(dir + file_path)

        return list

    def __len__(self):
        return len(self.images) - self.sequence_length - 1

    def __getitem__(self, index):
        image_list = []
        gt_segmentation_list = []
        keypoint_list = []
        seed = np.random.randint(0,99999)
        for i in range(self.sequence_length):
            img_path = self.images[index + i]
            mask_path = self.laserpoint_masks[index + i]
            glottal_mask_path = self.glottal_masks[index + i]
            vocalfold_mask_path = self.vocalfold_masks[index + i]
            point2D_path = self.laserpoints[index + i]

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
                random.seed(seed)
                imgaug.random.seed(seed)
                print(seed)
                augmentations = self.transform(image=image, masks=[seg, vocalfold_mask], keypoints=keypoints)
                
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
            #keypoints -= np.array([[1.0, 1.0]])

            image_list.append(image)
            gt_segmentation_list.append(seg)
            keypoint_list.append(keypoints)
        
        random.seed(0)
        imgaug.random.seed(0)
        return torch.concat(image_list), torch.stack(gt_segmentation_list), torch.stack(keypoint_list)

class HLEPlusPlus(Dataset):
    def __init__(self, config, is_train=True, transform=None):
        base_path = config['dataset_path']
        keys = config['train_keys'].split(",") if is_train else config['val_keys'].split(",")
        pad_keypoints = config['pad_keypoints']

        self.image_dirs = [os.path.join(base_path, key, "png/") for key in keys]
        self.mask_dirs = [os.path.join(base_path, key, "mask/") for key in keys]
        self.glottal_mask_dirs = [os.path.join(base_path, key, "glottal_mask/") for key in keys]
        self.laserpoints_dirs = [os.path.join(base_path, key, "points2d/") for key in keys]
        self.vocalfold_mask_dirs = [os.path.join(base_path, key, "vf_mask/") for key in keys]
        
        self.transform = transform
        self.train_test_split = 0.9
        self.pad_keypoints = pad_keypoints

        self.images = self.make_list(self.image_dirs)
        self.laserpoint_masks = self.make_list(self.mask_dirs)
        self.glottal_masks = self.make_list(self.glottal_mask_dirs)
        self.vocalfold_masks = self.make_list(self.vocalfold_mask_dirs)
        self.laserpoints = self.make_list(self.laserpoints_dirs)

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

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)

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
            augmentations = self.transform(image=image, masks=[seg, vocalfold_mask], keypoints=keypoints)
            
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
        keypoints -= np.array([[1.0, 1.0]])
        
        return image, seg, keypoints


# HLEPlusPlus optimized for sequences of batch-size
class SBHLEPlusPlus(HLEPlusPlus):
    def __init__(self, config, is_train=True, transform=None):
        batch_size = config['batch_size']
        do_shuffle = is_train
        
        self.batch_size = batch_size
        self.do_shuffle = do_shuffle
        super().__init__(config=config, is_train=is_train, transform=transform)

        self.transform = A.ReplayCompose(self.transform)
        self.replay = None
        if do_shuffle:
            self.images, self.laserpoint_masks, self.glottal_masks, self.vocalfold_masks, self.laserpoints = self.shuffle([self.images, self.laserpoint_masks, self.glottal_masks, self.vocalfold_masks, self.laserpoints], self.batch_size)

    def reduce_list_to_fit_batchsize(self, list, batch_size):
        if len(list) % batch_size != 0:
            return list[:-(len(list) % batch_size)]
        
        return list


    def make_list(self, dirs):
        list = []
        for dir in dirs:
            file_list = sorted(os.listdir(dir))
            file_list = self.reduce_list_to_fit_batchsize(file_list, self.batch_size)

            for file_path in file_list:
                list.append(dir + file_path)

        return list

    def restructure_list(self, list_, batch_size):
        new_list = []
        for i in range(len(list_) // batch_size):
            new_list.append(list_[i*batch_size:(i+1)*batch_size])

        return new_list

    def shuffle(self, lists_to_shuffle, batch_size):
        for list_ in lists_to_shuffle:
            assert len(list_) % batch_size == 0

        # Split every list into sublists of size batch size
        # For example batch_size = 2: [0, 1, 2, 3] -> [[0, 1], [2, 3]]
        restructured_lists = []
        for list_ in lists_to_shuffle:
            restructured_lists.append(self.restructure_list(list_, batch_size))

        # Zip all given lists, and shuffle these such that all the indices stay the same
        temp = list(zip(*restructured_lists))
        random.shuffle(temp)
        lists = list(zip(*temp))

        # Remove the sublists and return the shuffled lists
        reordered_lists = []
        for single_list in lists:
            recombined_list = []
            for sub_list in single_list:
                for item in sub_list:
                    recombined_list.append(item)
            reordered_lists.append(recombined_list)
        return reordered_lists


    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.laserpoint_masks[index]
        glottal_mask_path = self.glottal_masks[index]
        vocalfold_mask_path = self.vocalfold_masks[index]
        point2D_path = self.laserpoints[index]

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)

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
        keypoints -= np.array([[1.0, 1.0]])

        return image, seg, keypoints


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

if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    config = {
        'dataset_path': '../HLEDataset/dataset/',
        'batch_size': 4,
        'train_keys': "CF,CM,DD,FH,LS,RH,SS,TM",
        'val_keys': "MK,MS",
        'pad_keypoints': 200,
        'sequence_length': 6
        }
    val_transforms = A.load("train_transform_sequence.yaml", data_format='yaml')

    dataset = SequenceHLEPlusPlus(config, is_train=True, transform=val_transforms)
    val_loader = DataLoader(dataset, 
                            batch_size=config['batch_size'],
                            num_workers=1, 
                            pin_memory=True, 
                            shuffle=True)

    import Visualizer
    for images, gt_seg, keypoints in tqdm(val_loader, desc="Generating Video Frames"):
        a = 1#
        import matplotlib.pyplot as plt
        import Visualizer
        vis = Visualizer.Visualize2D(x=6, y=1)
        vis.draw_sequence(images)
        vis.show()
        plt.show(block=True)

    
    #segs = np.concatenate(seg_list, axis=0).astype(np.uint8)
    #num_classes = 4
    #elements_of_classes = []
    #for i in range(num_classes):
    #    elements_of_classes.append(np.count_nonzero(segs == i))

    #weights = []
    #for i in range(num_classes):
    #    weights.append(max(elements_of_classes) / elements_of_classes[i])