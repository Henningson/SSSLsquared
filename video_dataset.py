import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import tqdm
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
        #os.mkdir(image_dir)
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

if __name__ == "__main__":
    VideoDataset("/home/nu94waro/Documents/Reinhard/", "Human_P181133_Broc14.mp4")