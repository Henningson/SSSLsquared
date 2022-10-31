import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from dataset import HLEDataset
from torch.utils.data import DataLoader
from model import SPLSS, LSQLocalization
import Visualizer

# Hyperparameters etc.
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 10
NUM_WORKERS = 2
NUM_SAMPLEPOINTS = 150
LAMBDAS = [1.0, 1.0, 1.0]
IMAGE_HEIGHT = 512  # 1280 originally
IMAGE_WIDTH = 256  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
DATASET_BASE_DIR = "../HLEDataset/dataset/"
DATASET_TRAIN_KEYS = ["CF", "DD", "FH", "LS", "MK", "MS", "RH", "SS", "TM", "CM"]
DATASET_VALIDATION_KEYS = ["CM", "DD", "FH", "LS", "MK", "MS", "RH", "SS", "TM", "CM"]

def main():
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format='xy')
    )

    model = SPLSS(in_channels=1, out_channels=3, state_dict=torch.load("assets/pretrained.pth.tar")).to(DEVICE)
    val_ds = HLEDataset(base_path=DATASET_BASE_DIR, keys=DATASET_VALIDATION_KEYS, transform=val_transforms, is_train=False, pad_to=NUM_SAMPLEPOINTS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False)
    vis = Visualizer.Visualize2D(batched=True, x=4, y=1)

    loc = LSQLocalization(window_size=5)
    loop = tqdm(val_loader)
    for images, gt_seg, gt_points in loop:
        images = images.to(device=DEVICE)
        gt_seg = gt_seg.to(device=DEVICE)
        gt_points = gt_points.to(device=DEVICE)

        #utils.draw_images(images, 1, 1)

        # forward
        with torch.cuda.amp.autocast():

            #torch.cuda.synchronize()
            #t0 = time.time()
            #torch.cuda.synchronize()
            segmentation = model(images).softmax(dim=1)
            sigma, mean, amplitude = loc.test(segmentation)

            #segmentation = segmentation.argmax(dim=1)
            vis.draw_images(images)
            vis.draw_segmentation(segmentation.argmax(dim=1), 3, opacity=0.8)
            #vis.draw_heatmap(segmentation, heatmapaxis=1)
            #vis.draw_images(images)
            #vis.draw_points(mean)
            vis.show()


if __name__ == "__main__":
    main()

    #batch = torch.randn((8, 200, 200))