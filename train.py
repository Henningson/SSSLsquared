import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import Losses
from dataset import HLEDataset
from torch.utils.data import DataLoader
from model import SPLSS, LSQLocalization
from utils import (
    class_to_color,
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
    draw_points,
    draw_heatmap
)

# Hyperparameters etc.
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 10
NUM_WORKERS = 2
NUM_SAMPLEPOINTS = 100
LAMBDAS = [1.0, 1.0, 1.0]
IMAGE_HEIGHT = 512  # 1280 originally
IMAGE_WIDTH = 256  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
DATASET_BASE_DIR = "../HLEDataset/dataset/"
DATASET_TRAIN_KEYS = ["CF", "DD", "FH", "LS", "MK", "MS", "RH", "SS", "TM", "CM"]
DATASET_VALIDATION_KEYS = ["CF", "DD", "FH", "LS", "MK", "MS", "RH", "SS", "TM", "CM"]

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format='xy')
    )

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

    model = SPLSS(in_channels=1, out_channels=3, state_dict=torch.load("my_checkpoint.pth.tar")).to(DEVICE)
    CELoss = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1.0, 1.0], dtype=torch.float32, device=DEVICE))
    #CHMLoss = Losses.torch_loss_cHM()
    #L2Loss = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_ds = HLEDataset(base_path=DATASET_BASE_DIR, keys=DATASET_TRAIN_KEYS, transform=train_transform, is_train=True, pad_to=NUM_SAMPLEPOINTS)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True)
    val_ds = HLEDataset(base_path=DATASET_BASE_DIR, keys=DATASET_VALIDATION_KEYS, transform=val_transforms, is_train=False, pad_to=NUM_SAMPLEPOINTS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        
        #if epoch % 5 == 0:
        #    with torch.no_grad():
        #        val_image, gt_seg, gt_points = val_ds[0]

        #        pred_seg = model(val_image.unsqueeze(0).cuda())
                #pred_seg = pred_seg.argmax(axis=1)

                #pred_im = class_to_color(pred_seg, [torch.tensor([0, 0, 0], device=DEVICE), torch.tensor([0, 255, 0], device=DEVICE)])
                #draw_points(val_image.detach().cpu().numpy(), gt_points.detach().cpu().numpy(), pred_points.detach().cpu().numpy())
        #        loc = LSQLocalization(window_size=5)
        #        loc.test(pred_seg)
        #        draw_heatmap(pred_seg, axis=1)
    
        loop = tqdm(train_loader)
        for images, gt_seg, gt_points in loop:
            images = images.to(device=DEVICE)
            gt_seg = gt_seg.to(device=DEVICE)
            gt_points = gt_points.to(device=DEVICE)

            # forward
            with torch.cuda.amp.autocast():
                segmentation = model(images)
                loss_CE = CELoss(segmentation.float(), gt_seg.long())
                loss = loss_CE*LAMBDAS[0]

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())

        checkpoint = {"optimizer": optimizer.state_dict(),} | model.get_statedict()
        torch.save(checkpoint, "test_net.pth.tar")



if __name__ == "__main__":
    main()