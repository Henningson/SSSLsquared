from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import kornia as K

def create_random_labels_map(classes: int) -> Dict[int, Tuple[int, int, int]]:
    labels_map: Dict[int, Tuple[int, int, int]] = {}
    for i in classes:
        labels_map[i] = torch.randint(0, 255, (3, ))
    labels_map[0] = torch.zeros(3)
    return labels_map

def labels_to_image(img_labels: torch.Tensor, labels_map: Dict[int, Tuple[int, int, int]]) -> torch.Tensor:
    """Function that given an image with labels ids and their pixels intrensity mapping, creates a RGB
    representation for visualisation purposes."""
    assert len(img_labels.shape) == 2, img_labels.shape
    H, W = img_labels.shape
    out = torch.empty(3, H, W, dtype=torch.uint8)
    for label_id, label_val in labels_map.items():
        mask = (img_labels == label_id)
        for i in range(3):
            out[i].masked_fill_(mask, label_val[i])
    return out

def show_components(img, labels):
    color_ids = torch.unique(labels)
    labels_map = create_random_labels_map(color_ids)
    labels_img = labels_to_image(labels, labels_map)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,12))
    
    # Showing Original Image
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.axis("off")
    ax1.set_title("Orginal Image")
    
    #Showing Image after Component Labeling
    ax2.imshow(labels_img.permute(1,2,0).squeeze().numpy())
    ax2.axis('off')
    ax2.set_title("Component Labeling")


def getLargestConnectedComponent(labelmap):
    largestCC = labelmap == torch.histc(labelmap.int())[:1].argmax() + 1
    return largestCC

def connCompTest():
    img: np.ndarray = cv2.imread("cells_binary.png", cv2.IMREAD_GRAYSCALE)
    img_t: torch.Tensor = K.utils.image_to_tensor(img).cuda() # CxHxW
    img_t = img_t[None,...].float() / 255.
    print(img_t.shape)

    for i in range(1000):
        starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
        starter.record()
        labels_out = K.contrib.connected_components(img_t, num_iterations=150)
        bla = getLargestConnectedComponent(labels_out)
        ender.record()
        torch.cuda.synchronize()
        print(starter.elapsed_time(ender))



    show_components(img_t.cpu().numpy().squeeze(), bla.cpu().squeeze())
    #how_components(img_t.numpy().squeeze(), labels_out.squeeze())

    plt.show()

import albumentations as A
import utils
from dataset import HLEPlusPlus, SBHLEPlusPlus
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import torchvision
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    
    canvas = FigureCanvas(fig)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image_from_plot

def testAugmentations():
    config = utils.load_config("config.yml")

    train_transform = A.Compose(
        [
            A.Resize(height=config['image_height'], width=config['image_width']),
            A.Affine(translate_percent = 0.15, p=0.5),
            A.Rotate(limit=40, border_mode = cv2.BORDER_CONSTANT, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Perspective(scale=(0.05, 0.2), p=0.5),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            A.RandomBrightnessContrast(),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format='xy')
    )

    A.save(train_transform, "test.yaml", data_format="yaml")

    batch_size = 8
    train_ds = SBHLEPlusPlus(base_path=config['dataset_path'], keys=config['train_keys'].split(","), batch_size=batch_size, pad_keypoints=config['pad_keypoints'], transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=False)


    for im, seg, key in train_loader:
        print(im.max())
        grid = torchvision.utils.make_grid(im)
        image = show(grid)
        a = 1
        plt.show(block=True)

import random

if __name__ == "__main__":
    testAugmentations()
