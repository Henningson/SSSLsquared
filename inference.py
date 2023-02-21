import torch
import albumentations as A
import argparse
import os
import utils
import numpy as np
import matplotlib.cm as cm
import viewer
from darktheme.widget_template import DarkPalette

from PyQt5.QtWidgets import QApplication
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import HLEPlusPlus
from models.LSQ import LSQLocalization
import Visualizer

import sys
sys.path.append("models/")

DEVICE = 'cpu'#"cuda" if torch.cuda.is_available() else "cpu"


def class_to_color(prediction, class_colors):
    prediction = np.expand_dims(prediction, 1)
    output = np.zeros((prediction.shape[0], 3, prediction.shape[-2], prediction.shape[-1]), dtype=np.float32)
    for class_idx, color in enumerate(class_colors):
        mask = class_idx == prediction
        curr_color = color.reshape(1, 3, 1, 1)
        segment = mask*curr_color # should have shape 1, 3, 100, 100
        output += segment

    return output


def main():
    parser = argparse.ArgumentParser(
                    prog = 'Inference for Deep Neural Networks',
                    description = 'Loads  as input, and visualize it based on the keys given in the config file.',
                    epilog = 'For question, generate an issue at: https://github.com/Henningson/SSSLsquared or write an E-Mail to: jann-ole.henningson@fau.de')
    parser.add_argument("-c", "--checkpoint", type=str, default="checkpoints/UNET_SS_TM_7862/")
    parser.add_argument("-d", "--dataset_path", type=str, default='../HLEDataset/dataset/')

    args = parser.parse_args()

    checkpoint_path = args.checkpoint


    if checkpoint_path == "" or not os.path.isdir(checkpoint_path):
        print("\033[93m" + "Please provide a viable checkpoint path")

    config = utils.load_config(os.path.join(checkpoint_path, "config.yml"))
    config['dataset_path'] = args.dataset_path

    val_transforms = A.load(os.path.join(checkpoint_path, "val_transform.yaml"), data_format='yaml')

    neuralNet = __import__(config["model"])
    model = neuralNet.Model(config, state_dict=torch.load(os.path.join(checkpoint_path, "model.pth.tar"))).to(DEVICE)
    dataset = __import__('dataset').__dict__[config['dataset_name']]
    val_ds = dataset(config, is_train=False, transform=val_transforms)

    val_loader = DataLoader(val_ds, 
                            batch_size=config['batch_size'],
                            num_workers=2, 
                            pin_memory=True, 
                            shuffle=False)
    
    localizer = LSQLocalization(local_maxima_window = config["maxima_window"], 
                                gauss_window = config["gauss_window"], 
                                heatmapaxis = config["heatmapaxis"], 
                                threshold = 0.5,
                                device=DEVICE)

    video = []

    segmentations = []
    gt_segmentations = []

    perFramePoints = []
    gt_points = []
    colors = [np.array(cm.get_cmap("Set2")(i*(1/config['num_classes']))[0:3]) for i in range(config['num_classes'])]

    model.eval()
    for images, gt_seg, keypoints in tqdm(val_loader, desc="Generating Video Frames"):
        images = images.to(device=DEVICE)
        gt_seg = gt_seg.to(device=DEVICE)

        prediction = model(images).softmax(dim=1)

        #import matplotlib.pyplot as plt
        #bla = Visualizer.Visualize2D(x=6)
        #bla.draw_heatmap(prediction, heatmapaxis=2)
        #plt.show(block=True)

        segmentation = prediction.argmax(dim=1)

        _, means, _ = localizer.estimate(prediction, segmentation=torch.bitwise_or(segmentation == 2, segmentation == 3))

        segmentation = class_to_color(segmentation.detach().cpu().numpy(), colors)
        gt_seg = class_to_color(gt_seg.detach().cpu().numpy(), colors)


        min_val = images.cpu().detach().min(axis=-1)[0].min(axis=-1)[0]
        max_val = images.cpu().detach().max(axis=-1)[0].max(axis=-1)[0]
        normalized_images = (images.detach().cpu() - min_val.unsqueeze(-1).unsqueeze(-1)) / (max_val.unsqueeze(-1).unsqueeze(-1) - min_val.unsqueeze(-1).unsqueeze(-1))

        video.append(normalized_images.numpy())
        segmentations.append(segmentation)
        gt_segmentations.append(gt_seg)

        for i in range(images.shape[0]):
            cleaned_keypoints = keypoints[i][~torch.isnan(keypoints[i]).any(axis=1)]
            gt_points.append(cleaned_keypoints[:, [1, 0]] - 1)

        if means is None:
            perFramePoints.append(np.zeros([0,2]))
            continue

        for mean in means:
            mean = mean.cpu().detach().numpy()
            mean = mean[~np.isnan(mean).any(axis=1)]
            perFramePoints.append(mean)


    video = np.moveaxis((np.concatenate(video, axis=0)*255).astype(np.uint8), 1, -1)
    segmentations = (np.moveaxis(np.concatenate(segmentations, axis=0), 1, -1)*255).astype(np.uint8)
    gt_segmentations = (np.moveaxis(np.concatenate(gt_segmentations, axis=0), 1, -1)*255).astype(np.uint8)
    error = (segmentations == gt_segmentations)*255

    app = QApplication(sys.argv) 
    app.setPalette(DarkPalette())
    mw = viewer.MainWindow(video, segmentations, gt_segmentations, error, perFramePoints, gt_points)
    mw.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()