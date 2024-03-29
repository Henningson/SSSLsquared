import torch
import albumentations as A
import argparse
import os
import Utils.utils as utils
import numpy as np
import matplotlib.cm as cm
import Visualization.QTViewer as viewer
from darktheme.widget_template import DarkPalette


import os
from PyQt5.QtCore import QLibraryInfo

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)

from PyQt5.QtWidgets import QApplication
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.LSQ import LSQLocalization

import sys
sys.path.append("models/")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def class_to_color(prediction, class_colors):
    prediction = np.expand_dims(prediction, 1)
    output = np.zeros((prediction.shape[0], 3, prediction.shape[-2], prediction.shape[-1]), dtype=np.float32)
    for class_idx, color in enumerate(class_colors):
        mask = class_idx == prediction
        curr_color = color.reshape(1, 3, 1, 1)
        segment = mask*curr_color
        output += segment

    return output


def main():
    parser = argparse.ArgumentParser(
                    prog = 'Inference Viewer for SSSL^2',
                    description = 
                    'Visualize the prediction capabilities of a trained model. \n\
                    Please provide a path pointing to a checkpoint folder as well as the path ti. Usage \n\
                    _______Usage: \n\
                    A: Previous Frame \n\
                    D: Next Frame \n\
                    W: Toggle Predicted Keypoints (Green) \n\
                    S: Toggle Ground-Truth Keypoints (Blue) \n\
                    Scroll Mousewheel: Zoom In and Out',
                    epilog = 'If you have any questions, generate an issue at: https://github.com/Henningson/SSSLsquared or write an E-Mail to: jann-ole.henningson@fau.de')
    parser.add_argument("-c", "--checkpoint", type=str, default="checkpoints/UNETFULL_MKMS_9400/", help="The path pointing to the checkpoint folder.")
    parser.add_argument("-d", "--dataset_path", type=str, default='../HLEDataset/dataset/',       help="The path pointing to the dataset folder.")

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
    
    
    try:
        old_sequence_length = config["sequence_length"]
        config["sequence_length"] = 1
    except:
        old_sequence_length = 1
        config["sequence_length"] = 1


    val_ds = dataset(config, is_train=False, transform=val_transforms)
    val_loader = DataLoader(val_ds, 
                            batch_size=config["batch_size"]*old_sequence_length,
                            num_workers=2, 
                            pin_memory=True, 
                            shuffle=False)

    localizer = LSQLocalization(local_maxima_window = 7, 
                                gauss_window = config["gauss_window"], 
                                heatmapaxis = config["heatmapaxis"], 
                                threshold = 0.5,
                                device=DEVICE)


    colors = [np.array(cm.get_cmap("Set2")(i*(1/config['num_classes']))[0:3]) for i in range(config['num_classes'] - 1)]
    model.eval()

    if config["model"] == "ChannelDepthUNet" or config["model"] == "TwoDtoThreeDNet" or config["model"] == "TwoDtoThreeDNetNOBOT":
        visChannelwise(val_loader, model, localizer, config["batch_size"], old_sequence_length, colors)
    else:
        vis(val_loader, model, localizer, colors)

def vis(val_loader, model, localizer,  colors):
    video = []

    segmentations = []
    gt_segmentations = []
    perFramePoints = []
    gt_points = []

    for images, gt_seg, keypoints in tqdm(val_loader, desc="Generating Video Frames"):
        images = images.to(device=DEVICE)
        gt_seg = gt_seg.to(device=DEVICE)


        prediction = None
        segmentation = None
        prediction = model(images).softmax(dim=1)

        segmentation = prediction.argmax(dim=1)

        _, means, _ = localizer.estimate(prediction, segmentation=torch.bitwise_or(segmentation == 2, segmentation == 3))


        segmentation = torch.where(segmentation == 3, 2, segmentation)
        gt_seg = torch.where(gt_seg == 3, 2, gt_seg)
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
            gt_points.append(cleaned_keypoints[:, [1, 0]])

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

def visChannelwise(val_loader, model, localizer, batch_size, sequence_length, colors):

    video = []

    segmentations = []
    gt_segmentations = []
    predictions = []

    perFramePoints = []
    gt_points = []

    for images, gt_seg, keypoints in tqdm(val_loader, desc="Generating Video Frames"):
        if images.shape[0] != batch_size*sequence_length:
            continue

        images = images.to(device=DEVICE).reshape(batch_size, sequence_length, images.shape[-2], images.shape[-1])
        gt_seg = gt_seg.to(device=DEVICE).reshape(batch_size, sequence_length, gt_seg.shape[-2], gt_seg.shape[-1])
        keypoints = keypoints.reshape(batch_size, sequence_length, keypoints.shape[-2], keypoints.shape[-1])

        pred_seg = model(images).moveaxis(1, 2)
        softmax = pred_seg.softmax(dim=2)
        segmentation = softmax.argmax(dim=2)

        _, means, _ = localizer.estimate(softmax.flatten(0, 1), segmentation=torch.bitwise_or(segmentation == 2, segmentation == 3).flatten(0, 1))

        segmentation = segmentation.flatten(0, 1)
        images = images.flatten(0, 1)
        keypoints = keypoints.flatten(0, 1)
        gt_seg = gt_seg.flatten(0, 1)

        segmentation = torch.where(segmentation == 3, 2, segmentation)
        gt_seg = torch.where(gt_seg == 3, 2, gt_seg)
        segmentation = class_to_color(segmentation.detach().cpu().numpy(), colors)
        gt_seg = class_to_color(gt_seg.detach().cpu().numpy(), colors)

        min_val = images.cpu().detach().min(axis=-1)[0].min(axis=-1)[0]
        max_val = images.cpu().detach().max(axis=-1)[0].max(axis=-1)[0]
        normalized_images = (images.detach().cpu() - min_val.unsqueeze(-1).unsqueeze(-1)) / (max_val.unsqueeze(-1).unsqueeze(-1) - min_val.unsqueeze(-1).unsqueeze(-1))

        #predictions.append(prediction.detach().cpu().numpy())
        video.append(normalized_images.numpy())
        segmentations.append(segmentation)
        gt_segmentations.append(gt_seg)

        for i in range(keypoints.shape[0]):
            cleaned_keypoints = keypoints[i][~torch.isnan(keypoints[i]).any(axis=1)]
            #print(cleaned_keypoints.shape)
            gt_points.append((cleaned_keypoints[:, [1, 0]] - 1))

        if means is None:
            perFramePoints.append(np.zeros([0,2]))
            print("Continuing")
            continue

        for mean in means:
            mean = mean.cpu().detach().numpy()
            mean = mean[~np.isnan(mean).any(axis=1)]
            perFramePoints.append(mean)



    #predictions_concat = np.concatenate(predictions, axis=0)
    video = np.repeat(np.expand_dims(np.concatenate(video, axis=0)*255, -1), 3, axis=-1).astype(np.uint8)
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