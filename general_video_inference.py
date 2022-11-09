import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from video_dataset import VideoDataset
from torch.utils.data import DataLoader
from model import SPLSS, LSQLocalization
import Visualizer
import argparse

# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
IMAGE_HEIGHT = 1200  # 1280 originally
IMAGE_WIDTH = 800  # 1918 originally

def main():
    parser = argparse.ArgumentParser(
                    prog = 'Finding Local Maxima Through Deep Learning',
                    description = '',
                    epilog = '')
    parser.add_argument("-p", "--path")
    parser.add_argument("-i", "--input")
    args = parser.parse_args()

    model = SPLSS(in_channels=1, out_channels=3, state_dict=torch.load("reinhardnetv1.pth.tar")).to(DEVICE)
    dataset = VideoDataset(args.path, args.input)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    vis = Visualizer.Visualize2D(batched=True, x=1, y=1)

    loc = LSQLocalization(window_size=5)
    loop = tqdm(dataloader)
    for images in loop:
        images = images.to(device=DEVICE)
        
        with torch.cuda.amp.autocast():
            segmentation = model(images).softmax(dim=1)
            sigma, mean, amplitude = loc.test(segmentation)

            #segmentation = segmentation.argmax(dim=1)
            #vis.draw_segmentation(segmentation, 3)
            #vis.draw_heatmap(segmentation, heatmapaxis=1)
            vis.draw_images(images)
            vis.draw_points(mean)
            vis.show()


if __name__ == "__main__":
    main()

    #batch = torch.randn((8, 200, 200))