import torch
import albumentations as A
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime
from evaluate import evaluate
from printer import Printer
import os
import argparse
from models.UNet import Model
from models.LSQ import LSQLocalization
import mscoco
import score
import yaml
import math
import pickle
import torch
import torch.utils.data as data
import torch.distributed as dist
from torchvision import transforms
from torch.utils.data.sampler import Sampler, BatchSampler



import sys
sys.path.append("models/")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser(
                    prog = 'Keypoint Regularized Training for Semantic Segmentation',
                    description = 'Train a Segmentation Network that is optimized for simultaneously outputting keypoints',
                    epilog = 'Arguments can be used to overwrite values in a config file.')
    parser.add_argument("--epochs", type=int, default=30)
    #parser.add_argument("--learning_rate", type=float, )
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--features", type=int, nargs="+")
    parser.add_argument("--dataset_path", type=str)
    args = parser.parse_args()
    
    CHECKPOINT_PATH = "pretrained/"
    config = vars(args)
    config['in_channels'] = 3
    config['out_channels'] = 21

    model = Model(config=config).to(DEVICE)

    input_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225]),])
    dataset_train = mscoco.COCOSegmentation(args.dataset_path, split="train", transform=input_transform)
    dataset_val = mscoco.COCOSegmentation(args.dataset_path, split="val", transform=input_transform)
    
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=1, pin_memory=True, shuffle=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.004, momentum=0.9, weight_decay=0.0005)
    loss = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        #evaluate(dataloader_val, model)
        #train(dataloader_train, loss, model, optimizer, epoch)

        checkpoint = {"optimizer": optimizer.state_dict()} | model.get_statedict()
        torch.save(checkpoint, CHECKPOINT_PATH + str(args.features) + ".pth.tar")

        with open(CHECKPOINT_PATH + str(args.features) + ".yml", 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

    Printer.OKG("Training Done!")

def train(train_loader, loss_func, model, optimizer, epoch):
    Printer.Header("EPOCH: {0}".format(epoch))
    model.train()
    running_average = 0.0
    loop = tqdm(train_loader, desc="TRAINING")
    for images, target in loop:
        images = images.to(DEVICE)
        target = target.to(DEVICE)
        optimizer.zero_grad()

        images = images.to(device=DEVICE)

        # forward
        pred_seg = model(images)
        loss = loss_func(pred_seg.float(), target.long())

        loss.backward()

        running_average += loss.item()
        loop.set_postfix(loss=loss.item())
        optimizer.step()


def evaluate(dataloader, model):
    model.eval()
    metric = score.SegmentationMetric(21)
    Printer.Header("EVALUATION")
    count = 0
    for image, target in tqdm(dataloader):
        image = image.to(DEVICE)
        target = target.to(DEVICE)

        with torch.no_grad():
            outputs = model(image)

        metric.update(outputs[0], target)
        pixAcc, mIoU = metric.get()
        count += 1
    print("Acc: {:.3f}, mIoU: {:.3f}".format(pixAcc * 100, mIoU * 100))

if __name__ == "__main__":
    main()