import os
from PyQt5.QtCore import QLibraryInfo

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)

import torch
import albumentations as A
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import Utils.LRscheduler as LRscheduler
import datetime
import yaml
import Utils.printer as Printer
import os
import pygit2
import Utils.ConfigArgsParser as ConfigArgsParser
from models.LSQ import LSQLocalization
from typing import Tuple
import sys
import random
import Utils.Args as Args
import Utils.utils as utils
import wandb

sys.path.append("models/")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def main():
    parser = Args.GlobalArgumentParser(
        prog="Keypoint Regularized Training for Semantic Segmentation",
        description="Train a Segmentation Network that is optimized for simultaneously outputting keypoints",
        epilog="Arguments can be used to overwrite values in a config file.",
    )

    args = parser.parse_args()

    LOG_WANDB = args.logwandb
    LOAD_FROM_CHECKPOINT = args.checkpoint is not None

    if args.checkpoint_name:
        CHECKPOINT_PATH = os.path.join("checkpoints", args.checkpoint_name)
    else:
        CHECKPOINT_PATH = (
            args.checkpoint
            if LOAD_FROM_CHECKPOINT
            else os.path.join(
                "checkpoints", datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            )
        )

    CHECKPOINT_NAME = CHECKPOINT_PATH.split("/")[-1]

    # Always add magic number to path ._.
    if not LOAD_FROM_CHECKPOINT:
        magic_number = str(random.randint(0, 10000))
        CHECKPOINT_NAME += "_" + magic_number
        CHECKPOINT_PATH += "_" + magic_number

    CONFIG_PATH = (
        os.path.join(CHECKPOINT_PATH, "config.yml")
        if LOAD_FROM_CHECKPOINT
        else args.config
    )
    TRAIN_TRANSFORM_PATH = (
        os.path.join(CHECKPOINT_PATH, "train_transform.yaml")
        if LOAD_FROM_CHECKPOINT
        else "train_transform_sequence.yaml"
    )
    VAL_TRANSFORM_PATH = (
        os.path.join(CHECKPOINT_PATH, "val_transform.yaml")
        if LOAD_FROM_CHECKPOINT
        else "val_transform_sequence.yaml"
    )

    config = ConfigArgsParser.ConfigArgsParser(utils.load_config(CONFIG_PATH), args)

    # Gotta check this manually as sweeps do not allow nested lists
    if args.model_depth is not None:
        if args.model_depth == 0:
            config["features"] = [64, 128, 256]
        elif args.model_depth == 1:
            config["features"] = [64, 128, 256, 512]
        elif args.model_depth == 2:
            config["features"] = [64, 128, 256, 512, 1024]
        elif args.model_depth == 3:
            config["features"] = [32, 64, 128, 256]
        elif args.model_depth == 4:
            config["features"] = [32, 64, 128, 256, 512]
        elif args.model_depth == 5:
            config["features"] = [32, 64, 128, 256, 512, 1024]

    if not LOAD_FROM_CHECKPOINT:
        os.mkdir(CHECKPOINT_PATH)

    train_transform = A.load(TRAIN_TRANSFORM_PATH, data_format="yaml")
    val_transforms = A.load(VAL_TRANSFORM_PATH, data_format="yaml")

    neuralNet = __import__(config["model"])
    model = neuralNet.Model(
        config=config,
        state_dict=(
            torch.load(os.path.join("pretrained", str(config["features"]) + ".pth.tar"))
            if args.pretrain
            else None
        ),
        pretrain=True,
    ).to(DEVICE)
    loss = nn.CrossEntropyLoss(
        weight=torch.tensor(config["loss_weights"], dtype=torch.float32, device=DEVICE)
    )
    # loss = kornia.losses.dice_loss
    cpu_loss = nn.CrossEntropyLoss(
        weight=torch.tensor(config["loss_weights"], dtype=torch.float32, device="cpu")
    )

    if LOG_WANDB:
        repo = pygit2.Repository(".")
        num_uncommitted_files = repo.diff().stats.files_changed

        if num_uncommitted_files > 0:
            Printer.Warning("Uncommited changes! Please commit before training.")
            exit()

        wandb.init(project="SSSLSquared", config=config)
        wandb.config["loss"] = type(loss).__name__
        wandb.config["checkpoint_name"] = CHECKPOINT_NAME
        wandb.config["train_transform"] = A.to_dict(train_transform)
        wandb.config["validation_transform"] = A.to_dict(val_transforms)

    config.printDifferences(utils.load_config(CONFIG_PATH))

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = (
        optim.Adam(parameters, lr=config["learning_rate"])
        if config["optimizer"] == "adam"
        else optim.SGD(parameters, lr=config["learning_rate"])
    )

    scheduler = LRscheduler.PolynomialLR(
        optimizer, config["num_epochs"], last_epoch=config["last_epoch"]
    )

    dataset = __import__("dataset").__dict__[config["dataset_name"]]
    train_ds = dataset(config=config, is_train=True, transform=train_transform)
    val_ds = dataset(config=config, is_train=False, transform=val_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=True,
        shuffle=False,
    )

    localizer = LSQLocalization(
        local_maxima_window=config["maxima_window"],
        gauss_window=config["gauss_window"],
        heatmapaxis=config["heatmapaxis"],
        threshold=config["threshold"],
    )

    if LOG_WANDB:
        wandb.watch(model)

    # Save config stuff
    A.save(
        train_transform,
        CHECKPOINT_PATH + "/train_transform_sequence.yaml",
        data_format="yaml",
    )
    A.save(
        val_transforms,
        CHECKPOINT_PATH + "/val_transform_sequence.yaml",
        data_format="yaml",
    )

    for epoch in range(config["last_epoch"], config["num_epochs"]):
        # Train the network
        train(
            train_loader,
            loss,
            model,
            scheduler,
            epoch,
            localizer,
            use_regression=epoch > config["keypoint_regularization_at"] - 1,
            keypoint_lambda=config["keypoint_lambda"],
            log_wandb=False,
        )

        # Evaluate on Validation Set
        evaluate(
            val_loader,
            model,
            cpu_loss,
            localizer=(
                localizer if epoch > config["keypoint_regularization_at"] else None
            ),
            epoch=epoch,
            log_wandb=LOG_WANDB,
        )

        # Visualize Validation as well as Training Set examples
        visualize(
            val_loader, model, epoch, title="Val Predictions", log_wandb=LOG_WANDB
        )
        visualize(
            train_loader, model, epoch, title="Train Predictions", log_wandb=LOG_WANDB
        )

        checkpoint = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        } | model.get_statedict()
        torch.save(checkpoint, CHECKPOINT_PATH + "/model.pth.tar")

        config["last_epoch"] = epoch
        with open(CHECKPOINT_PATH + "/config.yml", "w") as outfile:
            yaml.dump(dict(config), outfile, default_flow_style=False)

    Printer.OKG("Training Done!")


def train(
    train_loader,
    loss_func,
    model,
    scheduler,
    epoch,
    localizer,
    use_regression=False,
    keypoint_lambda=0.1,
    log_wandb=False,
):
    Printer.Header("EPOCH: {0}".format(epoch))
    model.train()
    running_average = 0.0
    loop = tqdm(train_loader, desc="TRAINING")
    show = True
    for images, gt_seg, gt_keypoints in loop:
        scheduler.zero_grad()

        images = images.to(device=DEVICE)
        gt_seg = gt_seg.to(device=DEVICE)
        gt_keypoints = gt_keypoints.to(device=DEVICE)

        # forward
        pred_seg = model(images)

        loss = loss_func(pred_seg.float(), gt_seg.long())

        segmentation = pred_seg.softmax(dim=1)
        segmentation_argmax = segmentation.argmax(dim=1)

        if use_regression:
            try:
                _, pred_keypoints, _ = localizer.estimate(
                    segmentation,
                    torch.bitwise_or(
                        segmentation_argmax == 2, segmentation_argmax == 3
                    ),
                )
            except:
                Printer.Warning("Matrix singular.")
                continue

        loss.backward()
        scheduler.step()

        running_average += loss.item()
        loop.set_postfix(loss=loss.item())
    scheduler.update_lr()

    if log_wandb:
        print("Logging wandb")
        wandb.log({"Loss": running_average / len(train_loader)}, step=epoch)


def visualize(
    val_loader, model, epoch, title="Validation Predictions", num_log=1, log_wandb=False
):
    if not log_wandb:
        return

    model.eval()
    for images, gt_seg, _ in val_loader:
        images = images.to(device=DEVICE)
        gt_seg = gt_seg.to(device=DEVICE)

        pred_seg = model(images)
        images = utils.normalize_image_batch(images)

        for i in range(num_log):
            pred_temp = pred_seg[0, :, i].softmax(dim=0).argmax(dim=0)
            wandb.log(
                {
                    "{0} {1}".format(title, i): wandb.Image(
                        images[i, 0].detach().cpu().numpy(),
                        masks={
                            "predictions": {
                                "mask_data": pred_temp.detach().cpu().numpy(),
                                "class_labels": {
                                    0: "Background",
                                    1: "Glottis",
                                    2: "Vocalfold",
                                    3: "Laserpoints",
                                },
                            },
                            "ground_truth": {
                                "mask_data": gt_seg[0, i].detach().cpu().numpy(),
                                "class_labels": {
                                    0: "Background",
                                    1: "Glottis",
                                    2: "Vocalfold",
                                    3: "Laserpoints",
                                },
                            },
                        },
                    )
                },
                step=epoch,
            )
        return


import Metrics.KeypointMetrics as KeypointMetrics
from chamferdist import ChamferDistance
from torchmetrics.functional import dice, jaccard_index


def evaluate(
    val_loader, model, loss_func, localizer=None, epoch=-1, log_wandb=False
) -> Tuple[float, float, float, float, float, float, float]:
    running_average = 0.0
    num_images = 0
    count = 0

    model.eval()

    dice_val = 0.0
    iou = 0.0
    cham = 0.0
    f1 = 0.0
    TP = 0
    FP = 0
    FN = 0

    chamloss = ChamferDistance()

    l2_distances = []
    nme = 0.0
    precision = 0.0
    inference_time = 0
    loop = tqdm(val_loader, desc="EVAL")
    for images, gt_seg, keypoints in loop:
        count += 1

        images = images.to(device=DEVICE)
        gt_seg = gt_seg.long()

        keypoints = keypoints.float()
        pred_seg = model(images)

        softmax = pred_seg.softmax(dim=1).detach().cpu()
        argmax = softmax.argmax(dim=1)
        num_images += images.shape[0]

        dice_val += dice(argmax, gt_seg, num_classes=4)
        iou += jaccard_index(argmax, gt_seg, num_classes=4)

        loss = loss_func.cpu()(pred_seg.detach().cpu(), gt_seg).item()
        running_average += loss

        if localizer is not None:
            segmentation = pred_seg.softmax(dim=1)
            for i in range(segmentation.shape[0]):
                try:
                    _, pred_keypoints, _ = localizer.estimate(segmentation[i])
                except:
                    print("Matrix probably singular. Whoopsie.")
                    continue

                if pred_keypoints is None:
                    continue

                gt_keypoints = keypoints[i]
                gt_keypoints = keypoints.split(1, dim=0)
                gt_keypoints = [
                    keys[0][~torch.isnan(keys[0]).any(axis=1)][:, [1, 0]]
                    for keys in gt_keypoints
                ]

                TP_temp, FP_temp, FN_temp, distances = (
                    KeypointMetrics.keypoint_statistics(
                        pred_keypoints,
                        gt_keypoints,
                        2.0,
                        prediction_format="yx",
                        target_format="yx",
                    )
                )
                TP += TP_temp
                FP += FP_temp
                FN += FN_temp
                l2_distances = l2_distances + distances

                for j in range(len(pred_keypoints)):
                    cham += chamloss(
                        gt_keypoints[j].unsqueeze(0),
                        pred_keypoints[j].unsqueeze(0).detach().cpu(),
                        bidirectional=True,
                    )

        if localizer is not None:
            loop.set_postfix({"DICE": dice_val, "Loss": loss, "IOU": iou})
        else:
            loop.set_postfix({"DICE": dice_val, "Loss": loss, "IOU": iou})

        count += 1

    # Segmentation
    total_dice = dice_val / num_images
    total_IOU = iou / num_images
    total_CHAM = cham / num_images
    eval_loss = running_average / len(val_loader)

    if localizer is not None:
        # Keypoint Stuff
        try:
            precision = KeypointMetrics.precision(TP, FP, FN)
            f1 = KeypointMetrics.f1_score(TP, FP, FN)
            nme = sum(l2_distances) / len(l2_distances)
        except:
            precision = 0.0
            ap = 0.0
            f1 = 0.0
            nme = 0.0

    if log_wandb:
        wandb.log({"Eval Loss": eval_loss}, step=epoch)
        wandb.log({"Eval DICE": total_dice}, step=epoch)
        wandb.log({"Eval IOU": total_IOU}, step=epoch)
        wandb.log({"Inference Time (ms)": inference_time / num_images}, step=epoch)

    print("_______SEGMENTATION STUFF_______")
    print("Eval Loss: {1}".format(epoch, eval_loss))
    print("Eval IOU: {1}".format(epoch, total_IOU))
    print("Eval DICE {0}: {1}".format(epoch, total_dice))

    if localizer is not None:
        print("_______KEYPOINT STUFF_______")
        print("Precision: {0}".format(float(precision)))
        print("F1: {0}".format(float(f1)))
        print("NME: {0}".format(float(nme)))
        print("ChamferDistance: {0}".format(float(total_CHAM)))

    return (
        float(precision),
        float(f1),
        float(nme),
        float(total_IOU),
        float(total_dice),
        float(total_CHAM),
    )


if __name__ == "__main__":
    main()
