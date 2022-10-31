import torch
import torchvision
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x).argmax(axis=1)
            #print(preds.size())
            #preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
    #        dice_score += (2 * (preds * y).sum()) / (
    #            (preds + y).sum() + 1e-8
    #        )

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    #print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):

    try:
        os.mkdir(folder)
    except:
        pass

    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device)

        with torch.no_grad():
            preds = model(x).argmax(axis=1)#torch.sigmoid(model(x))
            #preds = (preds > 0.5).float()

        class_colors = [torch.tensor([0, 0, 0], device=device), torch.tensor([0, 255, 0], device=device), torch.tensor([0, 0, 255], device=device)]
        colored = class_to_color(preds, class_colors, device=device)
        colored_gt = class_to_color(y, class_colors)
        torchvision.utils.save_image(colored, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(colored_gt, f"{folder}{idx}.png")

    model.train()


def class_to_color(prediction, class_colors, device='cuda'):
    prediction = prediction.unsqueeze(0)
    output = torch.zeros(prediction.shape[0], 3, prediction.size(-2), prediction.size(-1), dtype=torch.float, device=device)
    for class_idx, color in enumerate(class_colors):
        mask = class_idx == torch.max(prediction, dim=1)[0]
        curr_color = color.reshape(1, 3, 1, 1)
        segment = mask*curr_color # should have shape 1, 3, 100, 100
        output += segment

    return output


def class_to_color_np(prediction, class_colors):
    prediction = np.expand_dims(prediction, 0)
    output = np.zeros(prediction.shape[0], 3, prediction.size(-2), prediction.size(-1))
    for class_idx, color in enumerate(class_colors):
        mask = class_idx == torch.max(prediction, dim=1)[0]
        curr_color = color.reshape(1, 3, 1, 1)
        segment = mask*curr_color # should have shape 1, 3, 100, 100
        output += segment

    return output

def draw_points(gt_image, gt_points, pred_points):
    blank = gt_image.copy()*255


    concat = None
    for i in range(blank.shape[0]):
        im = cv2.cvtColor(blank[i], cv2.COLOR_GRAY2BGR)
        im = draw_per_batch(im, gt_points, color=(255, 0, 0))
        im = draw_per_batch(im, pred_points[i], color=(0, 255, 0))


        if i == 0:
            concat = im
        else:
            concat = cv2.vconcat([concat, im])


    plt.imshow(concat, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def draw_segmentation(predictions, axis=1, x=2, y=4):
    seg = predictions.argmax(dim=1).unsqueeze(0)
    colors = [torch.tensor([0, 0, 0], device=DEVICE), torch.tensor([0, 255, 0], device=DEVICE), torch.tensor([255, 255, 255], device=DEVICE)]
    seg = class_to_color(seg, colors)

    fig = plt.figure()
    for i in range(seg.shape[0]):
        ax = fig.add_subplot(x, y, i + 1, projection='rectilinear')
        heat = seg[i, :, :, :].clone().detach().cpu().numpy()
        #heat /= heat.max()
        ax.imshow(np.moveaxis(heat, 0, -1), cmap = 'gray', interpolation = 'bicubic')
        ax.set_xticks([])
        ax.set_yticks([])  # to hide tick values on X and Y axis
    plt.show()


def draw_images(images, x=2, y=4):
    fig = plt.figure()
    for i in range(images.shape[0]):
        ax = fig.add_subplot(x, y, i + 1, projection='rectilinear')
        heat = images[i, 0, :, :].clone().detach().cpu().numpy()
        #heat /= heat.max()
        ax.imshow(heat, cmap = 'gray', interpolation = 'bicubic')
        ax.set_xticks([])
        ax.set_yticks([])  # to hide tick values on X and Y axis
    plt.show()


def draw_heatmap(prediction, axis=0, x=2, y=4):
    fig = plt.figure()
    for i in range(prediction.shape[0]):
        ax = fig.add_subplot(x, y, i + 1, projection='rectilinear')
        heat = prediction[i, axis, :, :].clone().detach().cpu().numpy()
        #heat /= heat.max()
        ax.imshow(heat, cmap = 'plasma', interpolation = 'bicubic')
        ax.set_xticks([])
        ax.set_yticks([])  # to hide tick values on X and Y axis
    plt.show()


def draw_per_batch(im, points, color=(255, 255, 255)):
    points = points.astype(np.int32)
    points = points[:, [1, 0]]
    in_bounds = np.bitwise_and(np.bitwise_and(points[:, 0] > 0, points[:, 1] > 0), np.bitwise_and(points[:, 0] < im.shape[0], points[:, 1] < im.shape[1]))
    points = points[in_bounds, :]
    points = points[:, [1, 0]]

    for j in range(points.shape[0]):
        cv2.circle(im, points[j], radius=2, color=color, thickness=-1)
    
    return im




# Indices: Tensor of size Nx3, like [[batch, y, x], ..]
# Batch: Image batch of size BATCH x X x Y
# Returns: Tensor of Size N x 3 x 3
def extractWindow(batch, indices, window_size=11):
    # Clean Windows, such that no image boundaries are hit

    batch_index = indices[:, 0]
    y = indices[:, 1]
    x = indices[:, 2]

    y = windows_out_of_bounds(y, batch.shape[1], window_size//2)
    x = windows_out_of_bounds(x, batch.shape[2], window_size//2)

    y_windows = y.unsqueeze(1).unsqueeze(1).repeat(1, window_size, window_size)
    x_windows = x.unsqueeze(1).unsqueeze(1).repeat(1, window_size, window_size)

    sub = torch.linspace(-window_size//2 + 1, window_size//2, window_size)
    x_sub, y_sub = torch.meshgrid(sub, sub, indexing="xy")

    y_windows += y_sub.unsqueeze(0).long().to(DEVICE)
    x_windows += x_sub.unsqueeze(0).long().to(DEVICE)

    # Catching windows
    windows = batch[
        batch_index.unsqueeze(-1), 
        y_windows.reshape(-1, window_size*window_size), 
        x_windows.reshape(-1, window_size*window_size)]

    return windows.reshape(-1, window_size, window_size), y_windows, x_windows

# Indices [n] Tensor
# Image size integer
# pad is window_size//2
def windows_out_of_bounds(indices, image_size, pad):
    # Move big ol indices
    indices = torch.where(indices + pad >= image_size, 
                    indices + ((image_size - pad) - indices) - 1, 
                    indices)

    indices = torch.where(indices - pad < 0,
                    indices + (pad - indices),
                    indices)
    
    return indices


def get_basis(x, y):
    """Return the fit basis polynomials: 1, x, x^2, ..., xy, x^2y, ... etc."""
    basis = []
    for i in range(3):
        for j in range(3 - i):
            basis.append(x**j * y**i)
    return basis