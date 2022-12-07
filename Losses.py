import torch
import sys
sys.path.append("ChamferDistancePytorch")
import chamfer2D.dist_chamfer_2D
import chamfer_python

def findNearestNeighbour(point, target_points) -> int:
        if len(point.shape) == 1:
            point = point.unsqueeze(0)
        dist = torch.linalg.norm(point - target_points, dim=1)
        return torch.argmin(dist), dist.min()


def chamfer(prediction, gt):
    loss = 0.0

    chamLoss = chamfer2D.dist_chamfer_2D.chamfer_2DDist()

    for i, pred in enumerate(prediction):
        batch_pred = pred[~torch.isnan(pred).any(axis=1)]
        batch_gt = gt[i][~torch.isnan(gt[i]).any(axis=1)]

        if batch_pred.nelement() == 0:
            continue

        if batch_gt.nelement() == 0:
            continue

        d1, d2, idx1, idx2 = chamLoss(batch_pred.unsqueeze(0), batch_gt[:, [1, 0]].unsqueeze(0))
        #loss += torch.mean(d1) + torch.mean(d2)
        loss += (torch.mean(d1) + torch.mean(d2)) / (2*(d1.shape[1] + d1.shape[1]))

    return loss


def nnLoss(pred, gt, threshold):
    batched_distance = 0 
    neighbours = 0

    for batch in range(len(pred)):
        batch_pred = pred[batch]
        batch_pred = batch_pred[~torch.isnan(batch_pred).any(axis=1)]

        batch_gt = gt[batch]
        batch_gt = batch_gt[~torch.isnan(batch_gt).any(axis=1)]
        
        if len(batch_pred) == 0:
            continue

        for i in range(batch_gt.shape[0]):
            _, distance = findNearestNeighbour(batch_gt[i], batch_pred)
            
            if distance < threshold:
                batched_distance += distance
                neighbours += 1

    #print(neighbours)
    return torch.tensor([batched_distance / neighbours], device=gt.device) if neighbours != 0 else torch.tensor([0], device=gt.device)


def nearestNeighborLoss(pointsA, pointsB, t0 = None, t1 = None):
    nearest_neighbor_loss = 0.0
    numel = 0

    for perFrameA, perFrameB in zip(pointsA, pointsB):
        distances, indices = knn(perFrameA, perFrameB, 1)
        distances = distances[0]

        if t0 is not None:
            distances *= ~(distances < t0)

        if t1 is not None:
            distances *= distances < t1

        nearest_neighbor_loss += distances.sum()
        numel += distances.numel()

    return nearest_neighbor_loss / numel

def gaussian(x, mean, sigma):
    return torch.exp((x - mean)*(x-mean) / sigma*sigma)

def gaussianNearestNeighbors(pointsA, pointsB, t0 = 0.5, t1 = 3.0):
    mean = t0 + ((t1 - t0) / 2)
    sigma = (t1 - t0)/4

    nearest_neighbor_loss = 0.0
    numel = 0

    for perFrameA, perFrameB in zip(pointsA, pointsB):
        distances, indices = knn(perFrameA, perFrameB, 1)
        distances = distances[0]

        # Maybe add gaussian here. Should test this
        distances = gaussian(distances, mean, sigma)

        nearest_neighbor_loss += distances.sum()
        numel += distances.numel()

    return nearest_neighbor_loss / numel

def knn(ref, query, k):
    dist = ref.unsqueeze(0) - query.repeat_interleave(ref.shape[0], dim=0).reshape(query.shape[0], -1, 2)
    dist = torch.linalg.norm(dist, dim=2)
    return torch.topk(dist, dim=0, k=1, largest=False)

import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


def to_one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels


class PolyLoss(_Loss):
    def __init__(self,
                 softmax: bool = True,
                 ce_weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean',
                 epsilon: float = 1.0,
                 ) -> None:
        super().__init__()
        self.softmax = softmax
        self.reduction = reduction
        self.epsilon = epsilon
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction='none')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
                You can pass logits or probabilities as input, if pass logit, must set softmax=True
            target: if target is in one-hot format, its shape should be BNH[WD],
                if it is not one-hot encoded, it should has shape B1H[WD] or BH[WD], where N is the number of classes, 
                It should contain binary values
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
       """
        if len(input.shape) - len(target.shape) == 1:
            target = target.unsqueeze(1).long()
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        # target not in one-hot encode format, has shape B1H[WD]
        if n_pred_ch != n_target_ch:
            # squeeze out the channel dimension of size 1 to calculate ce loss
            self.ce_loss = self.cross_entropy(input, torch.squeeze(target, dim=1).long())
            # convert into one-hot format to calculate ce loss
            target = to_one_hot(target, num_classes=n_pred_ch)
        else:
            # # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
            self.ce_loss = self.cross_entropy(input, torch.argmax(target, dim=1))

        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        pt = (input * target).sum(dim=1)  # BH[WD]
        poly_loss = self.ce_loss + self.epsilon * (1 - pt)

        if self.reduction == 'mean':
            polyl = torch.mean(poly_loss)  # the batch and channel average
        elif self.reduction == 'sum':
            polyl = torch.sum(poly_loss)  # sum over the batch and channel dims
        elif self.reduction == 'none':
            # BH[WD] 
            polyl = poly_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return (polyl)


if __name__ == "__main__":
    loss = PolyLoss(softmax=True)
    B, C, H, W = 2, 5, 3, 3
    input = torch.rand(B, C, H, W, requires_grad=True)
    target = torch.randint(low=0, high=C - 1, size=(B, 1, H, W)).long()
    output1 = loss(input, target)
    print(output1)
    output1 = loss(input, target.squeeze())
    print(output1)