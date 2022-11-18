import torch


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


if __name__ == "__main__":
    a = [torch.randn(10, 2), torch.randn(10, 2)]
    b = [torch.randn(10, 2), torch.randn(5, 2)]

    print(nearestNeighborLoss(a, b))
    print(nearestNeighborLoss(a, b, t1=3.0))

    print(gaussianNearestNeighbors(a, b, t0=0.25, t1=2))
    print(gaussianNearestNeighbors(a, b, t0=1.0,  t1=5.0))