import torch
import kornia

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


# Indices: Tensor of size Nx3, like [[batch, y, x], ..]
# Batch: Image batch of size BATCH x X x Y
# Returns: Tensor of Size N x 3 x 3
def extractWindow(batch, indices, window_size=11, device='cuda'):
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

    y_windows += y_sub.unsqueeze(0).long().to(device)
    x_windows += x_sub.unsqueeze(0).long().to(device)

    # Catching windows
    windows = batch[
        batch_index.unsqueeze(-1), 
        y_windows.reshape(-1, window_size*window_size), 
        x_windows.reshape(-1, window_size*window_size)]

    return windows.reshape(-1, window_size, window_size), y_windows, x_windows


# From: https://scipython.com/blog/linear-least-squares-fitting-of-a-two-dimensional-data/
# There's definitely a faster way to do this
def get_basis(x, y):
    """Return the fit basis polynomials: 1, x, x^2, ..., xy, x^2y, ... etc."""
    basis = []
    for i in range(3):
        for j in range(3 - i):
            basis.append(x**j * y**i)
    return basis


def get_split_indices(nonzeros_indexing, batch_size, device='cuda'):
    index, count = torch.unique(nonzeros_indexing, return_counts=True)
    splits = torch.zeros((batch_size,), dtype=count.dtype, device=device)
    splits[index] = count
    return torch.cumsum(splits, 0)[:-1]

            
def GuosBatchAnalytic(x, y, z):
    weights = z**2

    A = torch.stack(get_basis(x, y), dim=-1)
    A = (A.transpose(1, 2) * weights.unsqueeze(1)).transpose(1, 2)
    b = (torch.log(z) * weights)
    ATA = (A.transpose(1, 2) @ A).float()
    c = (torch.linalg.lstsq(ATA, A.transpose(1, 2)).solution @ b.unsqueeze(-1)).squeeze()
    #c = (torch.linalg.pinv(ATA) @ A.transpose(1, 2) @ b.unsqueeze(-1)).squeeze()
    #c = (ATA.inverse() @ A.transpose(1, 2) @ b.unsqueeze(-1)).squeeze()

    if len(c.shape) == 1:
        c = c.unsqueeze(0)

    return poly_to_gauss(c[:, [2,5]], c[:, [1,3]], c[:, [0,4]])


def poly_to_gauss(A, B, C):
    sigma = torch.sqrt(-1 / (2.0 * A))
    mu = B * sigma**2
    height = torch.exp(C + 0.5 * mu**2 / sigma**2)
    return sigma, mu, height


class LSQLocalization:
    def __init__(self, order=2, gauss_window=5, local_maxima_window=11, heatmapaxis=1, threshold=0.3, device='cuda'):
        super(LSQLocalization, self).__init__()
        self.order = order

        self.local_maxima_window_size = local_maxima_window
        self.gauss_window_size = gauss_window
        self.pad = self.gauss_window_size // 2
        
        
        self.heatmapaxis = heatmapaxis
        self.gauss_blur_sigma = 1.5
        self.threshold = threshold
        
        self.kernel = torch.ones(self.local_maxima_window_size, self.local_maxima_window_size).to(device)
        self.kernel[self.local_maxima_window_size//2, self.local_maxima_window_size//2] = 0


        sub = torch.linspace(-self.gauss_window_size//2 + 1, self.gauss_window_size//2, self.gauss_window_size)
        x_sub, y_sub = torch.meshgrid(sub, sub, indexing="xy")
        self.x_sub = x_sub.unsqueeze(0).to(device)
        self.y_sub = y_sub.unsqueeze(0).to(device)

        self.device = device

    def estimate(self, x, segmentation=None):
        heat = x[:, self.heatmapaxis, :, :].clone()
        heat = heat.unsqueeze(1)

        # Generate thresholded image
        threshed_heat = (heat > self.threshold) * heat
        threshed_heat = kornia.filters.gaussian_blur2d(threshed_heat, self.kernel.shape, [self.kernel.shape[0]/4, self.kernel.shape[0]/4])
        

        # Use dilation filtering to generate local maxima and squeeze first dimension
        dilated_heat = kornia.morphology.dilation(threshed_heat, self.kernel)
        local_maxima = threshed_heat > dilated_heat
        local_maxima = local_maxima[:, 0, :, :]

        if segmentation is not None:
            local_maxima = local_maxima * segmentation

        # Find local maxima and indices at which we need to split the data
        maxima_indices = local_maxima.nonzero()

        if len(maxima_indices) == 0:
            return None, None, None

        split_indices = get_split_indices(maxima_indices[:, 0], x.size(0), device=self.device).tolist()

        split_indices = [split_indices] if type(split_indices) == int else split_indices

        # Extract windows around the local maxima
        intensities, y_windows, x_windows = extractWindow(heat[:, 0, :, :], maxima_indices, self.gauss_window_size, device=self.device)

        # Reformat [-2, -1, 0, ..] tensors for x-y-indexing
        reformat_x = self.x_sub.repeat(x_windows.size(0), 1, 1).reshape(-1, self.gauss_window_size**2)
        reformat_y = self.y_sub.repeat(y_windows.size(0), 1, 1).reshape(-1, self.gauss_window_size**2)

        # Use Guos Weighted Gaussian Fitting algorithm based on the intensities of the non-thresholded image
        sigma, mu, amplitude = GuosBatchAnalytic(reformat_x, reformat_y, intensities.reshape(-1, self.gauss_window_size**2))

        # Add found mus to the initial quantized local maxima
        mu = maxima_indices[:, 1:] + mu[:, [1, 0]]

        # Split the tensors and return lists of sigma, mus, and the amplitudes per batch
        return torch.tensor_split(sigma, split_indices), torch.tensor_split(mu, split_indices), torch.tensor_split(amplitude, split_indices)




if __name__ == "__main__":
    device = 'cpu'
    loc = LSQLocalization(heatmapaxis=1, device=device)
    val = torch.rand(4, 2, 128, 128)

    val *= 0.0

    _, mean, _ = loc.estimate(val)
    a = 1