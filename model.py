from gettext import npgettext
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
import kornia
import utils


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, encoder, out_channels=3, features=[64, 128, 256, 512]):
        super(Decoder, self).__init__()
        self.ups = nn.ModuleList()
        self.encoder = encoder
        self.out_channels=out_channels

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))


    def forward(self, x):
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = self.encoder.skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(Encoder, self).__init__()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.in_channels = in_channels
        
        #Downsampling
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

    def forward(self, x):
        self.skip_connections = []
        for down in self.downs:
            x = down(x)
            self.skip_connections.append(x)
            x = self.pool(x)

        self.skip_connections = self.skip_connections[::-1]

        return x


class LocalizationModule(nn.Module):
    def __init__(self, order=2, window_size=3, heatmapaxis=1, threshold=0.5):
        super(LSQLocalization, self).__init__()
        self.order = order
        self.window_size = window_size
        self.pad = window_size // 2
        self.heatmapaxis = heatmapaxis
        self.gauss_blur_sigma = 1.5
        self.threshold = threshold
        
        self.kernel = torch.ones(self.window_size, self.window_size).to(DEVICE)
        self.kernel[self.window_size//2, self.window_size//2] = 0

    # Output is of form BATCHSIZE X NUM_POINTS X 2
    # Where the last dimensions contains [x, y] coordinates in image_space
    def forward(self, x):
        heat = x[:, self.heatmapaxis, :, :].clone()
        heat = heat.unsqueeze(1)

        # Generate thresholded image
        threshed_heat = (heat > self.threshold) * heat

        # Use dilation filtering to generate local maxima and squeeze first dimension
        dilated_heat = kornia.morphology.dilation(threshed_heat, self.kernel)
        local_maxima = threshed_heat > dilated_heat
        local_maxima = local_maxima[:, 0, :, :]

        # Find local maxima and indices at which we need to split the data
        maxima_indices = local_maxima.nonzero()
        split_indices = get_split_indices(maxima_indices[:, 0]).tolist()

        # Extract windows around the local maxima
        intensities, y_windows, x_windows = utils.extractWindow(heat[:, 0, :, :], maxima_indices, self.window_size)

        # Reformat [-2, -1, 0, ..] tensors for x-y-indexing
        reformat_x = self.x_sub.repeat(x_windows.size(0), 1, 1).reshape(-1, self.window_size**2)
        reformat_y = self.y_sub.repeat(y_windows.size(0), 1, 1).reshape(-1, self.window_size**2)

        # Use Guos Weighted Gaussian Fitting algorithm based on the intensities of the non-thresholded image
        sigma, mu, amplitude = GuosBatchAnalytic(reformat_x, reformat_y, intensities.reshape(-1, self.window_size**2))

        # Add found mus to the initial quantized local maxima
        mu = maxima_indices[:, 1:] + mu[:, [1, 0]]

        # Split the tensors and return lists of sigma, mus, and the amplitudes per batch
        return torch.tensor_split(sigma, split_indices), torch.tensor_split(mu, split_indices), torch.tensor_split(amplitude, split_indices)

class LSQLocalization:
    def __init__(self, order=2, gauss_window=5, local_maxima_window=11, heatmapaxis=1, threshold=0.5):
        super(LSQLocalization, self).__init__()
        self.order = order

        self.local_maxima_window_size = local_maxima_window
        self.gauss_window_size = gauss_window
        self.pad = self.gauss_window_size // 2
        
        
        self.heatmapaxis = heatmapaxis
        self.gauss_blur_sigma = 1.5
        self.threshold = threshold
        
        self.kernel = torch.ones(self.local_maxima_window_size, self.local_maxima_window_size).to(DEVICE)
        self.kernel[self.local_maxima_window_size//2, self.local_maxima_window_size//2] = 0


        sub = torch.linspace(-self.gauss_window_size//2 + 1, self.gauss_window_size//2, self.gauss_window_size)
        x_sub, y_sub = torch.meshgrid(sub, sub, indexing="xy")
        self.x_sub = x_sub.unsqueeze(0).to(DEVICE)
        self.y_sub = y_sub.unsqueeze(0).to(DEVICE)

    def test(self, x, segmentation=None):
        heat = x[:, self.heatmapaxis, :, :].clone()
        heat = heat.unsqueeze(1)

        # Generate thresholded image
        threshed_heat = (heat > self.threshold) * heat

        # Use dilation filtering to generate local maxima and squeeze first dimension
        dilated_heat = kornia.morphology.dilation(threshed_heat, self.kernel)
        local_maxima = threshed_heat > dilated_heat
        local_maxima = local_maxima[:, 0, :, :]

        if segmentation is not None:
            local_maxima = local_maxima * segmentation

        # Find local maxima and indices at which we need to split the data
        maxima_indices = local_maxima.nonzero()
        split_indices = get_split_indices(maxima_indices[:, 0]).tolist()

        # Extract windows around the local maxima
        intensities, y_windows, x_windows = utils.extractWindow(heat[:, 0, :, :], maxima_indices, self.gauss_window_size)

        # Reformat [-2, -1, 0, ..] tensors for x-y-indexing
        reformat_x = self.x_sub.repeat(x_windows.size(0), 1, 1).reshape(-1, self.gauss_window_size**2)
        reformat_y = self.y_sub.repeat(y_windows.size(0), 1, 1).reshape(-1, self.gauss_window_size**2)

        # Use Guos Weighted Gaussian Fitting algorithm based on the intensities of the non-thresholded image
        sigma, mu, amplitude = GuosBatchAnalytic(reformat_x, reformat_y, intensities.reshape(-1, self.gauss_window_size**2))

        # Add found mus to the initial quantized local maxima
        mu = maxima_indices[:, 1:] + mu[:, [1, 0]]

        # Split the tensors and return lists of sigma, mus, and the amplitudes per batch
        return torch.tensor_split(sigma, split_indices), torch.tensor_split(mu, split_indices), torch.tensor_split(amplitude, split_indices)


def get_split_indices(nonzeros_indexing):
    return (nonzeros_indexing - torch.concat([torch.zeros(1).to(DEVICE), nonzeros_indexing])[0:-1]).nonzero().squeeze()

            
def GuosBatchAnalytic(x, y, z):
    weights = z**2

    A = torch.stack(utils.get_basis(x, y), dim=-1)
    A = (A.transpose(1, 2) * weights.unsqueeze(1)).transpose(1, 2)
    b = (torch.log(z) * weights)
    ATA = (A.transpose(1, 2) @ A).float()
    #c = (torch.linalg.lstsq(ATA, A.transpose(1, 2)).solution @ b.unsqueeze(-1)).squeeze()
    #c = (torch.linalg.pinv(ATA) @ A.transpose(1, 2) @ b.unsqueeze(-1)).squeeze()
    c = (ATA.inverse() @ A.transpose(1, 2) @ b.unsqueeze(-1)).squeeze()
    return poly_to_gauss(c[:, [2,5]], c[:, [1,3]], c[:, [0,4]])


def poly_to_gauss(A, B, C):
    sigma = torch.sqrt(-1 / (2.0 * A))
    mu = B * sigma**2
    height = torch.exp(C + 0.5 * mu**2 / sigma**2)
    return sigma, mu, height

        # Return list of length batch_size, containing the 2d sub-pixel locations of heatmaps




class SPLSS(nn.Module):
    def __init__(self, in_channels, out_channels, state_dict=None, features=[64, 128, 256, 512], num_samplepoints=50, point_dims=2, device="cuda"):
        super(SPLSS, self).__init__()
        self.bottleneck_size = features[-1]*2

        self.encoder = Encoder(in_channels, features)
        self.decoder = Decoder(self.encoder, out_channels, features)
        #self.pointLocalizer = LocalizationModule(self.bottleneck_size, point_dims=point_dims, num_samplepoints=num_samplepoints)
        self.bottleneck = DoubleConv(features[-1], self.bottleneck_size)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        if state_dict:
            self.load_from_dict(state_dict)

    def get_statedict(self):
        return {"Encoder": self.encoder.state_dict(),
                "Bottleneck": self.bottleneck.state_dict(),
                "Decoder": self.decoder.state_dict(),
                "LastConv": self.final_conv.state_dict()}

    def load_from_dict(self, dict):
        self.encoder.load_state_dict(dict["Encoder"])
        self.bottleneck.load_state_dict(dict["Bottleneck"])
        self.decoder.load_state_dict(dict["Decoder"])
        self.final_conv.load_state_dict(dict["LastConv"])

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        #points = self.pointLocalizer(x)
        x = self.decoder(x)

        return self.final_conv(x)#, points


def test():
    CHM_loss = Losses.torch_loss_cHM()

    x = torch.randn((4, 3, 512, 256))
    y = torch.randn((4, 2, 100))
    model = SPLSS(in_channels=3, out_channels=3)
    seg = model(x)

    print(seg.shape)
    #print(points.shape)

    #print(Losses.CountingLoss(points.reshape(4, 2, -1), y))
    #print(CHM_loss.apply(points.reshape(4, 2, -1), y))

    # Seems to be working

if __name__ == "__main__":
    test()