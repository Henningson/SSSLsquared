import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

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
    def __init__(self, encoder, out_channels, features):
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
    def __init__(self, in_channels, features):
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


class Model(nn.Module):
    def __init__(self, config={'in_channels': 6, 'out_channels': 4, 'features': [64, 128, 256, 512], 'batch_size': 4}, pretrain=False, state_dict=None, device="cuda"):
        super(Model, self).__init__()
        try:
            in_channels = config['in_channels']
        except:
            in_channels = 3

        try: 
            out_channels = config['out_channels']
        except:
            out_channels = 4
            
        self.sequence_length = config['sequence_length']
        features = config['features']
        batch_size = config['batch_size']

        self.bottleneck_size = features[-1]*2

        self.encoder = Encoder(in_channels, features)
        self.bottleneck = DoubleConv(features[-1], self.bottleneck_size)
        self.decoder = Decoder(self.encoder, out_channels, features)
        self.conv_to_channel = DoubleConv(features[0], self.sequence_length)


        self.bottleneck = DoubleConv(features[-1], self.bottleneck_size)

        self.depth_conv = nn.Sequential(nn.Conv3d(1, 1, kernel_size=3, padding=1, stride=1),
                            nn.ReLU())
        self.final_conv = nn.Sequential(nn.Conv3d(1, out_channels, kernel_size=3, padding=1, stride=1),
                                        nn.ReLU(),
                                        nn.Conv3d(out_channels, out_channels, kernel_size=1))#, padding=1, stride=1))

        if state_dict:
            self.load_from_dict(state_dict)
        
        if pretrain:
            self.encoder.requires_grad_ = False

        self.features = features

    def get_statedict(self):
        return {"Encoder": self.encoder.state_dict(),
                "Bottleneck": self.bottleneck.state_dict(),
                "DepthConv": self.depth_conv.state_dict(),
                "ConvToChannel": self.conv_to_channel.state_dict(),
                "Decoder": self.decoder.state_dict(),
                "LastConv": self.final_conv.state_dict()}

    def load_from_dict(self, dict):
        self.encoder.load_state_dict(dict["Encoder"])
        self.bottleneck.load_state_dict(dict["Bottleneck"])
        self.decoder.load_state_dict(dict["Decoder"])
        self.conv_to_channel.load_state_dict(dict["ConvToChannel"])
        try:
            self.final_conv.load_state_dict(dict["LastConv"])
        except:
            print("Final conv not initialized.")

        try:
            self.depth_conv.load_state_dict(dict["DepthConv"])
        except:
            print("3D Depth Convolution not initialized.")

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.depth_conv(x.unsqueeze(1)).squeeze()
        x = self.decoder(x)
        x = self.conv_to_channel(x)
        x = self.final_conv(x.unsqueeze(1))
        return x


def test():

    batch_size = 4
    sequence_length = 6
    config={'batch_size': batch_size, 'in_channels': 6, 'out_channels': 4, 'features': [64, 128, 256, 512, 1024], 'sequence_length': sequence_length}
    x = torch.randn((batch_size, 6, 512, 256))
    y = torch.randn((batch_size, 4, 100))
    model = Model(config)
    model.eval()
    for i in range(500):
        starter_cnn, ender_cnn = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter_cnn.record()
        seg = model(x)
        ender_cnn.record()
        torch.cuda.synchronize()
        print(starter_cnn.elapsed_time(ender_cnn) / (batch_size * sequence_length))

    # Seems to be working

if __name__ == "__main__":
    test()