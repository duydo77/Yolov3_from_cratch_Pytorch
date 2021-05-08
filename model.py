import torch
import torch.nn as nn

# Touple:   (cout_chanel, kernel_size, stride)
# List:     B residual block followed by the number of repeats
# S:        scale prediction block and computing yolo loss
# U:        upsampling feature map and concatenating with previous layer
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4], # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x) 
            
class Residual(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels//2, kernel_size=1),
                    CNNBlock(channels//2, channels, kernel_size=3, padding=1)
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats
    
    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layers(x)
        
        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(2 * in_channels, 3 * (5 + num_classes), bn_act=False, kernel_size=1)
        )
        self.num_classes = num_classes   

    def forward(self, x):
        b, c, h, w = x.shape
        return (self.pred(x)
                .reshape(b, 3, (5 + num_classes), h, w)
                .permute(0, 1, 3, 4, 2)
        )
        # N x 3 x 13 x 13 x 5
        # N x 3 x 52 x 52 x 5 
        # image_id, anchor_id, h_offset_cell, w_offset_cell, pred_result_per_ancher_per_cell
    
class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_convolution_layers()

    def forward(self, x):
        output = []
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                output.append(layer(x))
                continue
            
            x = layer(x)
            
            if isinstance(layer, Residual) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return output

    def _create_convolution_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            
            if (isinstance(module, tuple)):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels, 
                        out_channels,
                        kernel_size = kernel_size, 
                        stride = stride,
                        padding = 1 if kernel_size == 3 else 0
                    )
                )
                in_channels = out_channels
            
            if isinstance(module, list):
                layers.append(
                    Residual(in_channels, num_repeats = module[1])
                ) 

            if isinstance(module, str):
                
                if (module == 'U'):
                    layers.append( nn.Upsample(scale_factor=2) )
                    in_channels = 3 * in_channels

                if (module == 'S'):
                    layers += [
                        Residual(in_channels),
                        CNNBlock(in_channels, in_channels//2, kernel_size=1),
                        ScalePrediction(in_channels//2, self.num_classes)
                    ]
                    in_channels = in_channels // 2

        return layers

if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2,3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("su")