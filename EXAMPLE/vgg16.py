import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# VGG-16 configuration
cfg_vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

# Create VGG-16 model
def vgg16(num_classes=1000, batch_norm=False):
    return VGG(make_layers(cfg_vgg16, batch_norm=batch_norm), num_classes=num_classes)

# Example usage
if __name__ == "__main__":
    model = vgg16(num_classes=1000, batch_norm=True)
    print(model)

    # Export the model to ONNX
    dummy_input = torch.randn(1, 3, 224, 224)  # Create a dummy input tensor
    onnx_file_path = "vgg16.onnx"
    torch.onnx.export(model, dummy_input, onnx_file_path, verbose=True)
    print(f"ONNX model has been saved to {onnx_file_path}")