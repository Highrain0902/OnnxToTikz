import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # Conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # MaxPool1

            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # Conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # MaxPool2

            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # Conv3
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # Conv4
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # MaxPool3
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),  # FC1
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),  # FC2
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)  # FC3
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Example of creating the model
model = AlexNet(num_classes=1000)
dummy_input = torch.randn(1,3,244,244)
onnx_file_path = "model.onnx"

torch.onnx.export(
    model,                         # 要导出的模型
    dummy_input,                   # 虚拟输入
    onnx_file_path,                # 输出 ONNX 文件路径
    verbose=False                  # 打印导出过程的详细信息
)
print("ONNX 文件已导出")

#该模型分为两部分：features 用于提取图像特征，classifier 用于分类。
#卷积层使用了 ReLU 激活函数。
#池化层使用 MaxPool2d 进行下采样。
#全连接层通过 Dropout 进行正则化，防止过拟合。
