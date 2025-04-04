from torch import nn
import torch
class EyeSeeNet(nn.Module):
    def __init__(self, num_class):
        super(EyeSeeNet, self).__init__()
        self.num_class = num_class
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 14 * 14 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_class)
        )

    def forward(self, left_eye, right_eye):
        x1 = self.conv(left_eye)
        x1 = x1.view(-1, 128 * 14 * 14)
        x2 = self.conv(right_eye)
        x2 = x2.view(-1, 128 * 14 * 14)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x
# if __name__ == '__main__':
#     import torch
#     x = torch.randn(1, 3, 256, 256)
#     model = EyeSeeNet(num_class=8)
#     y = model(x)
#     print(y.shape)