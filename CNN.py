import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 两个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 5),  # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # kernel_size, stride
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),  # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # kernel_size, stride
        )
        # 三个全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 22 * 22, 660),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(660, 88),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(88, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
