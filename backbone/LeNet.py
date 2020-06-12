import torch
import torch.nn as nn
from torch.nn import functional as F

class ConvNet1(nn.Module):
    """LeNet++ as described in the Center Loss paper."""
    def __init__(self, num_classes=10, feat_dim=2):
        super(ConvNet1, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.conv1_1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.prelu1_2 = nn.PReLU()

        self.conv2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.prelu2_2 = nn.PReLU()

        self.conv3_1 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.prelu3_2 = nn.PReLU()

        self.fc1 = nn.Linear(128 * 3 * 3, feat_dim)
        self.prelu_fc1 = nn.PReLU()
        self.fc2 = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)

        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)

        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 128 * 3 * 3)
        x = self.prelu_fc1(self.fc1(x))  # feature
        y = self.fc2(x)  # pred

        return x, y


class ConvNet2(nn.Module):
    """LeNet++ as described in the Center Loss paper."""
    def __init__(self, num_classes=10, feat_dim=2):
        super(ConvNet2, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = 2 # hyper param
        self.conv1_1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.prelu1_2 = nn.PReLU()

        self.conv2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.prelu2_2 = nn.PReLU()

        self.conv3_1 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.prelu3_2 = nn.PReLU()

        self.fc1 = nn.Linear(128 * 3 * 3, feat_dim)
        self.prelu_fc1 = nn.PReLU()
        self.fc2 = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)

        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)

        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 128 * 3 * 3)
        x = self.prelu_fc1(self.fc1(x))  # feature
        x = F.normalize(x, p=2, dim=1) * self.alpha
        y = self.fc2(x)  # pred

        return x, y


class ConvNet3(nn.Module):
    '''
    used for sphereface, cosface, arcface
    '''

    def __init__(self, num_classes=10, feat_dim=2):
        super(ConvNet3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256))
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=8, stride=1))
        self.fc1 = nn.Linear(512, feat_dim)
        self.fc2 = nn.Linear(feat_dim, num_classes)

    def forward(self, x, embed=False):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x) # feature
        y = self.fc2(x) # pred
        return x, y


__factory = {
    'softmax': ConvNet1,
    'center-loss': ConvNet1,
    'l2-softmax': ConvNet2,
    'sphereface': ConvNet3,
    'cosface': ConvNet3,
    'arcface': ConvNet3
}


def create(name, num_classes=10, feat_dim=2):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](num_classes=num_classes, feat_dim=feat_dim)


if __name__ == '__main__':
    a = create('cosface')