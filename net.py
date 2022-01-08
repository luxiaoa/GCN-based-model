import torch
import torch.nn as nn
from math import sqrt
from torch.nn import functional as F
from torchvision import models
from GCN import GCN
from BR import BR
#


class FCN_GCN(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes  # 21 in paper
        super(FCN_GCN, self).__init__()

        resnet = models.resnet50(pretrained=True)

        # self.conv1 = resnet.conv1  # 7x7,64, stride=2 (output_size=112x112)
        self.bn0 = resnet.bn1  # BatchNorm2d(64)
        self.relu = resnet.relu
        # self.maxpool = resnet.maxpool  # maxpool /2 (kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=4, padding=0)
        self.layer1 = resnet.layer1  # res-2 o/p = 56x56,256 16

        self.gcn1 = GCN(256, self.num_classes, 3)  # gcn_i after layer-1

        self.fc_classifer1 = nn.Sequential(

            nn.Linear(in_features=8 * 8 + 1, out_features=16),
            nn.ReLU()
        )
        self.fc_classifer2 = nn.Linear(in_features=16 + 1, out_features=6)

    def _classifier(self, in_c):
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_c / 2),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(in_c / 2, self.num_classes, 1),

        )

    def forward(self, x, qp):
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        fm1 = self.layer1(x)
        gc_fm1 = self.gcn1(fm1)
        out = gc_fm1
        out_flat = out.view(out.shape[0], -1)
        out_qp = torch.cat((out_flat, qp.squeeze(2)), dim=1)
        res_16 = self.fc_classifer1(out_qp)
        res = self.fc_classifer2(torch.cat((res_16, qp.squeeze(2)), dim=1))

        return res