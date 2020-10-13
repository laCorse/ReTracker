from __future__ import absolute_import
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
# For debug
from IPython import embed

class ResNet50(nn.Module):
    def __init__(self, num_classes, loss = {'softmax', 'metric'}, **kwargs):
        super(ResNet50, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)

        # 去掉下面两层（即原先网络的最后两层）
        # (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
        # (fc): Linear(in_features=2048, out_features=1000, bias=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.loss = loss


    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        # 归一化
        # x = 1. * x / (torch.norm(x, 2, dim=-1, keepdim=True).expand_as(x) + 1e-12)

        if not self.training:
            return x
        if self.loss == {'softmax'}:
            y = self.classifier(x)
            return y
        elif self.loss == {'metric'}:
            return x
        elif self.loss == {'softmax', 'metric'}:
            y = self.classifier(x)
            return y, x
        else:
            raise NotImplementedError("No such loss function!")
        # embed()




if __name__ == '__main__':
    model = ResNet50(num_classes=751)
    imgs = torch.Tensor(32, 3, 256, 128)
    f = model(imgs)