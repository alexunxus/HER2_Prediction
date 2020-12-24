import typing
import os

from torchvision.models import resnet50, resnext50_32x4d, resnet34, resnet18, vgg16, vgg11, vgg11_bn
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.optim

from .util import replace_bn

class DomainNorm(nn.Module):
    def __init__(self, channel: int, l2: bool=True) -> None:
        super(DomainNorm, self).__init__()
        self.normalize = nn.InstanceNorm2d(num_features=channel, affine=False)
        self.l2 = l2
        self.weight = nn.Parameter(torch.ones(1,channel,1,1))
        self.bias = nn.Parameter(torch.zeros(1,channel,1,1))
        self.weight.requires_grad = True
        self.bias.requires_grad = True

    def forward(self, x: Tensor) -> Tensor:
        x = self.normalize(x)
        if self.l2:
            return F.normalize(x, p=2, dim=1)
        return x * self.weight + self.bias

backbone_dict = {
    'R-18': resnet18,
    'R-34': resnet34,
    'baseline':resnet50,
    'R-50-xt': resnext50_32x4d,
    'vgg16': vgg16,
    'vgg11': vgg11,
    'vgg11_bn': vgg11_bn
    #'R-50-st': torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True),
    #'enet-b0': EfficientNet.from_pretrained(f'efficientnet-b0'),
    #'enet-b1': EfficientNet.from_pretrained(f'efficientnet-b1'),
}

norm_dict = {
    'gn': nn.GroupNorm,
    'dn': DomainNorm,
}

def get_backbone(string:str, pretrained:bool =True) -> nn.Module:
    if string == 'R-50-st':
        torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
        return torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=pretrained)
    if 'enet-b' in string:
        return EfficientNet.from_pretrained(f'efficientnet-b{string[-1]}'),
    return backbone_dict[string](pretrained=pretrained)
    
class CustomModel(nn.Module):
    def __init__(self, backbone:str, num_grade:int, resume_from:str=None, norm:str='bn')->None:
        super(CustomModel, self).__init__()
        if (resume_from and not os.path.isfile(resume_from)):
            raise ValueError(f"Path {resume_from} does not exist, cannot load weight.")
        weight_path = f'./checkpoint/.{backbone}_imagenet_weight'
        
        if resume_from is None and os.path.isfile(weight_path):
            resume_from = weight_path

        self.backbone = get_backbone(string=backbone, pretrained=False if resume_from else True)
        self.linear_side_chain = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000)
        )
        self.linear = nn.Linear(1000, num_grade)
        self.activation = nn.Sigmoid()

        assert norm in ['bn', 'gn', 'dn'], f"Unkonwn norm type {norm}"
        if norm != 'bn':
            print(f'Replacing bn to {norm}')
            self = replace_bn(self, norm_dict[norm])
        
        if resume_from:
            self.resume_from_path(resume_from)
        else: 
            self.init_weights()
            print(f"Saving model to {weight_path}")
            torch.save(self.state_dict(), weight_path)
        
        print(f"{backbone} is prepared.")

    def resume_from_path(self, resume_from:str)->None:
        self.load_state_dict(torch.load(resume_from), strict=False)
        print(f"Resume from checkpoint {resume_from}")

    def init_weights(self) -> None:
        tails = [m for m in self.linear_side_chain]
        tails.append(self.linear)
        for m in tails:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.BatchNorm1d:
                m.weight.data.fill_(0)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.add(x, self.linear_side_chain(x))
        x = self.linear(x)
        return self.activation(x)

def build_optimizer(type, model, lr):
    if type == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    if type == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    else:
        raise ValueError(f"Unknown optimizer type {type}")
