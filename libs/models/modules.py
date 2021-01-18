from math import sqrt
from itertools import product

import torch
import torch.nn as nn


def make_vgg():
    layers = []
    in_channels = 3  # 色チャネル数

    # vggモジュールで使用する畳み込み層やマックスプーリングのチャネル数
    config = [64, 64, "M", 128, 128, "M", 256, 256,
           256, "MC", 512, 512, 512, "M", 512, 512, 512]
    
    for v in config:
        if v=="M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif v=="MC":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers.extend([conv2d, nn.ReLU(inplace=True)])
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers.extend([pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)])

    return nn.ModuleList(layers)


def make_extras():
    layers = []
    in_channels = 1024

    cfg = [256, 512, 128, 256, 128, 256, 128, 256]
    layers.extend([nn.Conv2d(in_channels, cfg[0], kernel_size=(1))])
    layers.extend([nn.Conv2d(cfg[0], cfg[1], kernel_size=(3), stride=2, padding=1)])
    layers.extend([nn.Conv2d(cfg[1], cfg[2], kernel_size=(1))])
    layers.extend([nn.Conv2d(cfg[2], cfg[3], kernel_size=(3), stride=2, padding=1)])
    layers.extend([nn.Conv2d(cfg[3], cfg[4], kernel_size=(1))])
    layers.extend([nn.Conv2d(cfg[4], cfg[5], kernel_size=(3))])
    layers.extend([nn.Conv2d(cfg[5], cfg[6], kernel_size=(1))])
    layers.extend([nn.Conv2d(cfg[6], cfg[7], kernel_size=(3))])

    return nn.ModuleList(layers)


def make_loc_conf(num_classes=21, bbox_aspect_num=[4, 6, 6, 6, 4, 4]):
    loc_layers = []
    conf_layers = []

    loc_layers.extend([nn.Conv2d(512, bbox_aspect_num[0]*4, kernel_size=3, padding=1)])
    conf_layers.extend([nn.Conv2d(512, bbox_aspect_num[0]*num_classes, kernel_size=3, padding=1)])

    loc_layers.extend([nn.Conv2d(1024, bbox_aspect_num[1]*4, kernel_size=3, padding=1)])
    conf_layers.extend([nn.Conv2d(1024, bbox_aspect_num[1]*num_classes, kernel_size=3, padding=1)])

    loc_layers.extend([nn.Conv2d(512, bbox_aspect_num[2]*4, kernel_size=3, padding=1)])
    conf_layers.extend([nn.Conv2d(512, bbox_aspect_num[2]*num_classes, kernel_size=3, padding=1)])

    loc_layers.extend([nn.Conv2d(256, bbox_aspect_num[3]*4, kernel_size=3, padding=1)])
    conf_layers.extend([nn.Conv2d(256, bbox_aspect_num[3]*num_classes, kernel_size=3, padding=1)])

    loc_layers.extend([nn.Conv2d(256, bbox_aspect_num[4]*4, kernel_size=3, padding=1)])
    conf_layers.extend([nn.Conv2d(256, bbox_aspect_num[4]*num_classes, kernel_size=3, padding=1)])

    loc_layers.extend([nn.Conv2d(256, bbox_aspect_num[5]*4, kernel_size=3, padding=1)])
    conf_layers.extend([nn.Conv2d(256, bbox_aspect_num[5]*num_classes, kernel_size=3, padding=1)])

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)


class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tnesor(input_channels))
        self.scale = scale
        self.reset_parameters()
        self.eps = 1e-10

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.scale)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True)
        x = torch.div(x, norm)

        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        out = weights * Anno_xml2list

        return out


class DBox(object):
    def __init__(self, cfg):
        super(DBox, self).__init__()
        self.img_size = cfg["input_size"]
        self.feature_maps = cfg["feature_maps"]
        self.num_priors = len(self.feature_maps)
        self.steps = cfg["steps"]
        self.min_sizes = cfg["min_sizes"]
        self.max_sizes = cfg["max_sizes"]
        self.aspect_ratios = cfg["aspect_ratios"]

    def make_dbox_list(self):
        mean = []

        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.img_size / self.steps[k]

                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                s_k = self.min_sizes[k] / self.img_size

                mean.extend([cx, cy, s_k, s_k])

                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.img_size))
                mean.extend([cx, cy, s_k_prime, s_k_prime])

                for ar in self.aspect_ratios[k]:
                    mean.extend([cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)])
                    mean.extend([cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)])

        out = torch.Tensor(mean).view(-1, 4)
        out.clamp_(max=1, min=0)

        return out


def module_test():
    import pandas as pd

    vgg_test = make_vgg()
    extras_test = make_extras()
    loc_test, conf_test = make_loc_conf()
    print(vgg_test)
    print(extras_test)
    print(loc_test)
    print(conf_test)

    ssd_cfg = {
        "num_classes": 21,
        "input_size": 300,
        "bbox_aspect_num": [4, 6, 6, 6, 4, 4],
        "feature_maps": [38, 19, 10, 5, 3, 1],
        "steps": [8, 16, 32, 64, 100, 300],
        "min_sizes": [30, 60, 111, 162, 213, 264],
        "max_sizes": [60, 111, 162, 213, 264, 315],
        "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    }
    
    dbox = DBox(ssd_cfg)
    dbox_list = dbox.make_dbox_list()
    print(pd.DataFrame(dbox_list.numpy()))


if __name__ == "__main__":
    module_test()
