import os
import urllib

import torch
import torch.nn as nn

from .SSD import SSD

__all__ = ["get_model"]


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:  # バイアス項がある場合
            nn.init.constant_(m.bias, 0.0)


def get_model(input_size: int, n_classes: int, phase: str,pretrained: bool = True) -> nn.Module:
    ssd_cfg = {
        "num_classes": n_classes,
        "input_size": input_size,
        "bbox_aspect_num": [4, 6, 6, 6, 4, 4],
        "feature_maps": [38, 19, 10, 5, 3, 1],
        "steps": [8, 16, 32, 64, 100, 300],
        "min_sizes": [30, 60, 111, 162, 213, 264],
        "max_sizes": [60, 111, 162, 213, 264, 315],
        "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    }
    model=SSD(phase=phase, cfg=ssd_cfg)

    if pretrained:
        weights_dir = "./libs/models/weights"
        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)

        vgg_weights_path = os.path.join(weights_dir, "vgg16_reducedfc.pth") 

        if not os.path.exists(vgg_weights_path):
            print("start downloading weights of vgg")
            url = "https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth"
            urllib.request.urlretrieve(url, vgg_weights_path)

        vgg_weights = torch.load(vgg_weights_path)
        model.vgg.load_state_dict(vgg_weights)

    model.extras.apply(weights_init)
    model.loc.apply(weights_init)
    model.conf.apply(weights_init)

    return model
