import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import make_vgg, make_extras, L2Norm, make_loc_conf, DBox, Detect


class SSD(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()

        self.phase =phase
        self.num_classes = cfg["num_classes"]

        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(self.num_classes, cfg["bbox_aspect_num"])

        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()

        if phase == "inference":
            self.detect = Detect()

    def forward(self, x):
        sources = []
        loc = []
        conf = []

        for k in range(23):
            x = self.vgg[k](x)

        source1 = self.L2Norm(x)
        sources.append(source1)

        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        sources.append(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 ==1:
                sources.append(x)

        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        out = (loc, conf, self.dbox_list)

        if self.phase == "inference":
            return self.detect(out[0], out[1], out[2])
        else:
            return out


def ssd_test():
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
    ssd=SSD(phase="train", cfg=ssd_cfg)
    print(ssd)


if __name__ == "__main__":
    ssd_test()
