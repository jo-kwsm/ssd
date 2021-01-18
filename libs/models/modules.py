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
            layers.append(conv2d)
            layers.append(nn.ReLU(inplace=True))
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers.append(pool5)
    layers.append(conv6)
    layers.append(nn.ReLU(inplace=True))
    layers.append(conv7)
    layers.append(nn.ReLU(inplace=True))

    return nn.ModuleList(layers)


def module_test():
    vgg_test = make_vgg()
    print(vgg_test)

if __name__ == "__main__":
    module_test()
