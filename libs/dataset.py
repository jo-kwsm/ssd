import os, sys
from typing import Any, Optional

import pandas as pd
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

def get_dataloader(
    csv_file: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
    transform: Optional[transforms.Compose] = None
) -> DataLoader:

    data = VOCDataset(
        csv_file,
        transform=transform
    )
    
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return dataloader


class VOCDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        phase: str,
        transform: Optional[transforms.Compose] = None,
        transform_anno: Optional[any] = None
    ) -> None:
        super().__init__()
        assert os.path.exists(csv_file)

        csv_path = os.path.join(csv_file)

        self.df = pd.read_csv(csv_path)
        self.phase = phase
        self.transform = transform
        self.transform_anno = transform_anno

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Any:
        img, gt, w, h = self.pull_item(idx)
        return img, gt

    def pull_item(self, idx: int) -> Any:
        image_file_path = self.df.iloc[idx]["image_path"]
        anno_file_path = self.df.iloc[idx]["annotate_path"]
        
        img = img = cv2.imread(image_file_path)
        h, w, _ = img.shape
        anno_list = self.transform_anno(anno_file_path, w, h)
        # 前処理
        img, boxes, labels = self.transform(
            img, self.phase, anno_list[:, :4], anno_list[:, 4])
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return img, gt, w, h

def test():
    print(sys.path)
    from preprocessing import Anno_xml2list
    from transformer import DataTransform
    
    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']
    color_mean = (104, 117, 123)
    input_size = 300
    train_path = "csv/train.csv"
    val_path = "csv/val.csv"

    train_dataset = VOCDataset(train_path, phase="train", transform=DataTransform(
    input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))

    val_dataset = VOCDataset(val_path, phase="val", transform=DataTransform(
    input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))

    print(val_dataset.__getitem__(1))


if __name__ == "__main__":
    test()
