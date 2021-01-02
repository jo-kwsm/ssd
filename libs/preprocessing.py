from typing import Any, List
import numpy as np
import xml.etree.ElementTree as ET

class Anno_xml2list():
    def __init__(self, classes: List) -> None:
        self.classes = classes

    def __call__(self,
        xml_path: str,
        width: int,
        height: int
    ) -> np.array(Any):
        res = []

        xml = ET.parse(xml_path).getroot()

        for obj in xml.iter("object"):
            difficult = int(obj.find("difficult").text)
            if difficult == 1:
                continue
            
            obj_box = []

            name = obj.find("name").text.lower().strip()
            bndbox = obj.find("bndbox")

            pts = ["xmin", "ymin", "xmax", "ymax"]

            for pt in pts:
                cur_pixel = int(bndbox.find(pt).text) - 1
                if pt.find("y"):
                    cur_pixel /= height
                else:
                    cur_pixel /= width
                obj_box.append(cur_pixel)

            label_idx = self.classes.index(name)
            obj_box.append(label_idx)

            res += [obj_box]
        
        return np.array(res)

def test():
    import pandas as pd
    import cv2

    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

    transform_anno = Anno_xml2list(voc_classes)

    idx = 1
    val_list = pd.read_csv("csv/val.csv")
    image_path = val_list["image_path"]
    anno_path = val_list["annotate_path"]

    img = cv2.imread(image_path.iloc[idx])
    height, width, channels = img.shape
    res = transform_anno(anno_path.iloc[idx], width, height)
    print(res)

if __name__ == "__main__":
    test()
