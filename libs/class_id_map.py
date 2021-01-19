from typing import Dict


def get_cls2id_map() -> Dict[str, int]:
    cls2id_map = {
        "aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3,
        "bottle": 4, "bus": 5, "car": 6, "cat": 7,
        "chair": 8, "cow": 9, "diningtable": 10, "dog": 11,
        "horse": 12, "motorbike": 13, "person": 14, "pottedplant": 15,
        "sheep": 16, "sofa": 17, "train": 18, "tvmonitor": 18
    }
    return cls2id_map

def get_id2cls_map() -> Dict[int, str]:
    cls2id_map = get_cls2id_map()
    return {val: key for key, val in cls2id_map.items()}
