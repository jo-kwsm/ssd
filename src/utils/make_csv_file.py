import argparse
import glob, os, sys
from typing import Dict

import pandas as pd

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="make csv files for voc object detection dataset"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset/VOC2012/",
        help="path to a dataset dirctory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./csv",
        help="a directory where csv files will be saved",
    )

    return parser.parse_args()

def main() -> None:
    args = get_arguments()

    data: Dict[str, Dict[str, List[str]]] = {
        "train": {
            "image_path": [],
            "annotate_path": [],
        },
        "val": {
            "image_path": [],
            "annotate_path": [],
        },
        "test": {
            "image_path": [],
            "annotate_path": [],
        },
    }

    image_path_template = os.path.join(args.dataset_dir, "JPEGImages", "%s.jpg")
    annotate_path_template = os.path.join(args.dataset_dir, "Annotations", "%s.xml")

    for stage_name in ["train", "val"]:
        id_names = os.path.join(args.dataset_dir, "ImageSets/Main/%s.txt"%stage_name)
        
        for line in open(id_names):
            id_name = line.strip()
            image_path = image_path_template%id_name
            annotate_path = annotate_path_template%id_name
            data[stage_name]["image_path"].append(image_path)
            data[stage_name]["annotate_path"].append(annotate_path)

    # list を DataFrame に変換
    train_df = pd.DataFrame(
        data["train"],
        columns=["image_path", "annotate_path"],
    )

    val_df = pd.DataFrame(
        data["val"],
        columns=["image_path", "annotate_path"],
    )

    test_df = pd.DataFrame(
        data["test"],
        columns=["image_path", "annotate_path"],
    )

    # 保存ディレクトリがなければ，作成
    os.makedirs(args.save_dir, exist_ok=True)

    # 保存
    train_df.to_csv(os.path.join(args.save_dir, "train.csv"), index=None)
    val_df.to_csv(os.path.join(args.save_dir, "val.csv"), index=None)
    test_df.to_csv(os.path.join(args.save_dir, "test.csv"), index=None)

    print("Finished making csv files.")


if __name__ == "__main__":
    main()
