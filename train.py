import argparse
import os
import time

import pandas as pd
import torch
import torch.optim as optim

from libs.checkpoint import resume, save_checkpoint
from libs.class_id_map import get_cls2id_map
from libs.config import get_config
from libs.dataset import get_dataloader
from libs.device import get_device
from libs.loss_fn import get_criterion
from libs.mean_std import get_mean
from libs.models import get_model
from libs.preprocessing import Anno_xml2list
from libs.transformer import DataTransform

random_seed = 1234


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""
        train SSD for object detection with KMNIST Dataset.
        """
    )
    parser.add_argument("config", type=str, help="path of a config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Add --resume option if you start training from checkpoint.",
    )

    return parser.parse_args()


def main():
    args = get_arguments()
    config = get_config(args.config)

    result_path = os.path.dirname(args.config)
    experiment_name = os.path.basename(result_path)

    if os.path.exists(os.path.join(result_path, "final_model.prm")):
        print("Already done.")
        return

    device = get_device(allow_only_gpu=True)

    transform = DataTransform(config.size, get_mean())
    voc_classes = [get_cls2id_map()]

    train_loader = get_dataloader(
        config.train_csv,
        phase="train",
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
        pin_memory=True,
        drop_last=True,
        transform=transform
        transform_anno=Anno_xml2list(voc_classes),
    )

    val_loader = get_dataloader(
        config.val_csv,
        phase="val",
        batch_size=1,
        shuffle=True,
        num_workers=config.num_workers
        pin_memory=True,
        drop_last=True,
        transform=transform
        transform_anno=Anno_xml2list(voc_classes),
    )

    n_classes = len(voc_classes) + 1
    model = get_model(
        input_size=config.size,
        n_classes=n_classes,
        pretrained=config.pretrained,
    )
    model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=0.9,
        weight_decay=5e-4
    )

    begin_epoch = 0
    best_loss = float("inf")
    # TODO 要素の検討
    log = pd.DataFrame(
        columns=[
            "epoch",
            "lr",
            "train_time[sec]",
            "train_loss",
            "train_acc@1",
            "train_f1s",
            "val_time[sec]",
            "val_loss",
            "val_acc@1",
            "val_f1s",
        ]
    )

    if args.resume:
        resume_path = os.path.join(result_path, "checkpoint.pth")
        begin_epoch, model, optimizer, best_loss = resume(resume_path, model, optimizer)

        log_path = os.path.join(result_path, "log.csv")
        assert os.path.exists(log_path), "there is no checkpoint at the result folder"
        log = pd.read_csv(log_path)

    criterion = get_criterion(device=device)

    print("---------- Start training ----------")

    for epoch in range(begin_epoch, config.max_epoch):
        start = time.time()
        # TODO train
        train_time = int(time.time() - start)

        start = time.time()
        # TODO val
        val_loss = 0
        val_time = int(time.time() - start)

        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(result_path, "best_model.prm"),
            )

        save_checkpoint(result_path, epoch, model, optimizer, best_loss)

        # TODO logを保存

        # TODO logを出力

    torch.save(model.state_dict(), os.path.join(result_path, "final_model.prm"))

    os.remove(os.path.join(result_path, "checkpoint.pth"))

    print("Done")


if __name__ == "__main__":
    main()
