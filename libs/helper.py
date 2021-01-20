import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .meter import AverageMeter, ProgressMeter

__all__ = ["train", "evaluate"]


def do_one_iteration(
    sample: Dict[str, Any],
    model: nn.Module,
    criterion: Any,
    device: str,
    iter_type: str,
    optimizer: Optional[optim.Optimizer] = None,
) -> Tuple[int, float, np.ndarray, np.ndarray]:

    if iter_type not in ["train", "evaluate"]:
        raise ValueError("iter_type must be either 'train' or 'evaluate'.")

    if iter_type == "train" and optimizer is None:
        raise ValueError("optimizer must be set during training.")

    x = sample["img"].to(device)
    t = [ann.to(device) for ann in sample["target"]]

    batch_size = x.shape[0]

    output = model(x)
    loss_l, loss_c = criterion(output, t)
    loss = loss_l + loss_c

    # TODO 評価指標を計算
    # accs = calc_accuracy(output, t, topk=(1,))
    # acc1 = accs[0]
    # _, pred = output.max(dim=1)
    # gt = t.to("cpu").numpy()
    # pred = pred.to("cpu").numpy()
    pred = [ann.to("cpu").numpy() for ann in t]
    gt = [ann.to("cpu").numpy() for ann in t]

    if iter_type == "train" and optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
        optimizer.step()

    return batch_size, loss.item(), gt, pred


def train(
    loader: DataLoader,
    model: nn.Module,
    criterion: Any,
    optimizer: optim.Optimizer,
    epoch: int,
    device: str,
    interval_of_progress: int = 50,
) -> Tuple[float, float, float]:

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    # TODO object detection の評価指標に変更
    # top1 = AverageMeter("Acc@1", ":6.2f")

    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses], # TODO object detection の評価指標追加
        prefix="Epoch: [{}]".format(epoch),
    )

    gts = []
    preds = []

    model.train()

    end=time.time()
    for i, (imgs, targets) in enumerate(loader):
        sample = {
            "img": imgs,
            "target": targets,
        }

        data_time.update(time.time() - end)

        batch_size, loss, gt, pred = do_one_iteration(
            sample,
            model,
            criterion,
            device,
            "train",
            optimizer,
        )

        losses.update(loss, batch_size)

        gts += list(gt)
        preds += list(pred)

        batch_time.update(time.time()-end)
        end = time.time()

        if i!=0 and i % interval_of_progress == 0:
            progress.display(i)

    return losses.get_average()


def evaluate(
    loader: DataLoader,
    model: nn.Module,
    criterion: Any,
    device: str,
) -> float:
    losses = AverageMeter("Loss", ":.4e")

    gts = []
    preds = []

    model.eval()

    with torch.no_grad():
        for img, target in loader:
            sample = {
                "img": img,
                "target": target,
            }

            batch_size, loss, gt, pred = do_one_iteration(
                sample,
                model,
                criterion,
                device,
                "evaluate",
            )

            losses.update(loss, batch_size)

            gts += list(gt)
            preds += list(pred)

            # TODO object detection の評価指標計算

    return losses.get_average()
