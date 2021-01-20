import torch.nn as nn

from libs.loss_fn.MBL import MultiBoxLoss

__all__ = ["get_criterion"]

def get_criterion(
    jaccard_thresh: float=0.5,
    neg_pos: float=3,
    device: str="cuda",
) -> nn.Module:
    criterion = MultiBoxLoss(jaccard_thresh, neg_pos, device)

    return criterion
