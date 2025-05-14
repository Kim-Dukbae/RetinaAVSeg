import torch
import torch.nn.functional as F


class SoftMorphology:
    @staticmethod
    def erode(img: torch.Tensor) -> torch.Tensor:
        dims = len(img.shape)
        if dims == 4:
            p1 = -F.max_pool2d(-img, (3, 1), stride=(1, 1), padding=(1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), stride=(1, 1), padding=(0, 1))
            return torch.min(p1, p2)
        elif dims == 5:
            p1 = -F.max_pool3d(-img, (3, 1, 1), stride=1, padding=(1, 0, 0))
            p2 = -F.max_pool3d(-img, (1, 3, 1), stride=1, padding=(0, 1, 0))
            p3 = -F.max_pool3d(-img, (1, 1, 3), stride=1, padding=(0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)
        else:
            raise ValueError(f"Unsupported tensor dimensions: {dims}")

    @staticmethod
    def dilate(img: torch.Tensor) -> torch.Tensor:
        dims = len(img.shape)
        if dims == 4:
            return F.max_pool2d(img, (3, 3), stride=1, padding=1)
        elif dims == 5:
            return F.max_pool3d(img, (3, 3, 3), stride=1, padding=1)
        else:
            raise ValueError(f"Unsupported tensor dimensions: {dims}")

    @staticmethod
    def open(img: torch.Tensor) -> torch.Tensor:
        return SoftMorphology.dilate(SoftMorphology.erode(img))

    @staticmethod
    def skeletonize(img: torch.Tensor, iterations: int = 3) -> torch.Tensor:
        skel = F.relu(img - SoftMorphology.open(img))
        for _ in range(iterations):
            img = SoftMorphology.erode(img)
            delta = F.relu(img - SoftMorphology.open(img))
            skel = skel + F.relu(delta - skel * delta)
        return skel


def clDice_score(y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int = 3, iterations: int = 3, smooth: float = 1.0) -> float:
    """
    Computes the clDice score between prediction and ground truth.

    Args:
        y_pred (torch.Tensor): Predicted segmentation map, shape (N, H, W)
        y_true (torch.Tensor): Ground truth segmentation map, shape (N, H, W)
        num_classes (int): Number of classes (including background)
        iterations (int): Number of iterations for soft skeletonization
        smooth (float): Smoothing factor to avoid division by zero

    Returns:
        float: clDice score
    """
    y_pred = F.one_hot(y_pred.long(), num_classes).permute(0, 3, 1, 2).float()
    y_true = F.one_hot(y_true.long(), num_classes).permute(0, 3, 1, 2).float()

    y_pred_fg = y_pred[:, 1:, ...]  # exclude background
    y_true_fg = y_true[:, 1:, ...]

    skel_pred = SoftMorphology.skeletonize(y_pred_fg, iterations)
    skel_true = SoftMorphology.skeletonize(y_true_fg, iterations)

    tprec = (torch.sum(skel_pred * y_true_fg) + smooth) / (torch.sum(skel_pred) + smooth)
    tsens = (torch.sum(skel_true * y_pred_fg) + smooth) / (torch.sum(skel_true) + smooth)

    cl_dice = (2 * tprec * tsens) / (tprec + tsens + 1e-8)
    return cl_dice.item()
