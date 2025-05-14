import torch
import torch.nn as nn
import torch.nn.functional as F


class CF_Loss(nn.Module):
    def __init__(self, img_size, num_classes=3):
        super(CF_Loss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes

        p = torch.tensor(img_size, dtype=torch.float)
        n = torch.floor(torch.log2(p))
        self.sizes = 2 ** torch.arange(n.item(), 1, -1, dtype=torch.int)

        self.ce_loss = nn.CrossEntropyLoss()

    def get_count(self, sizes, p, masks):
        B = masks.shape[0]
        counts = torch.zeros((B, len(sizes), 2), device=masks.device)

        for idx, size in enumerate(sizes):
            stride = (size, size)
            pad_size = int((size - (p % size)) % size)
            pad = nn.ZeroPad2d((0, pad_size, 0, pad_size))
            pooled = F.avg_pool2d(pad(masks), kernel_size=size, stride=stride)
            pooled = pooled * ((pooled > 0) & (pooled < size * size))

            vein_diff = (pooled[:, 0] - pooled[:, 1]).abs().sum(dim=(1, 2))
            artery_diff = (pooled[:, 2] - pooled[:, 3]).abs().sum(dim=(1, 2))
            vein_norm = (pooled[:, 1] > 0).sum(dim=(1, 2)).clamp(min=1)
            artery_norm = (pooled[:, 3] > 0).sum(dim=(1, 2)).clamp(min=1)

            counts[:, idx, 0] = vein_diff / vein_norm
            counts[:, idx, 1] = artery_diff / artery_norm

        return counts

    def forward(self, prediction, ground_truth):
        prediction = prediction.to(self.device)
        ground_truth = ground_truth.to(self.device)

        # Cross-Entropy Loss
        loss_ce = self.ce_loss(prediction, ground_truth.long())

        # Softmax & One-hot encoding
        pred_soft = F.softmax(prediction, dim=1)
        gt_onehot = F.one_hot(ground_truth.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Vascular direction loss (artery/vein swap)
        loss_vd = (
            torch.abs(pred_soft[:, 2].sum() - gt_onehot[:, 1].sum()) +
            torch.abs(pred_soft[:, 1].sum() - gt_onehot[:, 2].sum())
        ) / (prediction.shape[0] * prediction.shape[2] * prediction.shape[3])

        # FD loss
        pred_fd = pred_soft[:, 1:3]
        gt_fd = gt_onehot[:, 1:3]
        merged = torch.cat([pred_fd, gt_fd], dim=1)

        counts = self.get_count(self.sizes, prediction.shape[-1], merged)

        artery = torch.sqrt(torch.sum(self.sizes * counts[..., 1] ** 2, dim=1)).mean()
        vein = torch.sqrt(torch.sum(self.sizes * counts[..., 0] ** 2, dim=1)).mean()
        total_scale = torch.sqrt(torch.sum(self.sizes ** 2))
        loss_fd = (artery + vein) / total_scale

        return loss_ce + loss_fd + loss_vd
