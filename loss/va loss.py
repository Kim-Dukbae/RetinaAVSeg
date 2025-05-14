import torch
import torch.nn as nn
import torch.nn.functional as F

class VALoss():
    def __call__(self, pred, mask):
        # log_softmax 적용
        pred_log = F.log_softmax(pred, dim=1)

        # CE loss
        ce = F.nll_loss(pred_log, mask)

        # artery/vein 마스크: boolean mask로 변경
        artery_mask = (mask == 1)
        vein_mask = (mask == 2)

        # artery 채널 logit에서 정맥 위치만 추출
        artery_logits_at_vein = pred[:, 1, :, :][vein_mask]  # (N,) 형태
        penalty_vein2artery = F.binary_cross_entropy_with_logits(
            artery_logits_at_vein, torch.zeros_like(artery_logits_at_vein)
        )

        # vein 채널 logit에서 동맥 위치만 추출
        vein_logits_at_artery = pred[:, 2, :, :][artery_mask]
        penalty_artery2vein = F.binary_cross_entropy_with_logits(
            vein_logits_at_artery, torch.zeros_like(vein_logits_at_artery)
        )

        # penalty는 오분류 시 감소되므로 1 / torch.exp()에 추가해서 증가시킴.
        miss_classification = (penalty_vein2artery + penalty_artery2vein) * 0.5

        return ce + miss_classification
