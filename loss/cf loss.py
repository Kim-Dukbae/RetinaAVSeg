import torch
import torch.nn as nn
import torch.nn.functional as F

class CF_Loss(nn.Module):
    def __init__(self, img_size,beta,alpha,gamma):
        super(CF_Loss, self).__init__()
        '''
        research paper: https://discovery.ucl.ac.uk/id/eprint/10188133/1/CF-Loss-accepted.pdf
        git hub : https://github.com/rmaphoh/feature-loss/blob/main/scripts/loss.py
        [KO]  
        이 코드는 Yukun Zhou 외 연구진의 "CF-Loss: Clinically-relevant feature optimised loss function..." (MedIA, 2024) 논문 구현을 기반으로 수정되었습니다.  
        원본 코드는 MIT License 하에 배포되었으며, 본 수정본은 실험 및 재현 목적에 맞게 일부 구조와 변수명을 개선한 것입니다.  
        원저작자: Yukun Zhou (2023)  
        수정자: 형준 김 (Hyungjun Kim, 2024)

        [EN]  
        This code is a modified implementation based on the original work by Yukun Zhou et al.,  
        published in the paper *“CF-Loss: Clinically-relevant feature optimised loss function...” (MedIA, 2024)*.  
        The original code was released under the MIT License.  
        This version includes structural refactoring and variable renaming for experimental and clarity purposes.  
        Original author: Yukun Zhou (2023)  
        Modified by: Hyungjun Kim (2024)

        * 원 논문에서 CF-Loss는 4-class 분할 (background, artery, vein, uncertain/overlap)을 기반으로 하지만,  
          본 프로젝트에서는 혈관의 lumen 관측 특성상 overlap class를 vein으로 통합하고, 3-class segmentation 구조로 구현하였습니다.
        '''

        # hyper parameters
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma

        # 입력 이미지 크기를 실수형 텐서로 변환 (예: 512 → tensor(512.))
        self.p = torch.tensor(img_size, dtype=torch.float)
        # 이미지 크기에서 log2 기반의 최대 분할 단계 계산 (정수 스케일 수)
        self.n = torch.floor(torch.log2(self.p))
        # 가장 큰 박스 크기부터 작은 박스 크기까지의 2의 거듭제곱 텐서 생성 (예: [256, 128, ..., 4])
        self.sizes = 2**torch.arange(self.n.item(), 1, -1).to(dtype=torch.int)

        # Entropy Loss 계산
        self.CE = nn.CrossEntropyLoss()
    
    def label_to_onehot(self, pred, ture):
        # label encoding -> one-hot encoding 변환
        # class 0: background, 1: artery, 2: vein
        encode_tensor  = F.one_hot(ture.to(torch.int64), num_classes=3).permute(0, 3, 1, 2).contiguous() 
        # one-hot encoding을 cuda device로 이동하고 float32 타입으로 변환
        encode_tensor  = encode_tensor .to(device=torch.device('cuda'), dtype=torch.float32)
        # 예측값에 softmax 적용.
        pred_softmax = F.softmax(pred,dim=1)
        return pred_softmax, encode_tensor

    def get_count(self, sizes, p, pred, true):
        # Batch size, box 수, 2(artery and vein의 저장값)
        counts = torch.zeros((pred.shape[0], len(sizes), 2))

        idx = 0 # index 초기화
        # sizes로부터 sequence 순회
        for size in sizes:
            stride = (size, size) # 게산할 box 크기
            pad_size = torch.where(
                (p % size) == 0, # 이미지 크기 p가 박스 크기 size로 나누어떨어지면 
                torch.tensor(0, dtype=torch.int), # 패딩은 0,
                (size - p % size).int() # 그렇지 않으면 박스 크기에 맞게 패딩 추가
            )

            # 오른쪽과 아래쪽에만 zero-padding 적용
            pad = nn.ZeroPad2d((0,pad_size, 0, pad_size))
            # 박스 단위의 요약된 값 하나(평균값)
            pool = nn.AvgPool2d(kernel_size = (size, size), stride = stride)

            # padding -> average pooling -> 2차원 값.
            s_pred = pool(pad(pred)) 
            s_true = pool(pad(true)) 
            # 유효한 박스만 선택하기 위한 
            s_pred = s_pred*((s_pred> 0) & (s_pred < (size*size)))
            s_true = s_true*((s_true> 0) & (s_true < (size*size)))

            eps = 1e-6  # 작은 값 추가로 ZeroDivision 방지
            counts[..., idx, 0] = (s_pred[:, 0, ...] - s_true[:, 0, ...]).abs().sum() / ((s_true[:, 0, ...] > 0).sum() + eps)
            counts[..., idx, 1] = (s_pred[:, 1, ...] - s_true[:, 1, ...]).abs().sum() / ((s_true[:, 1, ...] > 0).sum() + eps)
    
            idx += 1

        return counts
    
    def forward(self, pred, true):
        # cross entropy loss 계산
        loss_CE = self.CE(pred, true)
        
        # label encoding -> one-hot으로 변경 
        encode_tensor, pred_ = self.label_to_onehot(pred, true)
        # vessel 밀도 계산
        area = pred_.shape[0] * pred_.shape[2] * pred_.shape[3]
        diff_artery = (pred_[:, 1, ...] - encode_tensor[:, 1, ...]).abs().sum()
        diff_vein   = (pred_[:, 2, ...] - encode_tensor[:, 2, ...]).abs().sum()
        Loss_vd = (diff_artery + diff_vein) / area

        # 프렉탈 계산
        counts = self.get_count(self.sizes, self.p, pred_, true)
        artery_ = torch.sqrt(torch.sum(self.sizes*((counts[...,0])**2)))
        vein_ = torch.sqrt(torch.sum(self.sizes*((counts[...,1])**2)))
        size_t = torch.sqrt(torch.sum(self.sizes**2))
        loss_FD = (artery_+vein_)/size_t/pred_.shape[0]
        
        return self.beta*loss_CE + self.alpha*loss_FD + self.gamma*Loss_vd
