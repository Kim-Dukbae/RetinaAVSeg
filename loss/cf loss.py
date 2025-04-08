import torch.nn as nn

def encode_mask(ground_truth,prediction):
    masks_pred_softmax = F.softmax(prediction,dim=1).to(device )
    return ground_truth, masks_pred_softmax


class CF_Loss(nn.Module):
    def __init__(self, img_size):
        super(CF_Loss, self).__init__()

        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.p = torch.tensor(img_size, dtype=torch.float)
        self.n = torch.log(self.p) / torch.log(torch.tensor([2]).to(self.device))
        self.n = torch.floor(self.n)
        self.sizes = 2 ** torch.arange(self.n.item(), 1, -1).to(dtype=torch.int)
        self.CE =  nn.CrossEntropyLoss()

    def get_count(self, sizes, p, masks_pred_softmax):
        counts = torch.zeros((masks_pred_softmax.shape[0], len(sizes), 2))
        index = 0

        for size in sizes:
            stride = (size, size)
            pad_size = torch.where((p % size) == 0, torch.tensor(0, dtype=torch.int), (size - p % size).to(dtype=torch.int))
            pad = nn.ZeroPad2d((0, pad_size, 0, pad_size))
            pool = nn.AvgPool2d(kernel_size=(size, size), stride=stride)

            S = pad(masks_pred_softmax)
            S = pool(S)
            S = S * ((S > 0) & (S < (size * size)))
            counts[..., index, 0] = (S[:, 0, ...] - S[:, 1, ...]).abs().sum() / (S[:, 1, ...] > 0).sum()
            counts[..., index, 1] = (S[:, 2, ...] - S[:, 3, ...]).abs().sum() / (S[:, 3, ...] > 0).sum()

            index += 1

        return counts

    def forward(self, prediction, ground_truth):
        masks_pred_softmax = F.softmax(prediction,dim=1).to(device)
        encode_tensor = ground_truth
        encode_tensor= encode_tensor.to(self.device)
        masks_pred_softmax= masks_pred_softmax.to(self.device)
        loss_CE = self.CE(prediction, ground_truth)
        # loss_CE = self.CE(masks_pred_softmax, encode_tensor).to(self.device)

        # Swap class 1 (artery) and class 2 (vein) for Loss_vd calculation
        Loss_vd = (torch.abs(masks_pred_softmax[:, 2, ...].sum() - encode_tensor[:, 1, ...].sum()) +
                   torch.abs(masks_pred_softmax[:, 1, ...].sum() - encode_tensor[:, 2, ...].sum())) / (masks_pred_softmax.shape[0] * masks_pred_softmax.shape[2] * masks_pred_softmax.shape[3])

        masks_pred_softmax = masks_pred_softmax[:, 1:3, ...]
        encode_tensor = encode_tensor[:, 1:3, ...]
        masks_pred_softmax = torch.cat((masks_pred_softmax, encode_tensor), 1)
        counts = self.get_count(self.sizes, self.p, masks_pred_softmax)

        artery_ = torch.sqrt(torch.sum(self.sizes * ((counts[..., 1]) ** 2)))
        vein_ = torch.sqrt(torch.sum(self.sizes * ((counts[..., 0]) ** 2)))
        size_t = torch.sqrt(torch.sum(self.sizes ** 2))
        loss_FD = (artery_ + vein_) / size_t / masks_pred_softmax.shape[0]

        return loss_CE, loss_FD, Loss_vd
