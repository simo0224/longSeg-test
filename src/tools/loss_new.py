import torch.nn as nn
import torch


class myBinaryDiceLoss(nn.Module):

    def __init__(self, smooth=1e-6, weight=None):
        super(myBinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.weight = weight

    def forward(self, predict, target):
        '''
        calculate bi-class dice loss for 3D-UNet
        Args:
            predict: [B, D, H, W], unique=[0, 1]
            target: [B, D, H, W], unique=[0, 1]

        Returns:
            dice_loss: single torch.Tensor
        '''
        assert predict.shape == target.shape, f'In binary dice calculation, predict {predict.shape} & target {target.shape} do not match'
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(2) + target.pow(2), dim=1) + self.smooth

        loss = 1 - 2 * num / den
        loss = torch.sum(loss)
        if self.weight is not None:
            loss *= self.weight

        return loss
    

class myDiceLoss(nn.Module):

    def __init__(self, opt, smooth=1e-6, weights=None):
        '''
        Args:
            smooth: float, smooth value to avoid division by zero
            weights: list, weights for different classes
            labels: list, unique labels in target
        '''
        super(myDiceLoss, self).__init__()
        self.smooth = smooth
        self.weights = weights
        if opt.weights_loss is not None:
            self.weights = opt.weights_loss
        print(f'Weights: {weights}')
        self.classes = opt.classes
        self.labels = list(range(self.classes))
        self.ignore_index = None
        if opt.ignore_bg_label:
            self.ignore_index = 0

        # assert len(weights) == len(labels), 'Not matched weights & labels lengths'

    def forward(self, predict, target):
        '''
        calculate multi-class dice loss for 3D-UNet
        Args:
            predict: [B, C, D, H, W], logits
            target: [B, D, H, W]

        Returns:
            dice_loss: single torch.Tensor
        '''
        unique, counts  = target.unique(return_counts=True) # 获取 target 中的 unique labels
        # print(f"unique: {unique}; counts: {counts}")
        unique_list = unique.tolist()

        if self.weights is None:
            weights = torch.zeros(len(self.labels))
            for i in range(len(self.labels)):
                if i in unique_list:
                    ratio = counts[unique == i].item() / counts.sum().item()
                    weights[i] = 1 / ratio  # 取出对应数量的值
                else:
                    weights[i] = 0  # 如果标签不存在，则权重为0

            # self.weights = weights
        else:
            weights = self.weights

        if self.ignore_index is not None:
            weights[self.ignore_index] = 0

        # 归一化权重
        weights = torch.tensor(weights)
        weights /= weights.sum()

        # print(f"weights: {weights}")
        total_loss = 0
        predict = torch.softmax(predict, dim=1) # [B, C, D, H, W] -> [B, C, D, H, W]


        ## one-hot encode target
        target = target.unsqueeze(1)
        target = torch.zeros_like(predict).scatter_(1, target, 1) # -> [B, C, D, H, W]


        ## calculate dice loss for each class
        for i in range(len(self.labels)):
            # if i != self.ignore_index:

            dice_loss = myBinaryDiceLoss(smooth=self.smooth, weight=weights[i])(predict[:, i], target[:, i])
            total_loss += dice_loss
        
        return torch.mean(total_loss)
