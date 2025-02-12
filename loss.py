## This is for the loss function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class myDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(myDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, mask):
        '''
        input: logits (B=1, C=classes, D, H, W), mask (B=1, D, H, W)
        output: loss
        '''
        
        # mask = mask.squeeze(0) ## (B=1, D, H, W)
        n_classes = logits.size(1)
        probs = F.sigmoid(logits)

        ## mask coded in ONE-HOT
        mask_1hot = F.one_hot(mask, num_classes=n_classes).permute(0,4,1,2,3).float()

        ### check if mask_1hot is correct
        # cnt_wrong = 0
        # for i in range(mask.shape[1]):
        #     for j in range(mask.shape[2]):
        #         for k in range(mask.shape[3]):
        #             num = mask[0,i,j,k]
        #             if not mask_1hot[0,num,i,j,k]==1:
        #                 cnt_wrong+=1
        # print(cnt_wrong)

        ## NOT consider the background
        # seg_label_probs = probs[:,1:,:,:,:] ## (B=1, C=classes-1, D, H, W)
        # seg_label_mask = mask_1hot[:,1:,:,:,:] ## (B=1, C=classes-1, D, H, W)
        seg_label_probs = probs ## (B=1, C=classes-1, D, H, W)
        seg_label_mask = mask_1hot ## (B=1, C=classes-1, D, H, W)
        intersection = torch.sum(seg_label_probs * seg_label_mask, dim=(2, 3, 4))  # 在空间维度 (D, H, W) 上求和
        denominator = torch.sum(seg_label_probs + seg_label_mask, dim=(2, 3, 4))  # 在空间维度 (D, H, W) 上求和

        ## 计算 dice
        dice = (2. * intersection + self.smooth)/(denominator + self.smooth)
        # print("dice:", dice)
        dice = dice.squeeze(0)
        weight = torch.tensor([0.02,1,1,1]).to(dice.device)
        dice  = dice*weight
        dice = 1-torch.mean(dice)
        # print(dice)
        return dice

    
def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape).to(input.device)
    # print("Result device:", result.device)
    # print("Input device:", input.device)

    result = result.scatter_(1, input, 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        self.weight = torch.tensor([2e-1,1,1,1])

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weight[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]
