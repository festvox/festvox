import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F


# Custom Loss Function: https://discuss.pytorch.org/t/write-custom-loss-function/3369/2

class SelectiveCrossEntropyLoss(nn.CrossEntropyLoss):


     def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(SelectiveCrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

     # What does this mean? @weak_script_method
     def forward(self,input, target):

        # Ensure that things are 1d  
        assert len(input.shape) == 1

        # Get top k from the input

        # Define weights to multiply the input

        # Return the loss value
        return F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction) 


class SelectiveL1Loss(nn.L1Loss):


     def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(SelectiveL1Loss, self).__init__(size_average, reduce, reduction)

     # What does this mean? @weak_script_method
     def forward(self,input, target):

        loss =  F.l1_loss(input, target, reduction='none')
        loss = loss.view(-1)
        sorted_lengths, indices = torch.sort(loss, dim=0, descending=True)
        loss = loss[indices]
        altered_loss = torch.Tensor([loss[k] if k < int(0.8 * len(loss)) else 0 for k in range(len(loss))])
        print("Altered loss: ", altered_loss)
        return torch.mean(altered_loss, dim=0).cuda()

class SelectiveL1LossTopK(nn.L1Loss):


     def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(SelectiveL1Loss, self).__init__(size_average, reduce, reduction)

     # What does this mean? @weak_script_method
     def forward(self,input, target):

        loss =  F.l1_loss(input, target, reduction='none')
        loss = loss.view(-1)
        max_len = int(0.8 * len(loss))
        top_maxlen = torch.topk(loss, k=max_len, dim=0)[0]
        altered_loss = torch.mean(top_maxlen, dim=0)
        print("Altered loss: ", altered_loss)
        return altered_loss





