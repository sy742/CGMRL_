import torch.nn as nn
import torch.nn.functional as F
class Distillation_Loss(nn.Module):
    def __init__(self, T, alpha):
        super(Distillation_Loss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.T = T
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
    def forward(self, input, knowledge, target):
        dis_loss = nn.KLDivLoss(reduction='none')(nn.functional.log_softmax(input/self.T, dim=1),nn.functional.softmax(knowledge / self.T, dim=1)).sum(1).mean()*0.1
        ce_loss = self.ce(input, target)
        return dis_loss, ce_loss
