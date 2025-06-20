# Custom loss functions for the model 
import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    """Combined loss function for both regression and classification"""
    def __init__(self, alpha=0.7, beta=0.3):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    def forward(self, reg_pred, cls_pred, reg_target, cls_target):
        regression_loss = self.mse_loss(reg_pred, reg_target)
        classification_loss = self.ce_loss(cls_pred, cls_target)
        total_loss = self.alpha * regression_loss + self.beta * classification_loss
        return total_loss, regression_loss, classification_loss 