# Age classification model definition 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class AgeClassifierCNN(nn.Module):
    """CNN-based age classifier with both regression and classification heads"""
    def __init__(self, num_classes=7, dropout_rate=0.5, use_pretrained=True):
        super(AgeClassifierCNN, self).__init__()
        if use_pretrained:
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.backbone.fc = nn.Identity()
            feature_dim = 512
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            feature_dim = 512
        self.feature_processor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )
        self.regression_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        self.classification_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
    def forward(self, x, mode='regression'):
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        processed_features = self.feature_processor(features)
        if mode == 'regression':
            return self.regression_head(processed_features).squeeze()
        elif mode == 'classification':
            return self.classification_head(processed_features)
        else:
            reg_out = self.regression_head(processed_features).squeeze()
            cls_out = self.classification_head(processed_features)
            return reg_out, cls_out 