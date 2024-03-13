import torch
import torch.nn as nn
from torchvision import models

# Modify the VGG model for binary classification
class VGG_Binary(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG_Binary, self).__init__()
        # Load the pre-trained VGG16 model
        self.features = models.vgg16(pretrained=pretrained).features
        # Adjust the classifier part
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(True),
            nn.BatchNorm1d(4096),  # Batch normalization
            nn.Dropout(0.8),  # Increase dropout rate
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.BatchNorm1d(4096),  # Batch normalization
            nn.Dropout(0.8),  # Increase dropout rate
            nn.Linear(4096, 1),  # Output layer for binary classification
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



