# @author: Marcio Lopes
import torch
import torch.nn as nn
from torchvision import models, transforms


class ResNet50Split(nn.Module):
    """
    A Split-Learning friendly wrapper for ResNet50.

    Architecture decomposed into 6 sequential blocks for slicing:
    - Block 0: Stem (Conv1, BN, ReLU, MaxPool)
    - Block 1: Layer 1 (Bottleneck Stack 1)
    - Block 2: Layer 2 (Bottleneck Stack 2)
    - Block 3: Layer 3 (Bottleneck Stack 3)
    - Block 4: Layer 4 (Bottleneck Stack 4)
    - Block 5: Head (AvgPool, Flatten, FC)
    """

    def __init__(self, num_classes=10, pretrained=True):
        super(ResNet50Split, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.base_model = models.resnet50(weights=weights)

        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

        self.layers = nn.ModuleList()

        # [Index 0] Stem
        self.layers.append(nn.Sequential(
            self.base_model.conv1,
            self.base_model.bn1,
            self.base_model.relu,
            self.base_model.maxpool
        ))

        # [Index 1-4] Bottleneck Blocks
        self.layers.append(self.base_model.layer1)
        self.layers.append(self.base_model.layer2)
        self.layers.append(self.base_model.layer3)
        self.layers.append(self.base_model.layer4)

        # [Index 5] Classifier Head
        self.layers.append(nn.Sequential(
            self.base_model.avgpool,
            nn.Flatten(),
            self.base_model.fc
        ))

        # .to(device) is called by the Factory, not here

    def forward(self, x):
        """Full forward pass (for local testing only)."""
        for layer in self.layers:
            x = layer(x)
        return x

    def get_layer_count(self):
        return len(self.layers)