import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.model(x)

    def replace_output_layer(self, num_classes):
        self.model.fc = nn.Linear(2048, num_classes)

    def fine_tune(self):
        # Unfreeze the parameters of the last layer for fine-tuning
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
