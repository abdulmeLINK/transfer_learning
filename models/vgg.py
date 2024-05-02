import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

class VGG(nn.Module):
    def __init__(self, num_classes):
        super(VGG, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)

    def replace_output_layer(self, num_classes):
        self.model.classifier[6] = nn.Linear(4096, num_classes)
        return self.model

    def fine_tune(self):
        # Unfreeze the parameters of the last layer for fine-tuning
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier[6].parameters():
            param.requires_grad = True
        return self.model
