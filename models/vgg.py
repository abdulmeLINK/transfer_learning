import torch
import torch.nn as nn
import torchvision.models as models

class VGG(nn.Module):
    def __init__(self, num_classes):
        super(VGG, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def replace_output_layer(model, num_classes):
        model.model.classifier[6] = nn.Linear(4096, num_classes)
        return model


    @staticmethod
    def prune_fully_connected_layers(model, num_layers_to_prune):
        # VGG does not have fully connected layers to prune
        return model

    @staticmethod
    def integrate_layers(model, new_layers):
        # VGG does not have fully connected layers to integrate
        return model

    def fine_tune(self, freeze_features=True, unfreeze_classifier=True):
        if freeze_features:
            for param in self.model.parameters():
                param.requires_grad = False
        if unfreeze_classifier:
            for param in self.model.classifier[6].parameters():
                param.requires_grad = True
        return self

    def train_from_scratch(self):
        for param in self.parameters():
            param.requires_grad = True
        return self

    def load_weights(self, weight_file):
        self.load_state_dict(torch.load(weight_file))
        return self

    def save_weights(self, weight_file):
        torch.save(self.state_dict(), weight_file)
