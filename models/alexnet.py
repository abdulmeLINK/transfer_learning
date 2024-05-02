import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        print(x.size())  # Print the size of the feature map after the features layer
        x = self.avgpool(x)
        print(x.size())  # Print the size of the feature map after the avgpool layer
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    @staticmethod
    def replace_output_layer(model, num_classes):
        model.classifier[-1] = nn.Linear(4096, num_classes)
        return model

    @staticmethod
    def prune_fully_connected_layers(model, num_layers_to_prune):
        classifier = list(model.classifier.children())[:-num_layers_to_prune]
        model.classifier = nn.Sequential(*classifier)
        return model

    @staticmethod
    def integrate_layers(model, new_layers):
        model.classifier = nn.Sequential(*new_layers, *model.classifier.children())
        return model

    def fine_tune(self, freeze_features=True, unfreeze_classifier=True):
        if freeze_features:
            for param in self.features.parameters():
                param.requires_grad = False
        if unfreeze_classifier:
            for param in self.classifier.parameters():
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
