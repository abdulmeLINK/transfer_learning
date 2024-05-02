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

def train_model(model, dataloader, epochs=10):
    # Logic for training the whole network
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    return model
