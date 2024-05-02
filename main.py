import argparse
import torch
import torch.nn as nn
from utils.dataset import load_dataset
from models.alexnet import AlexNet
from models.resnet import ResNet
from models.vgg import VGG



def modify_model(model, num_classes):
    # Try to access the 'fc' attribute
    try:
        # Modify the fully connected layer to have 'num_classes' output features
        setattr(model, 'fc', nn.Linear(model.fc.in_features, num_classes))
    except AttributeError:
        # Handle the case where the attribute 'fc' does not exist
        print("Warning: The model does not have the attribute 'fc'.")

    return model

def train_model(model, train_loader):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Train the model
    for epoch in range(10):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        for _, (inputs, labels) in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        # Print average loss per epoch
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    return model

def test_model(model, test_loader):
    # Test the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct / total}%')

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["AlexNet", "ResNet", "VGG"], default="AlexNet", help="Choose the model architecture")
    parser.add_argument("--scenario", choices=["1", "2", "3", "4"], default="1", help="Choose the scenario")
    parser.add_argument("--bypass_train", action="store_true", help="Bypass the training phase")
    args = parser.parse_args()

    # Load and split the dataset
    train_set, test_set, num_classes = load_dataset()
    
    if args.model == "AlexNet":
        model = AlexNet(num_classes)
    elif args.model == "ResNet":
        model = ResNet(num_classes)
    elif args.model == "VGG":
        model = VGG(num_classes)
    
    model = modify_model(model, num_classes)

    if not args.bypass_train:
        model = train_model(model, train_set)
    else:
        # Load pre-trained weights
        model.load_weights(f"weights/{args.model}_Scenario{args.scenario}.pt")

    # Perform testing
    test_model(model, test_set)
