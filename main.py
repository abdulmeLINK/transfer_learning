import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.dataset import load_dataset
from models.alexnet import AlexNet
from models.resnet import ResNet
from models.vgg import VGG

# Apply scnerios for each network
def modify_model(model, num_classes, scenario):
    if scenario == 1:
        model = model.replace_output_layer(num_classes)
    elif scenario == 2:
        model = model.replace_output_layer(num_classes).fine_tune()
    elif scenario == 3:
        model = model.fine_tune()
    elif scenario == 4:
        pass
    return model

def train_model(model, train_loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

def test_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy}%')

def save_model(model, model_name, scenario):
    path = f"weights/{model_name}_Scenario{scenario}.pt"
    torch.save(model.state_dict(), path)
    print(f"Model saved successfully at {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning on LFW dataset')
    parser.add_argument('--model', default='AlexNet', choices=['AlexNet', 'ResNet', 'VGG'], help='Choose model architecture')
    parser.add_argument('--scenario', default=1, type=int, choices=[1, 2, 3, 4], help='Choose scenario')
    parser.add_argument('--bypass_train', action='store_true', help='Bypass the training phase')
    args = parser.parse_args()

    train_loader, test_loader, num_classes = load_dataset()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if args.model == "AlexNet":
        model = AlexNet(num_classes)
    elif args.model == "ResNet":
        model = ResNet(num_classes)
    elif args.model == "VGG":
        model = VGG(num_classes)

    model = modify_model(model, num_classes, args.scenario)

    if not args.bypass_train:
        train_model(model, train_loader)
        test_model(model, test_loader)
        save_model(model, args.model, args.scenario)
    else:
        # Load pre-trained weights
        model.load_weights(f"weights/{args.model}_Scenario{args.scenario}.pt")
        test_model(model, test_loader)
