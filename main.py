import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.dataset import load_dataset
from models.alexnet import AlexNet
from models.resnet import ResNet
from models.vgg import VGG

def modify_model(model, num_classes, scenario):
    if scenario == 1:
        # Objective: Adapt the model to classify categories in the new dataset.
        # Python Implementation: Create a new output layer that matches the number of categories in your new dataset and replace the existing output layer in the model.
        model = model.replace_output_layer(model, num_classes)
    elif scenario == 2:
        # Objective: Modify the model to better suit the new dataset.
        # Python Implementation: Prune the fully connected layers and add new layers. Train these new layers on the new dataset to learn the associated weights.
        model = model.replace_output_layer(model,num_classes).fine_tune()
    elif scenario == 3:
        # Objective: Further adapt the model to the new dataset.
        # Python Implementation: Disregard the weights in the later blocks of the CNN and the replaced fully connected layers. Train these layers on the new dataset to learn new weights.
        model = model.fine_tune(freeze_features=False, unfreeze_classifier=True)
    elif scenario == 4:
        # Objective: Improve the model’s performance on the new dataset.
        # Python Implementation: Train the entire network again on the new dataset, but this time, transfer only the architecture of the model, not the weights. This means learning all the weights from scratch based on the new dataset.
        model = model.fine_tune(freeze_features=False, unfreeze_classifier=False)
    return model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader):
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

def test_model(model, test_loader):
    model = model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
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
