from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

# Define a transform to resize images to a uniform size
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert NumPy array to PIL Image
    transforms.Grayscale(num_output_channels=3),  # Convert images to RGB if they're grayscale
    transforms.Resize((37, 50)),  # Resize images to a uniform size
    transforms.ToTensor()  # Convert images to tensors
])

def adjust_labels(labels):
    # Ensure that the labels start from 0
    min_label = min(labels)
    if min_label > 0:
        labels -= min_label
    return labels


def load_dataset(test_size=0.2, batch_size=32):
    # Load the LFW dataset
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    # Preprocess images to have the same dimensions
    X_resized = [transform(image) for image in lfw_people.images]
    X_resized = torch.stack(X_resized)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resized, lfw_people.target, test_size=test_size)

    # Adjust the labels in the train and test sets
    y_train = adjust_labels(y_train)
    y_test = adjust_labels(y_test)

    # Get the number of classes
    num_classes = len(set(lfw_people.target))

    # Create DataLoader objects
    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, num_classes
