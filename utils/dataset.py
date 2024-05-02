from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

# Define a transform to resize images to a uniform size
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert NumPy array to PIL Image
    transforms.Resize((37, 50)),  # Resize images to a uniform size
    transforms.ToTensor()  # Convert images to tensors
])

def load_dataset(test_size=0.2, batch_size=32):
    # Load the LFW dataset
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    # Preprocess images to have the same dimensions
    X_resized = [transform(image) for image in lfw_people.images]
    X_resized = torch.stack(X_resized)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resized, lfw_people.target, test_size=test_size)

    # Create DataLoader objects
    train_dataset = list(zip(X_train, y_train))
    test_dataset = list(zip(X_test, y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
