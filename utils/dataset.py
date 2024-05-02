from torchvision.datasets import LFW
from torch.utils.data import DataLoader, random_split

def load_dataset(dataset_path, train_size=0.8, download=True):
    # Load the LFW dataset
    dataset = LFW(root=dataset_path, download=download)

    # Split dataset into training and testing sets
    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    return train_set, test_set
