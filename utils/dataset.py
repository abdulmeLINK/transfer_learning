from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import numpy as np

def load_dataset(test_size=0.2):
    # Load the LFW dataset
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    # Preprocess images to have the same dimensions
    # Calculate the mean width and height
    mean_width = int(np.mean([image.shape[1] for image in lfw_people.images]))
    mean_height = int(np.mean([image.shape[0] for image in lfw_people.images]))
    
    # Resize images to the mean dimensions
    X_resized = [resize_image(image, mean_width, mean_height) for image in lfw_people.images]
    X_resized = np.array(X_resized)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resized, lfw_people.target, test_size=test_size)

    return (X_train, X_test), (y_train, y_test)

def resize_image(image, width, height):
    # Resize the image to the specified dimensions
    from skimage.transform import resize
    return resize(image, (height, width), mode='constant')