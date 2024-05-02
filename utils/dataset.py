from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

def load_dataset(test_size=0.2):
    # Load the LFW dataset
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(lfw_people.images, lfw_people.target, test_size=test_size)

    return (X_train, X_test), (y_train, y_test)
