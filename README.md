# Project Name

This project is a classification task on the LFW dataset using AlexNet, ResNet, and VGG-16 models.

## Project Structure

The project has the following structure:

```
my-project
├── main.py
├── models
│   ├── alexnet.py
│   ├── resnet.py
│   └── vgg.py
├── utils
│   └── dataset.py
├── weights
│   ├── AlexNet_Scenario1.pt
│   ├── AlexNet_Scenario2.pt
│   ├── AlexNet_Scenario3.pt
│   ├── AlexNet_Scenario4.pt
│   ├── ResNet_Scenario1.pt
│   ├── ResNet_Scenario2.pt
│   ├── ResNet_Scenario3.pt
│   ├── ResNet_Scenario4.pt
│   ├── VGG_Scenario1.pt
│   ├── VGG_Scenario2.pt
│   ├── VGG_Scenario3.pt
│   └── VGG_Scenario4.pt
└── README.md
```

## Files Description

- `main.py`: This file is the main entry point of the project. It contains the code for training and testing the classification models (AlexNet, ResNet, VGG-16) on the LFW dataset. It also includes the logic for performing feature extraction and fine-tuning on the models.

- `models/alexnet.py`: This file contains the implementation of the AlexNet classification model. It includes the architecture of the model and methods for replacing the output layer, pruning fully connected layers, and learning weight parameters.

- `models/resnet.py`: This file contains the implementation of the ResNet classification model. It includes the architecture of the model and methods for replacing the output layer, pruning fully connected layers, and learning weight parameters.

- `models/vgg.py`: This file contains the implementation of the VGG-16 classification model. It includes the architecture of the model and methods for replacing the output layer, pruning fully connected layers, and learning weight parameters.

- `utils/dataset.py`: This file contains utility functions for handling the LFW dataset. It includes methods for splitting the dataset into training and testing sets, as well as performing stratified splitting based on the labels.

- `weights/`: This directory contains the weight parameters for the different scenarios of each model. The weight files are named according to the model and scenario they belong to.

## Usage

To run the project, follow these steps:

1. Install the required dependencies.
2. Prepare the LFW dataset.
3. Run the `main.py` script with the desired model and scenario.

For example, to run the ResNet architecture on the first scenario, use the following command:

```
python main.py --model ResNet --scenario 1
```

To bypass the training phase and load pre-trained weights, use the `--bypass_train` option:

```
python main.py --model VGG --scenario 4 --bypass_train
```

## License

This project is licensed under the [MIT License](LICENSE).
```