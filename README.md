
# Transfer Learning

This project is a classification task on the LFW (Labeled Faces in the Wild) dataset using three different pre-trained models: AlexNet, ResNet, and VGG-16. The goal is to leverage these pre-trained models and apply transfer learning techniques to perform face recognition tasks.

```
├── dataset/
├── main.py
├── models
│   ├── alexnet.py
│   ├── resnet.py
│   └── vgg.py
├── README.md
├── tests.ipynb
├── tests.sh
├── requirements.txt
├── utils
│   └── dataset.py
└── weights
    ├── AlexNet_Scenario1.pt
    ├── AlexNet_Scenario2.pt
    ├── AlexNet_Scenario3.pt
    ├── AlexNet_Scenario4.pt
    ├── ResNet_Scenario1.pt
    ├── ResNet_Scenario2.pt
    ├── ResNet_Scenario3.pt
    ├── ResNet_Scenario4.pt
    ├── VGG_Scenario1.pt
    ├── VGG_Scenario2.pt
    ├── VGG_Scenario3.pt
    └── VGG_Scenario4.pt
```


## Files Description

- `main.py`: This file is the main entry point of the project. It contains the code for training and testing the classification models (AlexNet, ResNet, VGG-16) on the LFW dataset. It also includes the logic for performing feature extraction and fine-tuning on the models.

- `models/alexnet.py`, `models/resnet.py`, `models/vgg.py`: These files contain the implementation of the respective classification models. They include the architecture of the model and methods for replacing the output layer, pruning fully connected layers, and learning weight parameters.

- `utils/dataset.py`: This file contains utility functions for handling the LFW dataset. It includes methods for splitting the dataset into training and testing sets, as well as performing stratified splitting based on the labels.

- `weights/`: This directory contains the weight parameters for the different scenarios of each model. The weight files are named according to the model and scenario they belong to.

- `dataset/`: This directory contains the LFW dataset used for training and testing the models.

- `tests.ipynb`: This Jupyter notebook contains the code for running tests and visualizing the results in a more comprehensive manner.

- `tests.sh`: This shell script is used to run the tests in a batch mode.
- `requirements.txt`: This file lists the Python packages that your project depends on. Each line of the file specifies a package and its version number. You can use this file to install the same packages in another environment with the command `pip install -r requirements.txt`. This ensures that your project's dependencies are consistent across different environments, which helps prevent bugs and compatibility issues.

## Usage

To run the project, follow these steps:

1. Install the required dependencies. You can find the list of dependencies in the `requirements.txt` file. Use the command `pip install -r requirements.txt` to install them.

2. Prepare the LFW dataset. Download the dataset and place it in a directory named `data/` in the root of the project.

3. Run the `main.py` script with the desired model and scenario. The `--model` option specifies the model to use (AlexNet, ResNet, or VGG), and the `--scenario` option specifies the scenario to run (1, 2, 3, or 4).

For example, to run the ResNet architecture on the first scenario, use the following command:

```
python main.py --model ResNet --scenario 1
```

To bypass the training phase and load pre-trained weights, use the `--bypass_train` option:

```
python main.py --model VGG --scenario 4 --bypass_train
```

## Training Time

Running all trainings together on a T4 GPU on Google Colab takes approximately 250 seconds.

## Results

The models' performance varies depending on the scenario. Generally, the models perform in the following order in terms of accuracy: ResNet > VGG > AlexNet.

For a more detailed view of the results, including accuracy metrics, loss graphs, and confusion matrices for each model and scenario, please refer to the `tests.ipynb` Jupyter notebook. This notebook contains the code for running the tests and visualizing the results in a more comprehensive manner.

## License

This project is licensed under the [MIT License](LICENSE).
