
# Neural Network for Wine Quality Classification

This project implements a simple feed-forward neural network to classify wine quality using the [Wine Quality Dataset](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009). The dataset provides various chemical properties of red wine, and the neural network is designed to predict whether a wine is "good" or "bad" based on these properties.

## Project Overview
This project demonstrates a basic implementation of a neural network in Python using only `numpy`. The goal is to perform binary classification on the wine quality dataset, where wines with a quality score of 6 or higher are labeled as "good" and others as "bad."

### Features
- Binary classification model with one hidden layer
- Customizable neural network parameters (e.g., learning rate, number of neurons in the hidden layer)
- Preprocessing of data, including label binarization and feature scaling

## Dataset
The dataset used is the **Wine Quality Dataset** from Kaggle, which can be downloaded [here](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009).

- **Input Features**: Chemical properties such as acidity, chlorides, sulfur dioxide, etc.
- **Target Label**: Wine quality score (binarized into "good" vs. "bad")

## Installation
Clone the repository and install the required packages:

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```

## Usage
To run the neural network, follow these steps:

1. **Download the dataset** and place `winequality-red.csv` in the project directory.
2. **Run the script**:
    ```bash
    python main.py
    ```

The model will preprocess the data, train on it, and display the test accuracy.

### Example Commands
```bash
# Run the neural network training
python main.py
```

## Code Structure
- `main.py`: The main script that loads the dataset, trains the neural network, and evaluates it on the test data.
- `network.py`: Defines the `NeuralNetwork` class, which includes methods for forward and backward propagation, training, and prediction.
- `data_loader.py`: Contains functions for loading and preprocessing the dataset.
- `README.md`: Provides an overview and instructions for the project.

## Neural Network Architecture
- **Input Layer**: 11 features representing chemical properties of the wine
- **Hidden Layer**: 5 neurons with a sigmoid activation function
- **Output Layer**: 1 neuron with a sigmoid activation function for binary classification

### Training Parameters
- **Learning Rate**: 0.01
- **Epochs**: 1000

## Results
After training, the model achieved an accuracy of approximately 76% on the test set, classifying wines as "good" or "bad" based on their quality.

## License
This project is for my EECE490 Course

## Acknowledgments
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) for providing the Wine Quality Dataset
- [Kaggle](https://www.kaggle.com/) for hosting datasets for open access
