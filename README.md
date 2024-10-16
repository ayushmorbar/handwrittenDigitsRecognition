# Handwritten Digit Recognition using Neural Networks

This project implements a basic neural network (NN) and a convolutional neural network (CNN) to recognize handwritten digits from the MNIST dataset. It demonstrates how deep learning models can classify images of digits (0-9).

## Technologies Used
- Python 3.8
- TensorFlow & Keras
- Numpy & Pandas
- Matplotlib & Seaborn
- Jupyter Notebooks

## Project Structure

The following is the structure of the Handwritten Digit Recognition project:

```
Handwritten-Digit-Recognition/
│
├── src/
│   ├── __init__.py                    # Makes src a package
│   ├── basic_nn.py                    # Basic Neural Network
│   ├── cnn.py                         # Convolutional Neural Network
│   ├── data_loader.py                 # Function to load and preprocess data
│   ├── model_utils.py                 # Utility functions for model handling
│   ├── train.py                       # Training script for both models
│   └── evaluation.py                  # Evaluation script for accuracy, confusion matrix, etc.
│
├── notebooks/
│   ├── Basic_NN_MNIST.ipynb            # Basic NN Jupyter Notebook
│   └── CNN_MNIST.ipynb                 # CNN Jupyter Notebook
│
├── models/                             # Directory to save trained models
│   ├── basic_nn_model.h5
│   └── cnn_model.h5
│
├── results/                            # Save model performance results
│   ├── confusion_matrix_basic_nn.png
│   └── confusion_matrix_cnn.png
│
├── tests/                              # Unit tests for utility and model functions
│   ├── test_data_loader.py
│   └── test_model_utils.py
│
├── README.md                           # Main project documentation
├── requirements.txt                    # Python dependencies
├── LICENSE                             # License file (optional, e.g., MIT)
└── .gitignore                          # Ignore unnecessary files like datasets
```


## How to Run

1. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

2. Train the model:
    ```sh
    python src/train.py
    ```

3. Evaluate and visualize results:
    ```sh
    python src/evaluation.py
    ```

## Results
Confusion matrices, accuracy plots, and trained models are saved in the `results/` and `models/` directories.

## Future Work
- Data augmentation for improving CNN performance.
- Experimenting with more advanced CNN architectures like ResNet.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.