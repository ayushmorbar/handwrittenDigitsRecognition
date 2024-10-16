import tensorflow as tf
from tensorflow.keras.datasets import mnist

def load_mnist_data():
    """Loads the MNIST dataset and preprocesses the data for NN and CNN."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values to between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # One-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Reshape input for CNN model to include the channel dimension
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)