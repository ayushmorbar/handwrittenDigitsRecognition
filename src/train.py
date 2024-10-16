import tensorflow as tf
from src.data_loader import load_mnist_data
from src.basic_nn import build_basic_nn
from src.cnn import build_cnn

def train_model(model_type='basic_nn', epochs=5):
    """Trains a model based on the model type (basic_nn or cnn)."""
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # Choose model type
    if model_type == 'basic_nn':
        model = build_basic_nn(input_shape=(28, 28))
    elif model_type == 'cnn':
        model = build_cnn(input_shape=(28, 28, 1))
    else:
        raise ValueError("Model type not supported. Use 'basic_nn' or 'cnn'.")

    # Train the model
    model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

    # Save the trained model
    model.save(f'models/{model_type}_model.h5')

    return model

if __name__ == "__main__":
    # Train the basic NN model
    train_model('basic_nn', epochs=10)

    # Train the CNN model
    train_model('cnn', epochs=10)
