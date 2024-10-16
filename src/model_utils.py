from tensorflow.keras.models import load_model

def save_model(model, model_path):
    """Saves the Keras model to the specified path."""
    model.save(model_path)

def load_trained_model(model_path):
    """Loads the trained model from the specified path."""
    return load_model(model_path)