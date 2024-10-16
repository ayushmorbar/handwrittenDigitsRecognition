import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, x_test, y_test, model_type):
    """Evaluates the model and plots a confusion matrix."""
    # Predict the test set results
    y_pred = model.predict(x_test)

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, model_type)

def plot_confusion_matrix(y_true, y_pred, model_type):
    """Plots confusion matrix."""
    conf_matrix = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
    plt.title(f'Confusion Matrix - {model_type}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'results/confusion_matrix_{model_type}.png')
    plt.show()

if __name__ == "__main__":
    from src.train import train_model
    from src.data_loader import load_mnist_data
    
    # Load the data
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # Load trained model and evaluate
    model_basic = train_model('basic_nn', epochs=10)
    evaluate_model(model_basic, x_test, y_test, model_type='basic_nn')

    model_cnn = train_model('cnn', epochs=10)
    evaluate_model(model_cnn, x_test, y_test, model_type='cnn')
