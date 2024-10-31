import numpy as np
from load_mnist_dataset import download_mnist


def load_mnist_dataset():
    """Load the MNIST dataset."""

    train_X, train_Y = download_mnist(is_train=True)
    test_X, test_Y = download_mnist(is_train=False)
    return train_X, train_Y, test_X, test_Y

def normalize_data(train_X, test_X):
    """Normalize the data by dividing it by 255 such that the values are in the range [0, 1]."""

    train_X = np.array(train_X) / 255
    test_X = np.array(test_X) / 255
    return train_X, test_X

def one_hot_encode(labels, num_classes=10):
    one_hot_labels = np.zeros((len(labels), num_classes))
    for idx, label in enumerate(labels):
        one_hot_labels[idx][label] = 1
    return one_hot_labels

def create_batches(X, Y, batch_size=100):
    """Split the training data and training labels into batches of 100 elements"""

    num_samples = X.shape[0]        # Number of samples in the dataset
    for i in range(0, num_samples, batch_size):
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]
        yield X_batch, Y_batch

def softmax(z):
    z_max = np.max(z, axis=1, keepdims=True)        # Maximum value in each row
    z_stable = z - z_max        # to avoid overflow

    exp_z = np.exp(z_stable)    # this ensures all values are positive

    sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)        # Sum of the exponentials

    probabilities = exp_z / sum_exp_z   # Probabilities of each class

    return probabilities

def forward_propagation(X_batch, W, b):
    """Forward pass of the neural network.
    Compute z by multiplying input by weights and adding bias and then apply softmax to get the probabilities of each class."""

    z = np.dot(X_batch, W) + b
    output = softmax(z)
    return output

def update_weights_and_bias(X_batch, y_true, y_pred, W, b, learning_rate):
    error = y_true - y_pred  # (Target - y) - how much we need to adjust the prediction to make it closer to the true class.

    # Update weights using the formula: W = W + μ * (Target - y) * X^T
    # X_batch.T: (784, 100), error: (100, 10), W: (784, 10)
    W += learning_rate * np.dot(X_batch.T, error)

    # Update biases using the formula: b = b + μ * (Target - y)
    b += learning_rate * np.sum(error, axis=0)  # Sum across the batch

    return W, b


def compute_accuracy(X_test, Y_test, W, b):
    """Compute accuracy on the test set."""

    y_pred = forward_propagation(X_test, W, b)
    predicted_labels = np.argmax(y_pred, axis=1)        # Predicted labels

    true_labels = np.argmax(Y_test, axis=1)     # True labels

    # Calculate accuracy (percentage of correct predictions)
    accuracy = np.mean(predicted_labels == true_labels) * 100

    return accuracy

def main():
    train_X, train_Y = download_mnist(True)
    test_X, test_Y = download_mnist(False)

    train_X, test_X = normalize_data(train_X, test_X)

    train_Y = one_hot_encode(train_Y)
    test_Y = one_hot_encode(test_Y)

    print(f"Train data shape: {train_X.shape}, Train labels shape: {train_Y.shape}")
    print(f"Test data shape: {test_X.shape}, Test labels shape: {test_Y.shape}")

    learning_rate = 0.01
    num_epochs = 50
    batch_size = 100

    W = np.random.randn(784, 10) * 0.01
    b = np.zeros(10)

    for epoch in range(num_epochs):
        for X_batch, Y_batch in create_batches(train_X, train_Y, batch_size):
            y_pred = forward_propagation(X_batch, W, b)

            # Update weights and biases using gradient descent
            W, b = update_weights_and_bias(X_batch, Y_batch, y_pred, W, b, learning_rate)

        print(f"Epoch {epoch + 1} completed.")

    test_accuracy = compute_accuracy(test_X, test_Y, W, b)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()
