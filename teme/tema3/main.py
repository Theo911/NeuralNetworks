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

def sigmoid(z):
    z = np.clip(z, -4, 4)
    return 1 / (1 + np.exp(-z))

def forward_propagation(X_batch, W, b):
    """Performs a forward pass through the network."""
    layer_outputs = [X_batch]
    output = X_batch

    # Apply sigmoid for hidden layers
    for i in range(len(W) - 1):
        z = np.dot(output, W[i]) + b[i]
        output = sigmoid(z)
        layer_outputs.append(output)

    # Apply softmax for the output layer
    z = np.dot(output, W[-1]) + b[-1]
    output = softmax(z)
    layer_outputs.append(output)

    return output, layer_outputs


def sigmoid_derivative(z):
    sig = sigmoid(z)
    return sig * (1 - sig)


def backpropagate(X_batch, Y_batch, y_pred, W, b, layer_outputs):
    dW = [np.zeros_like(w) for w in W]
    db = [np.zeros_like(bi) for bi in b]

    # Output layer error (softmax + cross-entropy)
    error = y_pred - Y_batch
    dW[-1] = np.dot(layer_outputs[-2].T, error)
    db[-1] = np.sum(error, axis=0)

    # Backpropagate through hidden layers
    for i in range(len(W) - 2, -1, -1):
        error = np.dot(error, W[i + 1].T) * sigmoid_derivative(layer_outputs[i + 1])
        dW[i] = np.dot(layer_outputs[i].T, error)
        db[i] = np.sum(error, axis=0)

    return dW, db


def update_weights_and_bias(layer_outputs, y_true, y_pred, W, b, learning_rate):
    # do not use back propagation

    error = y_true - y_pred
    # Update weights for the output layer
    W[-1] += learning_rate * np.dot(layer_outputs[-2].T, error)  # Use the output from the last hidden layer
    # Update biases for the output layer
    b[-1] += learning_rate * np.sum(error, axis=0)  # Sum across the batch

    return W, b


def train_network(train_X, train_Y, layer_sizes, num_epochs, batch_size, learning_rate):
    W, b = initialize_parameters(layer_sizes)

    for epoch in range(num_epochs):
        for X_batch, Y_batch in create_batches(train_X, train_Y, batch_size):
            y_pred, layer_outputs = forward_propagation(X_batch, W, b)      # Forward pass

            W, b = update_weights_and_bias(layer_outputs, Y_batch, y_pred, W, b, learning_rate)

            # dW, db = backpropagate(X_batch, Y_batch, y_pred, W, b, layer_outputs)       # Backpropagation
            #
            # for i in range(len(W)):     # Update weights and biases
            #     W[i] -= learning_rate * dW[i]
            #     b[i] -= learning_rate * db[i]

        print(f"Epoch {epoch + 1} completed.")

    return W, b


def compute_accuracy(X_test, Y_test, weights, biases):
    """Compute accuracy on the test set."""

    # Forward propagation through the network
    a = X_test  # Start with input data
    for W, b in zip(weights, biases):  # Iterate through each layer
        z = np.dot(a, W) + b  # Compute the linear combination
        if W.shape[1] == 10:  # Last layer (output layer uses softmax)
            a = softmax(z)
        else:  # Hidden layers use sigmoid activation
            a = sigmoid(z)

    # Convert predicted probabilities to class labels (argmax to get class index)
    predicted_labels = np.argmax(a, axis=1)

    # Convert true labels (one-hot encoded) to class labels
    true_labels = np.argmax(Y_test, axis=1)

    # Calculate accuracy (percentage of correct predictions)
    accuracy = np.mean(predicted_labels == true_labels) * 100

    return accuracy


def initialize_parameters(layer_sizes, scale=0.01):
    """
    Initializes weights and biases.

    Parameters:
    - layer_sizes: List of integers, where each element represents the number of units in a layer.
                   For example, [784, 128, 64, 10] represents a network with input size 784,
                   two hidden layers with sizes 128 and 64, and output size 10.
    - scale: Float, scaling factor for random weight initialization (default is 0.01).

    Returns:
    - weights: List of numpy arrays representing weights for each layer.
    - biases: List of numpy arrays representing biases for each layer.
    """

    weights = []
    biases = []

    for i in range(1, len(layer_sizes)):
        weights.append(np.random.randn(layer_sizes[i - 1], layer_sizes[i]) * scale)
        biases.append(np.zeros((1, layer_sizes[i])))

    return weights, biases


def main():
    train_X, train_Y = download_mnist(True)
    test_X, test_Y = download_mnist(False)

    train_X, test_X = normalize_data(train_X, test_X)

    train_Y = one_hot_encode(labels=train_Y)
    test_Y = one_hot_encode(labels=test_Y)

    print(f"Train data shape: {train_X.shape}, Train labels shape: {train_Y.shape}")
    print(f"Test data shape: {test_X.shape}, Test labels shape: {test_Y.shape}")

    learning_rate = 0.001
    num_epochs = 50
    batch_size = 100
    layer_sizes = [784, 32, 16, 10]

    W, b = train_network(train_X=train_X, train_Y=train_Y, layer_sizes=layer_sizes,
                         num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)

    for i in range(len(W)):
        print(f"Weight shape: {W[i].shape}, Bias shape: {b[i].shape}")

    test_accuracy = compute_accuracy(test_X, test_Y, W, b)
    print(f"Test Accuracy: {test_accuracy:.2f}%")



if __name__ == "__main__":
    main()