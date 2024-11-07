import numpy as np
from load_mnist_dataset import download_mnist
from math import sqrt


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

def relu(z):
    return np.maximum(0, z)

def apply_dropout(output, dropout_rate):
    """Apply dropout to the output of a layer."""

    if not (0 <= dropout_rate <= 1):
        raise ValueError("dropout_rate must be between 0 and 1")

    num_neurons = len(output)
    num_to_drop = int(np.floor(num_neurons * dropout_rate))     # Number of neurons to drop

    indices_to_drop = np.random.choice(num_neurons, num_to_drop, replace=False)     # Randomly select neurons to drop

    # Create a copy of the output and set the selected indices to zero
    output_with_dropout = output.copy()
    output_with_dropout[indices_to_drop] = 0

    return output_with_dropout

def forward_propagation(X_batch, W, b, hidden_activation, output_activation, dropout_rate, training=True):
    """Performs a forward pass through the network."""
    layer_outputs = [X_batch]
    output = X_batch

    for i in range(len(W) - 1):
        z = np.dot(output, W[i]) + b[i]
        output = hidden_activation(z)

        if training:
            output = apply_dropout(output, dropout_rate)    # Apply dropout only during training

        layer_outputs.append(output)

    z = np.dot(output, W[-1]) + b[-1]
    output = output_activation(z)     # softmax for the output layer
    layer_outputs.append(output)

    return output, layer_outputs

def sigmoid_derivative(z):
    sig = sigmoid(z)
    return sig * (1 - sig)

def relu_derivative(output):
    return (output > 0).astype(float)

def cross_entropy_loss_derivative(y_pred, y_true):
    return y_pred - y_true

def backpropagate(Y_target, y_pred, W, b, layer_outputs, loss_derivative, hidden_activation_derivative):
    dW = [np.zeros_like(w) for w in W]
    db = [np.zeros_like(bi) for bi in b]

    # Output layer error (softmax + cross-entropy)
    error = loss_derivative(y_pred, Y_target)           # δ = y - y_target
    dW[-1] = np.dot(layer_outputs[-2].T, error)         # 𝜕C/𝜕W = y * δ
    db[-1] = np.sum(error, axis=0)                      # 𝜕C/𝜕b = δ

    # Backpropagate through hidden layers
    for i in range(len(W) - 2, -1, -1):
        error = np.dot(error, W[i + 1].T) * hidden_activation_derivative(layer_outputs[i + 1])      # δ = δ * W * σ'(z)
        dW[i] = np.dot(layer_outputs[i].T, error)                                                   # 𝜕C/𝜕W = y * δ
        db[i] = np.sum(error, axis=0)                                                               # 𝜕C/𝜕b = δ

    return dW, db


def update_weights_and_bias(layer_outputs, y_true, y_pred, W, b, learning_rate, loss_derivative):
    """Used just for forward pass when not using backpropagation."""

    error = loss_derivative(y_true, y_pred)
    W[-1] += learning_rate * np.dot(layer_outputs[-2].T, error)  # Use the output from the last hidden layer
    b[-1] += learning_rate * np.sum(error, axis=0)  # Sum across the batch

    return W, b

def shuffle_data(X, Y):
    """Shuffle the dataset while keeping features and labels aligned."""
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], Y[indices]


def compute_loss(X, Y, W, b):
    """Calculate the mean squared error (MSE) loss over the entire dataset."""
    # Perform forward propagation to get predictions
    y_pred, _ = forward_propagation(X, W, b, hidden_activation=sigmoid, output_activation=softmax, dropout_rate=0, training=False)

    # Compute the loss
    mse_loss = np.mean((y_pred - Y) ** 2)
    return mse_loss


def train_network(train_X, train_Y, test_X, test_Y, layer_sizes, num_epochs, batch_size, initial_lr, patience, decay_factor,
                  hidden_activation, output_activation, loss_derivative, hidden_activation_derivative, dropout_rate=0.5):
    W, b = initialize_parameters(layer_sizes)
    learning_rate = initial_lr
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        train_X, train_Y = shuffle_data(train_X, train_Y)

        for X_batch, Y_batch in create_batches(train_X, train_Y, batch_size):
            y_pred, layer_outputs = forward_propagation(X_batch, W, b, hidden_activation, output_activation, dropout_rate=dropout_rate, training=True)

            # W, b = update_weights_and_bias(layer_outputs, Y_batch, y_pred, W, b, learning_rate, loss_derivative)     # if only forward pass

            dW, db = backpropagate(Y_target=Y_batch, y_pred=y_pred, W=W, b=b, layer_outputs=layer_outputs,
                                   loss_derivative=loss_derivative,
                                   hidden_activation_derivative=hidden_activation_derivative)       # Backpropagation
            for i in range(len(W)):     # Update weights and biases
                W[i] -= learning_rate * dW[i]
                b[i] -= learning_rate * db[i]

        train_accuracy = compute_accuracy(train_X, train_Y, W, b)
        validation_accuracy = compute_accuracy(test_X, test_Y, W, b)
        print(f"Epoch {epoch + 1} completed. Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {validation_accuracy:.2f}%")
        #
        # train_loss = compute_loss(train_X, train_Y, W, b)
        # # Check if there's improvement in validation loss
        # if train_loss < best_loss:
        #     best_loss = train_loss
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #
        # # Adjust learning rate if metrics plateau
        # if patience_counter >= patience:
        #     learning_rate *= decay_factor
        #     patience_counter = 0  # Reset counter after decay
        #     print(f"Learning rate reduced to {learning_rate:.5f}")

    return W, b


def compute_accuracy(X_test, Y_test, weights, biases):
    """Compute accuracy on the test set."""

    # Forward propagation through the network
    a = X_test
    for W, b in zip(weights, biases):   # Iterate through the layers
        z = np.dot(a, W) + b
        if W.shape[1] == 10:  # last layer
            a = softmax(z)
        else:
            a = sigmoid(z)

    predicted_labels = np.argmax(a, axis=1)     # Predicted labels
    true_labels = np.argmax(Y_test, axis=1)     # Convert one-hot encoded labels to class labels

    accuracy = np.mean(predicted_labels == true_labels) * 100
    return accuracy


def initialize_parameters(layer_sizes):
    """
    Initializes weights and biases using Xavier initialization with a uniform distribution.

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
        # weights.append(np.random.randn(layer_sizes[i - 1], layer_sizes[i]) * scale)       # not so good

        weight_stddev = np.sqrt(1 / layer_sizes[i - 1])
        weights.append(np.random.randn(layer_sizes[i - 1], layer_sizes[i]) * weight_stddev)

        # limit = sqrt(1 / layer_sizes[i - 1])
        # weights.append(np.random.uniform(-limit, limit, (layer_sizes[i - 1], layer_sizes[i])))

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

    learning_rate = 0.0001
    num_epochs = 150
    batch_size = 64
    layer_sizes = [784, 100, 10]

    W, b = train_network(train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y, layer_sizes=layer_sizes,
                        num_epochs=num_epochs, batch_size=batch_size,
                        initial_lr=learning_rate, patience=3, decay_factor=0.8,
                        hidden_activation=sigmoid,
                        output_activation=softmax,
                        loss_derivative=cross_entropy_loss_derivative,
                        hidden_activation_derivative=sigmoid_derivative,
                        dropout_rate=0)

    # for i in range(len(W)):
    #     print(f"Weight shape: {W[i].shape}, Bias shape: {b[i].shape}")

    # test_accuracy = compute_accuracy(test_X, test_Y, W, b)
    # print(f"Test Accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
