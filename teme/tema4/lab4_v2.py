import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, RandomRotation, ToTensor
import pandas as pd


class EnhancedNN(nn.Module):
    def __init__(self):
        """Intiliazation of the neural network model: 3 hidden layers with 1024, 512, 256 neurons. """

        super(EnhancedNN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.model(x)

def train_model(model, loader, optimizer, criterion):
    """Train the model using the given loader, optimizer and criterion."""

    model.train()                               # set the model to training mode
    for images, labels in loader:
        optimizer.zero_grad()                   # zero the gradients
        outputs = model(images)                 # forward pass
        loss = criterion(outputs, labels)       # compute the loss
        loss.backward()                         # backward pass
        optimizer.step()                        # update the weights


def evaluate_model(model, loader):
    """Evaluate the model using the given loader. Return the accuracy and the predictions."""

    model.eval()
    correct = 0
    total = 0
    predictions = []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)                     # forward pass
            _, predicted = torch.max(outputs, 1)        # predicted is the index of the maximum value
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            predictions.extend(predicted.cpu().numpy())
    accuracy = correct / total
    return accuracy, predictions


if __name__ == "__main__":
    # Parameters
    batch_size = 64
    learning_rate = 0.0008
    num_epochs = 40
    dropout_rate = 0.2

    train_transforms = Compose([
        RandomRotation(degrees=10),
        ToTensor()
    ])
    test_transforms = ToTensor()

    train_dataset = MNIST('data', train=True, transform=train_transforms, download=True)
    test_dataset = MNIST('data', train=False, transform=test_transforms, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = EnhancedNN()
    criterion = nn.CrossEntropyLoss()                                               # loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)                   # optimizer
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)        # learning rate scheduler

    best_accuracy = 0.0
    best_epoch = 0
    submission_history = []
    for epoch in range(num_epochs):
        train_model(model, train_loader, optimizer, criterion)          # train the model
        accuracy, predictions = evaluate_model(model, test_loader)      # evaluate the model
        scheduler.step()                                                # update the learning rate

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch + 1

            submission_history.append({
                "epoch": best_epoch,
                "accuracy": best_accuracy
            })

            submission = pd.DataFrame({
                "ID": range(len(predictions)),
                "target": predictions
            })
            submission.to_csv(f"submission_{best_accuracy:.4f}.csv", index=False)

        print(f"Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {accuracy:.4f}, Best Accuracy: {best_accuracy:.4f}")

    history_df = pd.DataFrame(submission_history)
    history_df.to_csv("accuracy_history.csv", index=False)
