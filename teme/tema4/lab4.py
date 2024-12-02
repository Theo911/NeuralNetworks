import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pandas as pd

# Configurații
batch_size = 64
learning_rate = 0.01
num_epochs = 10

# Preprocesarea datelor
train_transforms = ToTensor()
test_transforms = ToTensor()

train_dataset = MNIST('data', train=True, transform=train_transforms, download=True)
test_dataset = MNIST('data', train=False, transform=test_transforms, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Model simplu fără straturi convoluționale
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)


# Inițializarea modelului
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Funcția de antrenare
def train_model(model, loader, optimizer, criterion):
    model.train()
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


# Funcția de evaluare
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            predictions.extend(predicted.cpu().numpy())
    accuracy = correct / total
    return accuracy, predictions


# Loop-ul de antrenare și evaluare
if __name__ == "__main__":
    for epoch in range(num_epochs):
        train_model(model, train_loader, optimizer, criterion)
        accuracy, _ = evaluate_model(model, test_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {accuracy:.4f}")

    # Generarea fișierului de predicții pentru Kaggle
    _, predictions = evaluate_model(model, test_loader)
    submission = pd.DataFrame({
        "ID": range(len(predictions)),
        "target": predictions
    })
    submission.to_csv("submission.csv", index=False)
