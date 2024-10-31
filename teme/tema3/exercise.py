import numpy as np
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten(),
                    download=True,
                    train=is_train)

    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)

    return mnist_data, mnist_labels


train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

fig, ax = plt.subplots(2, 2)

for i in range(4):
    img = train_X[i].reshape(28, 28)
    label = train_Y[i]
    ax[i // 2, i % 2].imshow(img, cmap='gray')
    ax[i // 2, i % 2].set_title(f'Label: {label}')


fig.tight_layout()
plt.show()