import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from model import ConvNet

device = torch.device('cude' if torch.cuda.is_available() else 'cpu')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# parameters
path = './cnn.path'
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# MNIST dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                             train=True,
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                            train=False,
                                            transform=transform)

# data loader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def img_show(imgs):
    imgs = imgs / 2 + 0.5
    npimgs = imgs.numpy()
    plt.imshow(np.transpose(npimgs, (1, 2, 0)))
    plt.show()


def view_imgs():
    datailer = iter(train_loader)
    images, labels = next(datailer)
    img_grid = torchvision.utils.make_grid(images[0:25], nrow=5)
    img_show(img_grid)


def train_model(epochs):
    model = ConvNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_steps = len(train_loader)
    for epoch in range(epochs):
        n_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_loss += loss.item()

        print(f'[{epoch + 1}] loss: {n_loss / n_steps:.3f}')
    print('Training Complete!')
    torch.save(model.state_dict(), path)


def load_model():
    model = ConvNet().to(device)
    loaded_model = ConvNet()
    loaded_model.load_state_dict(torch.load(path))
    loaded_model.to(device)
    loaded_model.eval()

    with torch.no_grad():
        n_correct1 = 0
        n_correct2 = 0
        n_samples = len(test_loader.dataset)

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            n_correct1 += (predicted == labels).sum().item()

            outputs2 = loaded_model(images)
            _, predicted2 = torch.max(outputs2, 1)
            n_correct2 += (predicted2 == labels).sum().item()

        acc = 100.0 * n_correct1/n_samples
        print(f'Accuracy of the model: {acc}%')

        acc = 100.0 * n_correct2/n_samples
        print(f'Accuracy of the loaded model: {acc}%')


if __name__ == '__main__':
    load_model()
