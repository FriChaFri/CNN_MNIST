# Load in the data from pyTorch

import torch
import torchvision
from constants import BATCH_SIZE

def load_data(BATCH_SIZE=BATCH_SIZE):
    normalise_data = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_dataset = torchvision.datasets.MNIST(
        root="data/MNIST",
        train=True,
        download=True,
        transform=normalise_data,
    )
    test_dataset = torchvision.datasets.MNIST(
        root="data/MNIST",
        train=False,
        download=True,
        transform=normalise_data,
    )
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    return trainloader, testloader


def show_batch(loader):
    import matplotlib.pyplot as plt
    images, labels = next(iter(loader))
    images = images.numpy()
    
    fig, axs = plt.subplots(1, 8, figsize=(12, 2))
    for i in range(8):
        axs[i].imshow(images[i][0], cmap="gray")
        axs[i].set_title(f"Label: {labels[i].item()}")
        axs[i].axis('off')
    plt.savefig("fig/sample_input_batch.png")


if __name__ == "__main__":
    trainloader, testloader = load_data()
    print("Loaded data successfully")
    show_batch(trainloader)
