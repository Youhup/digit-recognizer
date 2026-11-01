import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from model import DigitCNN

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 5
WEIGHTS_PATH = "mnist_cnn.pth"

# Use the GPU if there is one, otherwise the CPU
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Training images get small random rotations and shifts, so the model tolerates imperfect drawings
train_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    v2.ToDtype(torch.float32, scale=True),  # converts pixels from 0-255 to 0-1
])

# Test images are not augmented, so we measure accuracy on the real digits
test_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

train_data = datasets.MNIST(root="data", train=True, download=True, transform=train_transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)  # new order each epoch
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

model = DigitCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_one_epoch():
    """Update the weights once per batch and return the average training loss."""
    model.train()  # turns on Dropout and lets BatchNorm update its statistics
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        loss = loss_fn(model(images), labels)

        optimizer.zero_grad()  # clear the gradients from the previous batch
        loss.backward()  # compute new gradients
        optimizer.step()  # update the weights

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate():
    """Return the accuracy on the test set."""
    model.eval()  # turns off Dropout and makes BatchNorm use its saved statistics
    correct = 0

    with torch.no_grad():  # no gradients needed, which saves time and memory
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)  # the digit with the highest score
            correct += (predictions == labels).sum().item()

    return correct / len(test_data)


for epoch in range(EPOCHS):
    train_loss = train_one_epoch()
    accuracy = evaluate()
    print(f"Epoch {epoch + 1}/{EPOCHS} | loss: {train_loss:.4f} | test accuracy: {accuracy:.2%}")

# Save only the learned weights, not the whole model object
torch.save(model.state_dict(), WEIGHTS_PATH)

print(f"\nFinal test accuracy: {accuracy:.2%}")
print(f"Weights saved to {WEIGHTS_PATH}")
