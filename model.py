import torch.nn as nn


class DigitCNN(nn.Module):
    """Small CNN for MNIST. Input: [batch, 1, 28, 28]. Output: 10 logits, one per digit."""

    def __init__(self):
        super().__init__()

        # Convolution blocks detect visual patterns (edges, then shapes)
        self.features = nn.Sequential(
            # Block 1: 1 grayscale channel -> 16 feature maps
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # padding=1 keeps the size at 28x28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # halves the size: 28x28 -> 14x14

            # Block 2: 16 -> 32 feature maps
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # halves the size: 14x14 -> 7x7
        )

        # The classifier turns those patterns into one score per digit
        self.classifier = nn.Sequential(
            nn.Flatten(),  # [batch, 32, 7, 7] -> [batch, 32 * 7 * 7]
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),  # randomly drops 25% of values during training to reduce overfitting
            nn.Linear(128, 10),  # raw scores (logits); CrossEntropyLoss handles the softmax
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
