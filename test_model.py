"""Create a simple test model for CLI testing."""

import torch
import torch.nn as nn


class SimpleTestModel(nn.Module):
    """Simple CNN for testing."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


if __name__ == "__main__":
    model = SimpleTestModel()
    model.eval()

    # Save model
    torch.save(model, "test_model.pt")
    print("✓ Test model saved to test_model.pt")

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
