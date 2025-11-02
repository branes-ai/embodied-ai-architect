"""Create a ResNet18 model for CLI testing."""

import torch
import torchvision.models as models

# Create ResNet18
model = models.resnet18(weights=None)
model.eval()

# Save model
torch.save(model, "resnet18.pt")
print("✓ ResNet18 model saved to resnet18.pt")

# Print model info
total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,}")
