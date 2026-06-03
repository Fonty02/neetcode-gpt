import torch
import torch.nn as nn
from torchtyping import TensorType  # type: ignore

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        # Fixed the typo: Linear(512, 0) -> Linear(512, 10) for 10-class classification
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
            nn.Sigmoid()
        )

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)  # ensure reproducibility
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        output = self.model(images)
        # Round to 4 decimal places
        output_rounded = torch.round(output * 10000) / 10000
        return output_rounded