import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        with torch.no_grad():
            for layer in model:
                x = layer(x)

                if isinstance(layer, nn.Linear):
                    mean = round(x.mean().item(), 4)
                    std = round(x.std().item(), 4)

                    # x shape: (batch_size, n_neurons)
                    dead_neurons = (x <= 0).all(dim=0)
                    dead_fraction = round(dead_neurons.float().mean().item(), 4)

                    stats.append({
                        "mean": mean,
                        "std": std,
                        "dead_fraction": dead_fraction
                    })
        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        stats = []
        for layer in model:
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad
                if grad is not None:
                    mean = round(grad.mean().item(), 4)
                    std = round(grad.std().item(), 4)
                    # Frobenius norm (default p=2) of the gradient matrix
                    norm = round(torch.norm(grad).item(), 4)
                    stats.append({"mean": mean, "std": std, "norm": norm})
                else:
                    # Fallback in case gradient is missing (should not happen)
                    stats.append({"mean": 0.0, "std": 0.0, "norm": 0.0})
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        # Typical thresholds:
        # - dead_neurons: any dead_fraction > 0.5 (more than half the neurons dead)
        # - exploding_gradients: any gradient norm > 10.0
        # - vanishing_gradients: any gradient norm < 1e-4
        for act in activation_stats:
            if act["dead_fraction"] > 0.5:
                return "dead_neurons"

        for grad in gradient_stats:
            if grad["norm"] > 10.0:
                return "exploding_gradients"

        for grad in gradient_stats:
            if grad["norm"] < 1e-4:
                return "vanishing_gradients"

        return "healthy"