import torch
import torch.nn as nn
from typing import List

class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        dead_fractions = []
        hooks = []

        def hook_fn(module, input, output):
            # output shape: (batch_size, ...)
            # Flatten all dimensions except batch
            batch_size = output.shape[0]
            flat_output = output.view(batch_size, -1)
            # Check which neurons output 0 for all samples in the batch
            dead_neurons = (flat_output == 0).all(dim=0)  # shape: (num_neurons,)
            dead_fraction = dead_neurons.float().mean().item()
            dead_fractions.append(round(dead_fraction, 4))

        # Register hook on every ReLU layer
        for name, layer in model.named_modules():
            if isinstance(layer, nn.ReLU):
                hooks.append(layer.register_forward_hook(hook_fn))

        # Forward pass
        with torch.no_grad():
            _ = model(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        if any(df > 0.5 for df in dead_fractions):
            return 'use_leaky_relu'
        if dead_fractions and dead_fractions[0] > 0.3:
            return 'reinitialize'
        # Check strictly increasing
        strictly_increasing = all(dead_fractions[i] < dead_fractions[i+1] for i in range(len(dead_fractions)-1))
        if strictly_increasing and dead_fractions[-1] > 0.1:
            return 'reduce_learning_rate'
        if max(dead_fractions) < 0.1:
            return 'healthy'
        return 'healthy'