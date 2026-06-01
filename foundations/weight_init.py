import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std=math.sqrt(2/(fan_in+fan_out))
        tensor= torch.randn(fan_out,fan_in)*std
        nested_list = torch.round(tensor, decimals=4).tolist()
        return nested_list
        

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std=math.sqrt(2/fan_in)
        tensor= torch.randn(fan_out,fan_in)*std
        nested_list = torch.round(tensor, decimals=4).tolist()
        return nested_list

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:

        torch.manual_seed(0)
        
        # dimensions[0] is input_dim, dimensions[1:] are hidden_dim
        dims = [input_dim] + [hidden_dim] * num_layers
        weights = []

        # 2. Build weight matrices
        for i in range(num_layers):
            # Calculate standard deviation based on initialization type
            if init_type == "kaiming":
                # He initialization for ReLU networks
                std = math.sqrt(2.0 / dims[i])
            elif init_type == "xavier":
                # Glorot initialization (assuming uniform scaling adapted for Normal)
                std = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
            elif init_type == "random":
                std = 1.0  # Plain N(0,1), no scaling
            else:
                raise ValueError(f"Unknown init_type: {init_type}")

            # Generate weight matrix of shape (out_features, in_features)
            # It is crucial that this happens BEFORE x = torch.randn(...)
            w = torch.randn(dims[i+1], dims[i]) * std
            weights.append(w)

        # 3. Generate the random input AFTER weights to maintain correct RNG state
        x = torch.randn(1, input_dim)
        stds = []
        
        # 4. Forward pass
        for w in weights:
            # Linear transformation: x (1, in) @ w.T (in, out) -> (1, out)
            x = x @ w.T
            # ReLU activation
            x = torch.relu(x)
            # Calculate standard deviation (with Bessel's correction, default in PyTorch)
            # Round to 2 decimal places and append
            stds.append(round(x.std().item(), 2))

        return stds

