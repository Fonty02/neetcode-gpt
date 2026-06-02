import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        x = np.array(x, dtype=float)            
        gamma = np.array(gamma, dtype=float)
        beta = np.array(beta, dtype=float)
        running_mean = np.array(running_mean, dtype=float)
        running_var = np.array(running_var, dtype=float)

        row, col = len(x), len(x[0])

        if training:
            mean, var = [0.0] * col, [0.0] * col
            for idx_r, rowL in enumerate(x):
                for idx_c, element in enumerate(rowL):
                    mean[idx_c] += element
            mean = np.array([m / row for m in mean])
            for idx_r, rowL in enumerate(x):
                for idx_c, element in enumerate(rowL):
                    var[idx_c] += (element - mean[idx_c]) ** 2
            var = np.array([v / row for v in var])

            # aggiornamento running stats (EMA)
            running_mean = (1 - momentum) * running_mean + momentum * mean
            running_var  = (1 - momentum) * running_var  + momentum * var
        else:
            mean, var = running_mean, running_var   # inference: usa le running stats

        x_hat = (x - mean) / np.sqrt(var + eps)      # ← x_hat assegnato direttamente
        y = x_hat * gamma + beta

        return (np.round(y, 4).tolist(),
                np.round(running_mean, 4).tolist(),
                np.round(running_var, 4).tolist())

        
        
