import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        x = np.asarray(x, dtype=float)
        W1 = np.asarray(W1, dtype=float)
        b1 = np.asarray(b1, dtype=float)
        W2 = np.asarray(W2, dtype=float)
        b2 = np.asarray(b2, dtype=float)
        y_true = np.asarray(y_true, dtype=float)

        # ── Forward pass ──────────────────────────────────────
        z1 = W1 @ x + b1                          # pre-ReLU
        a1 = np.maximum(z1, 0.0)                  # ReLU
        y_hat = W2 @ a1 + b2                      # output
        L = float(np.round(np.mean((y_hat - y_true) ** 2), 4))

        # ── Backward pass ─────────────────────────────────────
        n = len(y_hat)
        dy_hat = 2 * (y_hat - y_true) / n         # dL/dy_hat

        db2 = np.round(dy_hat, 4)
        dW2 = np.round(np.outer(dy_hat, a1), 4)

        da1  = W2.T @ dy_hat                      # dL/da1
        dz1  = da1 * (z1 > 0).astype(float)       # dL/dz1  (ReLU gate)

        db1 = np.round(dz1, 4)
        dW1 = np.round(np.outer(dz1, x), 4)

        # ── Conversione a liste Python (fix formato) ──────────
        return {
            'loss': L,
            'dW1':  (dW1 + 0.0).tolist(),
            'db1':  (db1 + 0.0).tolist(),
            'dW2':  (dW2 + 0.0).tolist(),
            'db2':  (db2 + 0.0).tolist()
        }