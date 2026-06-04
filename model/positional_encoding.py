import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.
        out = np.zeros((seq_len, d_model))        
        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    out[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                else:
                    out[pos, i] = np.cos(pos / (10000 ** ((i - 1) / d_model)))
                    
        return np.round(out, 5)

