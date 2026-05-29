import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        m=max(z)
        s=sum([np.exp(x-m) for x in z])
        return np.round(np.array([np.exp(x-m)/s for x in z]),4)
