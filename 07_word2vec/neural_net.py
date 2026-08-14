import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray



"""
This is the architecture for CBOW

"""

class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class CBOW(Module):
    def __init__(self, vocab_size, dimension_size):
        self.embbeddings_matrix = np.random.uniform(-1,1,(vocab_size,dimension_size))
        self.output_matrix = np.random.uniform(-1,1,(dimension_size,vocab_size))

    def aggregate(self,input):
        mean = np.mean(self.embbeddings_matrix[input], axis=0)

        return mean

    def forward(self, input: NDArray):
        h = self.aggregate(input=input)
        out = h @ self.output_matrix

        return out, h
