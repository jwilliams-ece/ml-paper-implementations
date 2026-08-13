import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray



"""
This is the architecture for CBOW

"""


class CBOW():
    def __init__(self, vocab_size, dimension_size):
        self.embbeddings_matrix = np.random.uniform(-1,1,(vocab_size,dimension_size))
        self.output_matrix = np.random.uniform(-1,1,(dimension_size,vocab_size))

    def aggregate(self,input):
        mean = np.mean(self.embbeddings_matrix[input], axis=0)
        
        return mean

    def forward(self, input: NDArray):
        h = self.aggregate(input=input)
        out = h @ self.output_matrix

        return out


        

    # This uses a shallow nn with a single hidden layer

    # The input layer consists of the aggregated context vector,
    # representing the contexed words

    # The hidden layer contains a set of neurons
    # The output layer has one neuron for each word in the vocabulary
    # The dimensions of the weights matrix is VxD

    # The model learns to predict the target word by adjusting the weights
    # between the input and hidden layers during training
