import torch
import torch.nn as nn
import numpy as np



"""
This is the architecture for CBOW

"""


class CBOW():
    def __init__(self, input_dim):
        pass


    # This uses a shallow nn with a single hidden layer

    # The input layer consists of the aggregated context vector,
    # representing the contexed words

    # The hidden layer contains a set of neurons
    # The output layer has one neuron for each word in the vocabulary
    # The dimensions of the weights matrix is VxD

    # The model learns to predict the target word by adjusting the weights
    # between the input and hidden layers during training
