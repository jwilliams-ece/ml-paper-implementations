import numpy as np

from .neural_net import CBOW
from .data import(
    vocab_size, 
    EMBEDDING_DIM,
    get_data,
)


model = CBOW(vocab_size=vocab_size, dimension_size=EMBEDDING_DIM)
data_pairs, _ , _ = get_data()

def softmax(logits):
    output = []
    probs = []

    for i in logits:
        for j in logits:
            x += np.exp(j)
        
        prob = np.exp(i) / x

    return output

def train():

    for context, target in data_pairs:
        logits = model(context)
