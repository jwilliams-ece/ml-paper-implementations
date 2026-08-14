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
    shifted_logits = logits - np.max(logits)

    exp_values = np.exp(shifted_logits)
    denominator = exp_values.sum()

    probabilities = exp_values / denominator

    return probabilities

def negative_log_loss(probabilites, target):
    loss = - (np.log(probabilites[target] + 1e-10))
    return loss

def do_gradient(predictions, target):
    arr = predictions.copy()
    arr[target] -= 1
    do = arr

    return do

def embeddings_gradient(h, output_grad):
    dW = np.outer(np.transpose(h), output_grad)

    return dW

def dh_gradient(output_grad):
    dh = output_grad @ np.transpose(model.output_matrix)

    return dh

def dE_gradient(context,dh):
    c = len(context)
    dE_context = dh/c

    return dE_context

def update_weights(lr, context, dE_context):
    for vector in context:
        model.embbeddings_matrix[vector] = (model.embbeddings_matrix[vector]) -  (lr * dE_context)




def train():

    for context, target in data_pairs:
        logits = model(context)




