import torch
import torch.nn as nn

"""
This is the architecture for CBOW

"""

class CBOW(nn.Module):
    def __init__(self, vocab_size, dimension_size):
        super().__init__()
        self.embeddings_matrix = nn.Embedding(vocab_size,dimension_size)
        self.output_matrix = nn.Linear(dimension_size,vocab_size)

    def aggregate(self,input):
        embeddings = self.embeddings_matrix(input)
        context_vector = embeddings.mean(dim=1)

        return context_vector

    def forward(self, input):
        h = self.aggregate(input=input)
        out = self.output_matrix(h)

        return out, h
