import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.embeddings = nn.Embedding(10,4)

    def forward(self, x: torch.Tensor):
        embedded = self.embeddings(x)
        shape = embedded.shape

        return shape

vocab_size = 10
embed_dim = 4

embeddings = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embed_dim)
input_tensor = torch.tensor([[1,2,3],
                             [1,2,3],
                             [1,2,3]])

input_tensor_2 = torch.tensor([1,2,3])

# The num of sinusoids = embed_dim (one sinusoid per dimension)
# Repeated for each word
# Vector of sinusoids gets added to indexed embeddings

def positional_encoding(input_pos, embed_dim, embedded_vector=None):
    """Generate sinusoidal positional encodings and add them to embeddings.

    Args:
        input_pos: Number of token positions in the input sequence.
        embed_dim: Number of dimensions in each token embedding.
        embedded_vector: Matrix of token embeddings with one row per position.

    Returns:
        The token embeddings with positional encodings added.

    Notes:
        This implementation assumes that embed_dim is even so that each
        sine value has a corresponding cosine value.
    """
    positional_matrix = torch.empty(0, embed_dim)

    for pos_idx in range(input_pos):
        positional_vector = torch.tensor([])

        # Each i represents one sine/cosine pair of embedding dimensions.
        for i in range(embed_dim // 2):
            denominator = 10_000 ** (2 * i / embed_dim)

            sine = np.sin(pos_idx / denominator)
            cosine = np.cos(pos_idx / denominator)

            positional_vector = torch.cat(
                (positional_vector, torch.tensor(sine).unsqueeze(0))
            )
            positional_vector = torch.cat(
                (positional_vector, torch.tensor(cosine).unsqueeze(0))
            )

        positional_matrix = torch.cat(
            (positional_matrix, positional_vector.unsqueeze(0)),
            dim=0,
        )

    return positional_matrix + embedded_vector   


embedded_vector = embeddings(input_tensor_2)    

matrix = positional_encoding(3,4, embedded_vector=embedded_vector)

seq_len =4
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
# Simulated raw attention scores (e.g., query @ key.T)
attention_scores = torch.randn(4, 4)

# Replace 'True' positions in the mask with negative infinity
masked_scores = attention_scores.masked_fill(causal_mask, float('-inf'))
print(masked_scores)
# Output will have -inf values in the upper right triangle, 
# ensuring the model ignores those future values.



