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

def positional_encoding(input_pos, embed_dim, embedded_vector = None):
    # e.g. embedded vector = [ [.231, .56116,.005], [.098,.0068,.0098], [.231, .56116, .005] ] 
    # input_pos = 3, embed_dim = 3
    # input_pos corresponds to the idx of each input token
    # Creating a matrix of values of the same dimension of the embedded_vector to add to it
    positional_matrix = torch.empty(0,embed_dim)
    for pos_idx in range(1, input_pos + 1):
            positional_vector = torch.tensor([])
            for dim_idx in range(1, embed_dim + 1):
                if dim_idx % 2 == 0:
                    wav = np.sin(pos_idx / (10_000**(2 * dim_idx / embed_dim)))
                else:
                    wav = np.cos(pos_idx / (10_000**(2 * dim_idx / embed_dim)))

                positional_vector = torch.cat((positional_vector,torch.tensor(wav).unsqueeze(0)))
            positional_matrix = torch.cat((positional_matrix, positional_vector.unsqueeze(0)), dim=0)
                
    return positional_matrix + embedded_vecotor    


embedded_vecotor = embeddings(input_tensor_2)    

matrix = positional_encoding(3,4, embedded_vector=embedded_vecotor)
print(matrix)



