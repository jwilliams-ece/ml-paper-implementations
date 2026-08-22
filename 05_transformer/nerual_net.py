import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F



class Transformer(nn.Module):
    # Input: Tokenized sequence
    def __init__(self, vocab_size, d_model, heads):
        super().__init__()

        # Convert input sequence into emmbedded sequence 
        # Add positional encoding to retain sequence order
        self.input_embedding = nn.Embedding(vocab_size, d_model)

        # Linear transformation to compute Q, K, and V
        # in_features = d_model, out_features = d_k/d_v
        d_k = d_v = d_model // heads
        self.Q_layer = nn.Linear(in_features=d_model, out_features=d_k, dtype=torch.float64) 
        self.K_layer = nn.Linear(in_features=d_model, out_features=d_k, dtype=torch.float64) 
        self.V_layer = nn.Linear(in_features=d_model, out_features=d_v, dtype=torch.float64)

    class Encoder():
        # Recieves the values from the positional encoder
        # Needs Query(Q), Key(K), and Value(V) maticies to do scaled dot product attention
        # use nn.Linear to compute Q
        # Perform MatMul on Q @ K 
        def __init__(self, Q, K, V, dim):

            self.Q_matrix = Q
            self.K_matrix = K
            self.V_matrix = V
            self.dim = dim

        def self_attention(self):
            QT_mul = self.Q_matrix @ torch.transpose(self.K_matrix, dim0=0, dim1=1)
            sqrt_dk = torch.sqrt(torch.tensor(self.dim))
            sft_max = F.softmax((QT_mul / sqrt_dk), dim=1)
            attention = sft_max @ self.V_matrix

            return attention


    class Decoder():
        pass



    class PositionalEncoder():
        def __init__(self, embedded_matrix):
            """Generate sinusoidal positional encodings and add them to embeddings.

            Args:
                embedded_vector: Matrix of token embeddings with one row per position.

            Returns:
                The token embeddings with positional encodings added.

            Notes:
                This implementation assumes that embed_dim is even so that each
                sine value has a corresponding cosine value.
            """
            self.input_pos = embedded_matrix.shape[0] # Number of token positions in the input sequence.
            self.embed_dim = embedded_matrix.shape[1] # Number of dimensions in each token embedding.
            self.embedded_matrix = embedded_matrix

        def positional_encoding(self):
            positional_matrix = torch.empty(0, self.embed_dim)

            for pos_idx in range(self.input_pos):
                positional_vector = torch.tensor([])

                # Each i represents one sine/cosine pair of embedding dimensions.
                for i in range(self.embed_dim // 2):
                    denominator = 10_000 ** (2 * i / self.embed_dim)

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

            return positional_matrix + self.embedded_matrix   


    def forward(self, input):
        # Input is a 2D tensor of tokens
        embedded_matrix = self.input_embedding(input)
        positional_matrix = self.PositionalEncoder(embedded_matrix=embedded_matrix).positional_encoding()
        Q_layer = self.Q_layer(positional_matrix)
        K_layer = self.K_layer(positional_matrix)
        V_layer = self.V_layer(positional_matrix)

        attention = self.Encoder(Q=Q_layer, K=K_layer, V=V_layer, dim=1).self_attention()

        return attention
        

test_tokens = torch.tensor([0,1,2])

model = Transformer(vocab_size=200,d_model=20,heads=1)
attention = model(test_tokens)

print(attention)

    
