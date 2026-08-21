import torch
import torch.nn as nn
import numpy as np



class Transformer(nn.Module):
    # Input: Tokenized sequence
    def __init__(self, vocab_size, input_dim):
        super().__init__()

        # Convert input sequence into emmbedded sequence 
        # Add positional encoding to retain sequence order
        self.input_embedding = nn.Embedding(vocab_size, input_dim)


    class Encoder():
        # Recieves the values from the positional encoder
        # Needs Query(Q), Key(K), and Value(V) maticies to do scaled dot product attention
        # use nn.Linear to compute Q
        # Perform MatMul on Q @ K 
        pass



    class Decoder():
        pass



    class PositionalEncoder():
        def __init__(self, input_pos, embed_dim, embedded_vector):
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
            self.input_pos = input_pos
            self.embed_dim = embed_dim
            self.embedded_vector = embedded_vector

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

            return positional_matrix + self.embedded_vector   


    def forward(self, input):
        # Input is a 2D tensor of tokens
        embedded_vector = self.input_embedding(input)
        pass
        




model = Transformer(200,20)


    
