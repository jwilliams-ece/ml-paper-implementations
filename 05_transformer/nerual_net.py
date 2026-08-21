import torch
import torch.nn as nn

"""

This is an implementation of the Attention is All You Need Architecure

"""

# Transformer class that employs the Encoder -> Decoder architecture 

# Encoder Class

# Decoder Class

# forward pass output generation



# Encoder: Input is tokens, output is vecotrized tokens of continuous values


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
        def __init__(self, num_embeddings, embed_dim):
            pass


    def forward(self, input):
        # Input is a vector of tokens
        pass
        




model = Transformer(200,20)


    
