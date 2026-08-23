import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F



class Transformer(nn.Module):
    # Input: Tokenized sequence
    def __init__(self, vocab_size, d_model, heads, text_len):
        super().__init__()

        # Convert input sequence into emmbedded sequence 
        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_input_embedding = nn.Embedding(vocab_size,d_model)

        # Linear transformation to compute Q, K, and V
        # in_features = d_model, out_features = d_k/d_v
        d_k = d_v = d_model // heads
        self.Q_layer = nn.Linear(in_features=d_model, out_features=d_k, dtype=torch.float64) 
        self.K_layer = nn.Linear(in_features=d_model, out_features=d_k, dtype=torch.float64) 
        self.V_layer = nn.Linear(in_features=d_model, out_features=d_v, dtype=torch.float64)

        # decoder level Q, K, V layers
        self.decoder_Q_layer = nn.Linear(in_features=d_model, out_features=d_k, dtype=torch.float64) 
        self.decoder_K_layer = nn.Linear(in_features=d_model, out_features=d_k, dtype=torch.float64) 
        self.decoder_V_layer = nn.Linear(in_features=d_model, out_features=d_v, dtype=torch.float64)

        # encoder-decoder Q, K, and V layers
        self.encode_decode_Q = nn.Linear(in_features=d_model,out_features=d_k, dtype=torch.float64)
        self.encode_decode_K = nn.Linear(in_features=d_model,out_features=d_k, dtype=torch.float64)
        self.encode_decode_V = nn.Linear(in_features=d_model,out_features=d_k, dtype=torch.float64)

        # Normalization for encoder sublayers
        # Encoder layers
        self.layer_norm_1 = nn.LayerNorm(d_k, dtype=torch.float64)
        self.layer_norm_2 = nn.LayerNorm(d_k, dtype=torch.float64)

        # Decoder layers
        self.layer_norm_3 = nn.LayerNorm(d_k, dtype=torch.float64)
        self.layer_norm_4 = nn.LayerNorm(d_k, dtype=torch.float64)
        self.layer_norm_5 = nn.LayerNorm(d_k, dtype=torch.float64)

        # Position-wise FFN for encoder
        self.encoder_ffn_w1 = nn.Linear(in_features=d_k, out_features=2048, dtype=torch.float64)
        self.encoder_ffn_w2 = nn.Linear(in_features=2048, out_features=d_k, dtype=torch.float64)

        # decoder FFN
        self.decoder_ffn_w1 = nn.Linear(in_features=d_k, out_features=2048, dtype=torch.float64)
        self.decoder_ffn_w2 = nn.Linear(in_features=2048, out_features=d_k, dtype=torch.float64)

        # output linear layer
        self.linear_output = nn.Linear(in_features=d_k, out_features=vocab_size, dtype=torch.float64)

    class Encoder():
        # Recieves the values from the positional encoder
        # Needs Query(Q), Key(K), and Value(V) maticies to do scaled dot product attention
        # use nn.Linear to compute Q
        # Perform MatMul on Q @ K 
        def __init__(self, Q, K, V, d_k):

            self.Q_matrix = Q
            self.K_matrix = K
            self.V_matrix = V
            self.d_k = d_k

        def self_attention(self):
            QT_mul = self.Q_matrix @ torch.transpose(self.K_matrix, dim0=0, dim1=1)
            sqrt_dk = torch.sqrt(torch.tensor(self.d_k))
            sft_max = F.softmax((QT_mul / sqrt_dk), dim=1)
            attention = sft_max @ self.V_matrix

            return attention


    class Decoder():
        def __init__(self, Q=None, K=None, V=None, d_k=None, Q_2=None, K_2=None, V_2=None):

            self.Q_matrix = Q
            self.K_matrix = K
            self.V_matrix = V
            self.d_k = d_k

            self.Q2_matrix = Q_2
            self.K2_matrix = K_2
            self.V2_matrix = V_2

        def masked_self_attention(self):
            mask = torch.triu(torch.ones(self.Q_matrix.shape[0],self.K_matrix.shape[0]), diagonal=1).bool()

            QK_t = self.Q_matrix @ torch.transpose(self.K_matrix, dim0=0,dim1=1)
            normalized = QK_t / torch.sqrt(torch.tensor(self.d_k))
            masked = normalized.masked_fill(mask, float('-inf'))
            softmask = torch.softmax(masked, dim=1)
            masked_attention = softmask @ self.V_matrix

            return masked_attention

        def self_attention(self):
            QT_mul = self.Q2_matrix @ torch.transpose(self.K2_matrix, dim0=0, dim1=1)
            sqrt_dk = torch.sqrt(torch.tensor(self.d_k))
            sft_max = F.softmax((QT_mul / sqrt_dk), dim=1)
            attention = sft_max @ self.V2_matrix

            return attention



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


    def forward(self, encoder_input, decoder_input):


        # 1) Encoder pass
        # Input is a 2D tensor of tokens
        embedded_matrix = self.input_embedding(encoder_input)
        positional_matrix = self.PositionalEncoder(embedded_matrix=embedded_matrix).positional_encoding()
        Q_layer = self.Q_layer(positional_matrix)
        K_layer = self.K_layer(positional_matrix)
        V_layer = self.V_layer(positional_matrix)

        # this is the feed forward section of the encoder 
        # TODO d_k needs to be modular not hard coded
        attention = self.Encoder(Q=Q_layer, K=K_layer, V=V_layer, d_k=20).self_attention()
        layer_norm_1 = self.layer_norm_1((attention + positional_matrix))
        encoder_ffn_w1 = self.encoder_ffn_w1(layer_norm_1)
        r1 = F.relu(encoder_ffn_w1)
        encoder_ffn_w2 = self.encoder_ffn_w2(r1)
        layer_norm_2 = self.layer_norm_2(encoder_ffn_w2 + layer_norm_1)


        # 2) Decoder pass
        decoder_embedded_matrix = self.decoder_input_embedding(decoder_input)
        decoder_positional_matrix = self.PositionalEncoder(embedded_matrix=decoder_embedded_matrix).positional_encoding()
        decoder_q = self.decoder_Q_layer(decoder_positional_matrix)
        decoder_k = self.decoder_K_layer(decoder_positional_matrix)
        decoder_v = self.decoder_V_layer(decoder_positional_matrix)

        # apply the masked attention; add & Normalize
        # TODO make d_k more modular
        masked_attention = self.Decoder(Q=decoder_q,K=decoder_k,V=decoder_v,d_k=20).masked_self_attention()
        layer_norm_3 = self.layer_norm_3((masked_attention + decoder_positional_matrix))
        encode_decode_Q = self.encode_decode_Q(layer_norm_3)
        encode_decode_K = self.encode_decode_K(layer_norm_2)
        encode_decode_V = self.encode_decode_V(layer_norm_2)
        decoder_attention = self.Decoder(Q_2=encode_decode_Q, K_2=encode_decode_K, V_2=encode_decode_V, d_k=20).self_attention()
        layer_norm_4 = self.layer_norm_4(decoder_attention + layer_norm_3)

        # Feed forward layer
        decoder_ffn_w1 = self.decoder_ffn_w1(layer_norm_4)
        ffn_relu = F.relu(decoder_ffn_w1)
        decoder_ffn_w2 = self.decoder_ffn_w2(ffn_relu)
        layer_norm_5 = self.layer_norm_5(decoder_ffn_w2 + layer_norm_4)
        linear_output = self.linear_output(layer_norm_5)
        soft_max = F.softmax(linear_output, dim=1)

        return soft_max
        


# Test data
test_tokens = torch.tensor([0,1,2])
text_len = len(test_tokens)

decoder_input_tokens = torch.tensor([3,4,5])

model = Transformer(vocab_size=200,d_model=20,heads=1,text_len=text_len)
soft_max = model(test_tokens, decoder_input_tokens)

print(soft_max)

    
