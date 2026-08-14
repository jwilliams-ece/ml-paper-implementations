import numpy as np

# Available data
tokens = [
    "the", "quick", "brown", "fox",
    "jumps", "over", "the", "lazy", "dog"
]

# sort data and store into dict of numerical keys/values
vocab = sorted(set(tokens))
vocab_size = len(vocab)
EMBEDDING_DIM = 4 # size of each vector

def get_data():
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}

    encoded = [word_to_idx[word] for word in tokens]

    window = 2
    pairs = []


    for center_idx in range(len(encoded)):
        context = []
        for offset in range(-window, window + 1):

            if offset == 0:
                continue

            context_idx = center_idx + offset

            if 0 <= context_idx < len(encoded):  
                context.append(
                    encoded[context_idx]
                )

        pairs.append((
            context,
            encoded[center_idx]
        ))

    
    X_train = []
    y_train = []

    for context, target in pairs:
        X_train.append(context)
        y_train.append(target)

    return pairs, X_train, y_train