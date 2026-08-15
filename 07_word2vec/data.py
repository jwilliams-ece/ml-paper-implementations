import torch
from torch.utils.data import Dataset, DataLoader

from collections import Counter


# -----------------------------
# Load and preprocess text8
# -----------------------------

with open(
    r"C:\Users\marve\Desktop\papers\07_word2vec\data\text8\text8",
    "r",
    encoding="utf-8",
) as f:
    text = f.read()

tokens = text.split()

word_counts = Counter(tokens)

min_count = 5

# Keep words that appear often enough
vocab_words = [
    word
    for word, count in word_counts.items()
    if count >= min_count
]

vocab_size = len(vocab_words)

word_to_idx = {
    word: i
    for i, word in enumerate(vocab_words)
}

idx_to_word = {
    i: word
    for word, i in word_to_idx.items()
}


# -----------------------------
# Encode the actual corpus
# -----------------------------

encoded = [
    word_to_idx[word]
    for word in tokens
    if word in word_to_idx
]


# -----------------------------
# CBOW Dataset
# -----------------------------

class CBOWDataset(Dataset):
    def __init__(self, encoded_corpus, window_size):
        self.encoded_corpus = encoded_corpus
        self.window_size = window_size

    def __len__(self):
        return len(self.encoded_corpus)

    def __getitem__(self, center_idx):
        context = []

        for offset in range(
            -self.window_size,
            self.window_size + 1
        ):
            if offset == 0:
                continue

            context_idx = center_idx + offset

            if 0 <= context_idx < len(self.encoded_corpus):
                context.append(
                    self.encoded_corpus[context_idx]
                )

        target = self.encoded_corpus[center_idx]

        context = torch.tensor(
            context,
            dtype=torch.long
        )

        target = torch.tensor(
            target,
            dtype=torch.long
        )

        return context, target


# -----------------------------
# Build dataset
# -----------------------------

def get_data_loader(window_size, batch_size):
    dataset = CBOWDataset(
        encoded_corpus=encoded,
        window_size=window_size,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataset, dataloader


def test():
    window_size = 5

    dataset = CBOWDataset(
    encoded_corpus=encoded,
    window_size=window_size,
    )


    print("Vocabulary size:", vocab_size)
    print("Encoded corpus size:", len(encoded))
    print("Number of training examples:", len(dataset))


    # Inspect one example
    context, target = dataset[0]

    print("\nContext IDs:")
    print(context)

    print("\nTarget ID:")
    print(target)

    print("\nContext words:")
    print([
    idx_to_word[token.item()]
    for token in context
    ])

    print("\nTarget word:")
    print(idx_to_word[target.item()])
