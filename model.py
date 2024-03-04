import math

import torch
from torch import nn


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # this is from the paper where embedding is multiplied by square root of d_model
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    # d_model as size of vector
    # max lenght of sentence
    # dropout is to make model less over fit
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of dimensions (seq len to d_model) as for each word a d_model size embedding is required
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape seq len
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()) * (-math.log(1000.0) / d_model)
        # apply sin to even and cosine to odd
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        # register buffer needs to be used when ever we want to save the state of pe along with other params
        self.register_buffer('pe', pe)

    def forward(self, x):
        # we dont want this to be learnable tensor hence marking it as requires_grad as False
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)



