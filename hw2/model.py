import torch
import torch.nn as nn
import torch.nn.functional as F

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim) -> None:
        super(CBOWModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.linear = nn.Linear(in_features=embedding_dim, out_features=vocab_size)

    def forward(self, batch):
        Y = self.embed(batch)
        Y = Y.mean(axis=1)
        Y = self.linear(Y)
        return Y


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.linear = nn.Linear(in_features=embedding_dim, out_features=vocab_size)

    def forward(self, batch):
        Y = self.embed(batch)
        Y = self.linear(Y)
        return Y


# class TransformerModel(nn.Module):
#     def __init__(self, embedding_dim, v2i) -> None:
#         super(TransformerModel, self).__init__()

#         self.embedding = nn.Embedding(num_embeddings=len(v2i), embedding_dim=embedding_dim)
#         self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=)
#         self.transformer_encoder = nn.TransformerEncoder