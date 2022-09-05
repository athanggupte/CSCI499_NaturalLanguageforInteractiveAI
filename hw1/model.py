# IMPLEMENT YOUR MODEL CLASS HERE
from log import get_logger
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

log = get_logger()

class Model(torch.nn.Module):
    def __init__(
        self,
        device,
        # input_len,
        vocab_size,
        n_actions,
        n_targets,
        embedding_dim,
        lstm_hidden_dim
    ):
        super(Model, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        # self.input_len = input_len
        self.n_actions = n_actions
        self.n_targets = n_targets
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        # Embedding layer
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0, device=device)

        # LSTM Sequence layer
        self.lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_dim)

        # Fully Connected layers for both outputs
        self.fc_a = torch.nn.Linear(in_features=lstm_hidden_dim, out_features=n_actions)
        self.fc_t = torch.nn.Linear(in_features=lstm_hidden_dim, out_features=n_targets)

    def forward(self, x, x_lengths):
        batch_size, seq_len = x.size(0), x.size(1)

        # self.embedding
        # input  : (batch_size, seq_len)
        # output : (batch_size, seq_len, embedding_dim)
        embeds = self.embedding(x)

        # convert to (seq_len, batch_size, embedding_dim)
        # embeds = embeds.view(seq_len, batch_size, -1)
        embeds = embeds.transpose(0, 1).contiguous()

        # pack padded sequence before passing to RNN
        # output : (per_seq_len, batch_size, embedding_dim)
        packed_embeds = pack_padded_sequence(embeds, x_lengths, enforce_sorted=False)

        # self.lstm
        # input  : (per_seq_len, batch_size, embedding_dim)
        # output : (per_seq_len, batch_size, lstm_hidden_dim)
        # h_n    : (1, batch_size, lstm_hidden_dim)
        rnn_out, (h_n, c_n) = self.lstm(packed_embeds)

        # pad RNN's output packed sequence to make matrix
        # output : (seq_len, batch_size, lstm_hidden_dim)
        # padded_rnn_out, padded_rnn_lengths = pad_packed_sequence(rnn_out)

        # transpose and squeeze final hidden state
        # output : (batch_size, lstm_hidden_dim)
        h_n = h_n.transpose(0, 1).contiguous().squeeze(1)

        # self.fc_a
        # input  : (batch_size, lstm_hidden_dim)
        # output : (batch_size, n_actions)
        actions = self.fc_a(h_n)

        # self.fc_t
        # input  : (batch_size, lstm_hidden_dim)
        # output : (batch_size, n_targets)
        targets = self.fc_t(h_n)

        return actions, targets