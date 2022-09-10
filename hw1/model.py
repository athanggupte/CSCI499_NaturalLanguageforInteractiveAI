# IMPLEMENT YOUR MODEL CLASS HERE
from log import get_logger
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

log = get_logger()

class Model(torch.nn.Module):
    def __init__(
        self,
        device,
        max_len,
        vocab_size,
        n_actions,
        n_targets,
        embedding_dim,
        lstm_hidden_dim,
    ):
        super(Model, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.max_len = max_len
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


class AttentionModel(torch.nn.Module):
    def __init__(
        self,
        device,
        max_len,
        vocab_size,
        n_actions,
        n_targets,
        embedding_dim,
        lstm_hidden_dim,
    ):
        super(AttentionModel, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_actions = n_actions
        self.n_targets = n_targets
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        # Embedding layer
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0, device=device)

        # LSTM Sequence layer
        self.lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_dim)
        self.lstm2 = torch.nn.LSTM(input_size=lstm_hidden_dim, hidden_size=lstm_hidden_dim)

        # Attention Layer
        self.attn = torch.nn.Linear(in_features=embedding_dim + lstm_hidden_dim, out_features=max_len)

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
        embeds_trans = embeds.transpose(0, 1).contiguous()

        # pack padded sequence before passing to RNN
        # output : (per_seq_len, batch_size, embedding_dim)
        packed_embeds = pack_padded_sequence(embeds_trans, x_lengths, enforce_sorted=False)

        # self.lstm
        # input  : (per_seq_len, batch_size, embedding_dim)
        # output : (per_seq_len, batch_size, lstm_hidden_dim)
        # h_n    : (1, batch_size, lstm_hidden_dim)
        rnn_out, (h_n, c_n) = self.lstm(packed_embeds)

        # pad RNN's output packed sequence to make matrix
        # output : (max_len, batch_size, lstm_hidden_dim)
        padded_rnn_out, padded_rnn_lengths = pad_packed_sequence(rnn_out, total_length=self.max_len)

        # transpose output sequence
        # output : (batch_size, max_len, lstm_hidden_dim)
        padded_rnn_out_trans = padded_rnn_out.transpose(0, 1)

        # pad embeddings to max_len for attn calculation
        # input  : (seq_len, batch_size, embedding_dim)
        # output : (max_len, batch_size, embedding_dim)
        padded_embeds = F.pad(embeds_trans, pad=(0,0, 0,0, 0,self.max_len-seq_len))

        # calculate attn at each input in the seq from the embedding and the output at that timestep
        # attn_inputs : (max_len, batch_size, lstm_hidden_dim + embedding_dim)
        attn_inputs = torch.cat((padded_rnn_out, padded_embeds), dim=2)
        attn_weights = torch.empty(self.max_len, batch_size, self.max_len, device=self.device)
        for i in range(self.max_len):
            # attn_inputs[i] : (batch_size, lstm_hidden_dim + embedding_dim)
            # attn_weights[i] : (batch_size)
            attn_weights[i] = F.softmax(self.attn(attn_inputs[i]))

        # apply self.attn output as weights to the lstm output
        # bmm = batch-wise matrix-matrix multiplication
        # inputs : (batch_size, max_len, max_len), (batch_size, max_len, lstm_hidden_dim)
        # output : (batch_size, max_len, lstm_hidden_dim)
        attn_applied = torch.bmm(attn_weights.transpose(0, 1), padded_rnn_out_trans)

        # pack padded sequence before passing to RNN
        # output : (per_seq_len, batch_size, embedding_dim)
        packed_attn_applied = pack_padded_sequence(attn_applied.transpose(0, 1), x_lengths, enforce_sorted=False)

        # self.lstm2
        # input  : (per_seq_len, batch_size, lstm_hidden_dim)
        # output : (per_seq_len, batch_size, lstm_hidden_dim)
        # h_n    : (1, batch_size, lstm_hidden_dim)
        rnn2_out, (h2_n, c2_n) = self.lstm2(packed_attn_applied)

        # transpose and squeeze final hidden state
        # output : (batch_size, lstm_hidden_dim)
        h2_n = h2_n.transpose(0, 1).contiguous().squeeze(1)

        # self.fc_a
        # input  : (batch_size, lstm_hidden_dim)
        # output : (batch_size, n_actions)
        actions = self.fc_a(h2_n)

        # self.fc_t
        # input  : (batch_size, lstm_hidden_dim)
        # output : (batch_size, n_targets)
        targets = self.fc_t(h2_n)

        return actions, targets


class ActionToTargetModel(torch.nn.Module):
    def __init__(
        self,
        device,
        max_len,
        vocab_size,
        n_actions,
        n_targets,
        embedding_dim,
        lstm_hidden_dim,
    ):
        super(ActionToTargetModel, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.max_len = max_len
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
        self.fc_t = torch.nn.Linear(in_features=lstm_hidden_dim + n_actions, out_features=n_targets)

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
        # input  : (batch_size, lstm_hidden_dim + n_actions)
        # output : (batch_size, n_targets)
        targets = self.fc_t(torch.cat((h_n, actions), dim=1))

        return actions, targets


class TargetToActionModel(torch.nn.Module):
    def __init__(
        self,
        device,
        max_len,
        vocab_size,
        n_actions,
        n_targets,
        embedding_dim,
        lstm_hidden_dim,
    ):
        super(TargetToActionModel, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_actions = n_actions
        self.n_targets = n_targets
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        # Embedding layer
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0, device=device)

        # LSTM Sequence layer
        self.lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_dim)

        # Fully Connected layers for both outputs
        self.fc_t = torch.nn.Linear(in_features=lstm_hidden_dim, out_features=n_targets)
        self.fc_a = torch.nn.Linear(in_features=lstm_hidden_dim + n_targets, out_features=n_actions)

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

        # self.fc_t
        # input  : (batch_size, lstm_hidden_dim)
        # output : (batch_size, n_targets)
        targets = self.fc_t(h_n)

        # self.fc_a
        # input  : (batch_size, lstm_hidden_dim + n_actions)
        # output : (batch_size, n_actions)
        actions = self.fc_a(torch.cat((h_n, targets), dim=1))

        return actions, targets