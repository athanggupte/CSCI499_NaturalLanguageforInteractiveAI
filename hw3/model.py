# IMPLEMENT YOUR MODEL CLASS HERE

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, device, embedding_dim, hidden_dim, v2i):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Embedding layer
        self.embedding = nn.Embedding(len(v2i), embedding_dim, padding_idx=0)

        # LSTM Sequence layer
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x, x_lens):
        batch_size, seq_len = x.size(0), x.size(1)

        # self.embedding
        # x      : (batch_size, seq_len)
        # embeds : (batch_size, seq_len, embedding_dim)
        embeds = self.embedding(x)

        # pack padded sequence before passing to RNN
        # packed_embeds : (batch_size, per_seq_len, embedding_dim)
        packed_embeds = pack_padded_sequence(embeds, x_lens, batch_first=True, enforce_sorted=False)

        # self.lstm
        # lstm_out : (batch_size, per_seq_len, hidden_dim)
        # h_n, c_n : (batch_size, 1, hidden_dim)
        lstm_out, (h_out, c_out) = self.lstm(packed_embeds)
        
        return lstm_out, (h_out, c_out)


class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, device, embedding_dim, hidden_dim, a2i, t2i):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Embedding layer
        self.action_embedding = nn.Embedding(len(a2i), embedding_dim, padding_idx=0)
        self.target_embedding = nn.Embedding(len(t2i), embedding_dim, padding_idx=0)

        # LSTM Sequence layer
        self.lstm = nn.LSTM(input_size=embedding_dim * 2, hidden_size=hidden_dim, batch_first=True)

    def forward(self, y_act, y_tar, y_lens, h_enc, c_enc):
        '''
        y : in case of student-forcing, y is a single pair of (<start>_act, <start>_tar) tokens (y_lens = 1)
            in case of teacher-forcing, y is list of all the labels of the expected output sequence
        '''
        batch_size, seq_len = y_act.size(0), y_act.size(1)

        # Split label
        # y            : (batch_size, seq_len, 2)
        # y_act, y_tar : (batch_size, seq_len)
        

        # self.embedding
        # act_emb, tar_emb : (batch_size, seq_len, embedding_dim)
        # embeds           : (batch_size, seq_len, embedding_dim * 2)
        act_emb = self.action_embedding(y_act)
        tar_emb = self.target_embedding(y_tar)
        embeds = torch.cat((act_emb, tar_emb), dim=2)

        # pack padded sequence before passing to RNN
        # packed_embeds : (batch_size, per_seq_len, embedding_dim * 2)
        packed_embeds = pack_padded_sequence(embeds, y_lens, batch_first=True, enforce_sorted=False) #if len(y_lens) > 1 else embeds

        # self.lstm
        # lstm_out     : (batch_size, per_seq_len, hidden_dim)
        # h_out, c_out : (batch_size, 1, hidden_dim)
        lstm_out, (h_out, c_out) = self.lstm(packed_embeds, (h_enc, c_enc))

        lstm_out, out_lens = pad_packed_sequence(lstm_out, batch_first=True) #if len(y_lens) > 1 else (lstm_out, None)
        
        return lstm_out, (h_out, c_out), out_lens


class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, device, encoder, decoder, hidden_dim, a2i, t2i):
        super().__init__()
        assert(device == encoder.device == decoder.device)
        assert(encoder.hidden_dim == hidden_dim == decoder.hidden_dim)
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.out_actions_dim = len(a2i)
        self.out_targets_dim = len(t2i)
        # self.out_dim = len(a2i) + len(t2i)

        self.fc_a = nn.Linear(in_features=hidden_dim, out_features=self.out_actions_dim)
        self.fc_t = nn.Linear(in_features=hidden_dim, out_features=self.out_targets_dim)

    def forward(self, x, x_lens, y=None, y_lens=None, max_out_len=25):
        batch_size, seq_len = x.size(0), x.size(1)

        # self.encoder
        # x            : (batch_size, seq_len)
        # enc_out      : (batch_size, per_seq_len, hidden_dim)
        # h_enc, c_enc : (batch_size, 1, hidden_dim)
        enc_out, (h_enc, c_enc) = self.encoder(x, x_lens)

        # y is provided if teacher_forcing is enabled
        if y is not None:
            y_act, y_tar = y[:,:,0], y[:,:,1]

            # y           : (batch_size, out_seq_len)
            # out_actions : (batch_size, out_seq_len, len(a2i))
            # out_targets : (batch_size, out_seq_len, len(t2i))
            out_actions = torch.zeros((y.size(0), y.size(1), self.out_actions_dim), device=self.device)
            out_targets = torch.zeros((y.size(0), y.size(1), self.out_targets_dim), device=self.device)

            # self.encoder
            # y            : (batch_size, seq_len, 2)
            # dec_out      : (batch_size, per_seq_len, hidden_dim)
            # h_enc, c_enc : (batch_size, 1, hidden_dim)
            dec_out, (h_dec, c_dec), dec_out_lens = self.decoder(y_act, y_tar, y_lens, h_enc, c_enc)
            for i in range(dec_out.size(1)):
                # self.fc
                # input   : (batch_size, hidden_dim)
                # out_act : (batch_size, len(a2i))
                # out_tar : (batch_size, len(t2i))
                out_actions[:,i,:] = self.fc_a(dec_out[:,i,:])
                out_targets[:,i,:] = self.fc_t(dec_out[:,i,:])
            return out_actions, out_targets
        else:
            # out_actions : (batch_size, max_seq_len, len(a2i))
            # out_targets : (batch_size, max_seq_len, len(t2i))
            out_actions = torch.zeros((batch_size, max_out_len, self.out_actions_dim), device=self.device)
            out_targets = torch.zeros((batch_size, max_out_len, self.out_targets_dim), device=self.device)
            
            y_act = torch.ones((batch_size, 1), dtype=torch.int32, device=self.device)
            y_tar = torch.ones((batch_size, 1), dtype=torch.int32, device=self.device)
            y_lens = torch.ones((batch_size), dtype=torch.int64)
            h_dec, c_dec = h_enc, c_enc
            for i in range(max_out_len):
                # self.decoder
                # y            : (batch_size, 1, 2)
                # dec_out      : (batch_size, 1, hidden_dim)
                # h_enc, c_enc : (batch_size, 1, hidden_dim)
                dec_out, (h_dec, c_dec), _ = self.decoder(y_act, y_tar, y_lens, h_dec, c_dec)
                out_actions[:,i,:] = self.fc_a(dec_out[:,0,:])
                out_targets[:,i,:] = self.fc_t(dec_out[:,0,:])
                y_act = torch.argmax(out_actions[:,i,:], dim=1, keepdim=True)
                y_tar = torch.argmax(out_targets[:,i,:], dim=1, keepdim=True)
            return out_actions, out_targets
            