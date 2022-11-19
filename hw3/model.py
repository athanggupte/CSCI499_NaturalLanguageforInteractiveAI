# IMPLEMENT YOUR MODEL CLASS HERE
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def __init__(self, device, encoder, decoder, hidden_dim, a2i, t2i, max_out_len=25):
        super().__init__()
        assert(device == encoder.device == decoder.device)
        assert(encoder.hidden_dim == hidden_dim == decoder.hidden_dim)
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.out_actions_dim = len(a2i)
        self.out_targets_dim = len(t2i)
        self.max_out_len = max_out_len
        # self.out_dim = len(a2i) + len(t2i)

        self.fc_a = nn.Linear(in_features=hidden_dim, out_features=self.out_actions_dim)
        self.fc_t = nn.Linear(in_features=hidden_dim, out_features=self.out_targets_dim)

    def forward(self, x, x_lens, y=None, y_lens=None):
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
            out_actions = torch.zeros((batch_size, self.max_out_len, self.out_actions_dim), device=self.device)
            out_targets = torch.zeros((batch_size, self.max_out_len, self.out_targets_dim), device=self.device)
            
            y_act = torch.ones((batch_size, 1), dtype=torch.int32, device=self.device)
            y_tar = torch.ones((batch_size, 1), dtype=torch.int32, device=self.device)
            y_lens = torch.ones((batch_size), dtype=torch.int64)
            h_dec, c_dec = h_enc, c_enc

            stop_mask = torch.ones((batch_size, 1), dtype=torch.long, device=self.device)
            for i in range(self.max_out_len):
                # self.decoder
                # y            : (batch_size, 1, 2)
                # dec_out      : (batch_size, 1, hidden_dim)
                # h_enc, c_enc : (batch_size, 1, hidden_dim)
                dec_out, (h_dec, c_dec), _ = self.decoder(y_act, y_tar, y_lens, h_dec, c_dec)
                out_actions[:,i,:] = self.fc_a(dec_out[:,0,:])
                out_targets[:,i,:] = self.fc_t(dec_out[:,0,:])
                y_act = torch.argmax(out_actions[:,i,:], dim=1, keepdim=True) * stop_mask
                y_tar = torch.argmax(out_targets[:,i,:], dim=1, keepdim=True) * stop_mask
                stop_mask = stop_mask * (y_act != 2).long()
                
                if torch.all(stop_mask == 0):
                    break
                #print("y_act : ", y_act)
                #print("stop_mask : ", stop_mask)
            return out_actions, out_targets
            


class EncoderDecoderAttention(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, device, encoder, decoder, hidden_dim, a2i, t2i, max_out_len=25, num_attn_heads=1, attn_stride=0, attn_window=0):
        super().__init__()
        assert(device == encoder.device == decoder.device)
        assert(encoder.hidden_dim == hidden_dim == decoder.hidden_dim)
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.out_actions_dim = len(a2i)
        self.out_targets_dim = len(t2i)
        self.num_attn_heads = num_attn_heads
        self.max_out_len = max_out_len
        self.attn_stride = attn_stride
        self.attn_window = attn_window
        # self.out_dim = len(a2i) + len(t2i)

        self.attn = nn.ModuleList([nn.Linear(in_features=hidden_dim * 2, out_features=1, device=device) for _ in range(num_attn_heads)])

        self.fc_a = nn.Linear(in_features=hidden_dim, out_features=self.out_actions_dim)
        self.fc_t = nn.Linear(in_features=hidden_dim, out_features=self.out_targets_dim)


    # Calculates the attention weights and applies it to predict the action and target for the ith decoder output
    def _calculate_attn_for_dec_i(self, batch_size, i, dec_out_i, enc_out, out_actions, out_targets):
        out_actions_h = torch.zeros((batch_size, self.num_attn_heads, self.out_actions_dim), device=self.device)
        out_targets_h = torch.zeros((batch_size, self.num_attn_heads, self.out_targets_dim), device=self.device)

        if self.attn_stride > 0:
            start_idx = self.attn_stride * i
            end_idx = min(start_idx + self.attn_window, enc_out.size(1))
            enc_out = enc_out[:,start_idx : end_idx]

        dec_out_i = dec_out_i.broadcast_to((batch_size, enc_out.size(1), dec_out_i.size(2)))

        attn_input = torch.cat((dec_out_i, enc_out), dim=2)
        # print("attn_input : ", attn_input.size())

        for aidx in range(self.num_attn_heads):
            attn_weights = F.softmax(self.attn[aidx](attn_input), dim=1)
            # print("attn_weights : ", attn_weights.size())
            attn_applied = torch.bmm(attn_weights.view(batch_size, attn_weights.size(2), attn_weights.size(1)), enc_out).squeeze(1)

            out_actions_h[:,aidx,:] = self.fc_a(attn_applied)
            out_targets_h[:,aidx,:] = self.fc_t(attn_applied)

        out_actions[:,i,:] = F.normalize(torch.sum(out_actions_h, dim=1), dim=1)
        out_targets[:,i,:] = F.normalize(torch.sum(out_targets_h, dim=1), dim=1)



    def forward(self, x, x_lens, y=None, y_lens=None):
        batch_size, seq_len = x.size(0), x.size(1)

        # self.encoder
        # x            : (batch_size, seq_len)
        # enc_out      : (batch_size, per_seq_len, hidden_dim)
        # h_enc, c_enc : (batch_size, 1, hidden_dim)
        enc_out, (h_enc, c_enc) = self.encoder(x, x_lens)

        enc_out, enc_lens = pad_packed_sequence(enc_out, batch_first=True)

        # y is provided if teacher_forcing is enabled
        if y is not None:
            y_act, y_tar = y[:,:,0], y[:,:,1]

            # y           : (batch_size, out_seq_len)
            # out_actions : (batch_size, out_seq_len, len(a2i))
            # out_targets : (batch_size, out_seq_len, len(t2i))
            out_actions = torch.zeros((batch_size, y.size(1), self.out_actions_dim), device=self.device)
            out_targets = torch.zeros((batch_size, y.size(1), self.out_targets_dim), device=self.device)

            # self.encoder
            # y            : (batch_size, out_seq_len, 2)
            # dec_out      : (batch_size, out_seq_len, hidden_dim)
            # h_enc, c_enc : (batch_size, 1, hidden_dim)
            dec_out, (h_dec, c_dec), dec_out_lens = self.decoder(y_act, y_tar, y_lens, h_enc, c_enc)

            for i in range(dec_out.size(1)):
                dec_out_i = dec_out[:,i,:].unsqueeze(1)

                self._calculate_attn_for_dec_i(batch_size, i, dec_out_i, enc_out, out_actions, out_targets)
                
            return out_actions, out_targets
        else:
            # out_actions : (batch_size, max_seq_len, len(a2i))
            # out_targets : (batch_size, max_seq_len, len(t2i))
            out_actions = torch.zeros((batch_size, self.max_out_len, self.out_actions_dim), device=self.device)
            out_targets = torch.zeros((batch_size, self.max_out_len, self.out_targets_dim), device=self.device)
            
            y_act = torch.ones((batch_size, 1), dtype=torch.int32, device=self.device)
            y_tar = torch.ones((batch_size, 1), dtype=torch.int32, device=self.device)
            y_lens = torch.ones((batch_size), dtype=torch.int64)
            h_dec, c_dec = h_enc, c_enc

            stop_mask = torch.ones((batch_size, 1), dtype=torch.long, device=self.device)
            for i in range(self.max_out_len):
                # self.decoder
                # y            : (batch_size, 1, 2)
                # dec_out      : (batch_size, 1, hidden_dim)
                # h_enc, c_enc : (batch_size, 1, hidden_dim)
                dec_out, (h_dec, c_dec), _ = self.decoder(y_act, y_tar, y_lens, h_dec, c_dec)

                self._calculate_attn_for_dec_i(batch_size, i, dec_out, enc_out, out_actions, out_targets)

                y_act = torch.argmax(out_actions[:,i,:], dim=1, keepdim=True) * stop_mask
                y_tar = torch.argmax(out_targets[:,i,:], dim=1, keepdim=True) * stop_mask
                stop_mask = stop_mask * (y_act != 2).long()
                
                if torch.all(stop_mask == 0):
                    print()
                    break
            return out_actions, out_targets
            

from transformer_models import *

def get_transformer_encoder():
    from transformers import AutoModelForSequenceClassification
    return AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')