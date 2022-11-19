import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class EncoderDecoderTransformerAttention(nn.Module):
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
        self.max_out_len = max_out_len
        self.num_attn_heads = num_attn_heads
        self.attn_stride = attn_stride
        self.attn_window = attn_window
        self.hidden_dim = hidden_dim
        # self.out_dim = len(a2i) + len(t2i)

        self.attn = nn.ModuleList([nn.Linear(in_features=hidden_dim * 2, out_features=1, device=device) for _ in range(num_attn_heads)])

        self.fc_a = nn.Linear(in_features=hidden_dim, out_features=self.out_actions_dim)
        self.fc_t = nn.Linear(in_features=hidden_dim, out_features=self.out_targets_dim)


    # Calculates the attention weights and applies it to predict the action and target for the ith decoder output
    def _calculate_attn_for_dec_i(self, batch_size, i, dec_out_i, enc_out, out_actions, out_targets):
        out_actions_h = torch.zeros((batch_size, self.num_attn_heads, self.out_actions_dim), device=self.device)
        out_targets_h = torch.zeros((batch_size, self.num_attn_heads, self.out_targets_dim), device=self.device)

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
        # enc_out      : (batch_size, seq_len, hidden_dim)
        enc_out = self.encoder(x)

        if self.attn_window == 0:
            self.attn_window = enc_out.size(1)

        # h_dec, c_dec : (batch_size, 1, hidden_dim)
        h_dec, c_dec = enc_out[:,-1].unsqueeze(0), torch.zeros((1, batch_size, self.hidden_dim), device=self.device)

        # y is provided if teacher_forcing is enabled
        if y is not None:
            y_act, y_tar = y[:,:,0], y[:,:,1]

            # y           : (batch_size, out_seq_len)
            # out_actions : (batch_size, out_seq_len, len(a2i))
            # out_targets : (batch_size, out_seq_len, len(t2i))
            out_actions = torch.zeros((batch_size, y.size(1), self.out_actions_dim), device=self.device)
            out_targets = torch.zeros((batch_size, y.size(1), self.out_targets_dim), device=self.device)


            # self.decoder
            # y            : (batch_size, out_seq_len, 2)
            # dec_out      : (batch_size, out_seq_len, hidden_dim)
            # h_dec, c_dec : (batch_size, 1, hidden_dim)
            dec_out, (h_dec, c_dec), dec_out_lens = self.decoder(y_act, y_tar, y_lens, h_dec, c_dec)

            for i in range(dec_out.size(1)):
                dec_out_i = dec_out[:,i,:].unsqueeze(1).broadcast_to((batch_size, enc_out.size(1), dec_out.size(2)))

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

            stop_mask = torch.ones((batch_size, 1), dtype=torch.long, device=self.device)
            for i in range(self.max_out_len):
                # self.decoder
                # y            : (batch_size, 1, 2)
                # dec_out      : (batch_size, 1, hidden_dim)
                # h_enc, c_enc : (batch_size, 1, hidden_dim)
                dec_out, (h_dec, c_dec), _ = self.decoder(y_act, y_tar, y_lens, h_dec, c_dec)

                dec_out_i = dec_out.broadcast_to((batch_size, enc_out.size(1), dec_out.size(2)))

                self._calculate_attn_for_dec_i(batch_size, i, dec_out_i, enc_out, out_actions, out_targets)

                y_act = torch.argmax(out_actions[:,i,:], dim=1, keepdim=True) * stop_mask
                y_tar = torch.argmax(out_targets[:,i,:], dim=1, keepdim=True) * stop_mask
                stop_mask = stop_mask * (y_act != 2).long()
                
                if torch.all(stop_mask == 0):
                    break
            return out_actions, out_targets
            


class EncoderTransformer(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    
    Implemented as instructed in https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, device, embedding_dim, hidden_dim, num_heads, num_layers, v2i, max_len=512):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.embedding = nn.Embedding(len(v2i), embedding_dim, padding_idx=0)

        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.init_positional_encoding()

    def init_positional_encoding(self):
        """
        Implemented from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        """
        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2) * (-math.log(2 * self.max_len) / self.embedding_dim))
        pe = torch.zeros(self.max_len, self.embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pos_encoding = pe.unsqueeze(0).to(self.device)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        embeds = self.embedding(x)
        embeds = embeds + self.pos_encoding[:,:seq_len]
        
        # self.encoder
        # input  : (batch_size, seq_len, embedding_dim)
        # output : (batch_size, seq_len, hidden_dim)
        enc_out = self.encoder(embeds)
        
        return enc_out


