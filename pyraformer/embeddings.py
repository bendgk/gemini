import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()

        padding = 1 if torch.__version__>='1.5.0' else 2

        self.conv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=padding, padding_mode='circular')

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        return self.conv(x.permute(0, 2, 1)).transpose(1, 2)
    
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TemporalEmbedding, self).__init__()

        #seconds, minutes, hours, day
        d_inp = 4
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)

# Embedding for Gemini Financial Data
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, doupout=0.1, max_len=10000):
        super(DataEmbedding, self).__init__()

        self.token_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model, max_len=max_len)
        self.temporal_embedding = TemporalEmbedding(d_model)

        self.dropout = nn.Dropout(doupout)

    def forward(self, x, t):
        x = self.token_embedding(x) + self.position_embedding(x) + self.temporal_embedding(t)
        return self.dropout(x)
    

