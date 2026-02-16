import torch
import torch.nn as nn

#TODO
from layers import EncoderLayer, Decoder, Predictor
from layers import Bottleneck_Construct
from layers import get_mask, get_subsequent_mask

from embeddings import DataEmbedding

class Encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()

        #attention decoder
        self.mask, self.all_size = get_mask(opt.input_size, opt.window_size, opt.inner_size, opt.device)

        self.layers = nn.ModuleList([
            EncoderLayer(opt.d_model, opt.d_inner_hid, opt.n_heads, opt.d_k, opt.d_v, dropout=opt.dropout, normalize_before=False) for i in range(opt.n_layer)
        ])

        self.enc_embedding = DataEmbedding(opt.enc_in, opt.d_model, opt.dropout)
        self.conv_layers = eval(opt.CSCM)(opt.d_model, opt.window_size, opt.d_bottleneck)

    def forward(self, x, t):
        seq_enc = self.enc_embedding(x, t)

        mask = self.mask.repeat(len(seq_enc), 1, 1).to(x.device)
        seq_enc = self.conv_layers(seq_enc)

        for i in range(len(self.layers)):
            seq_enc, _ = self.layers[i](seq_enc, mask)

        return seq_enc
    
class Model(nn.Module):
    def __init__(self, input_size=168, predict_size=168, d_model=512, d_inner_hid=512, d_k=128, d_v=128, n_head=6, n_layer=6, window_size=[4, 4, 4], dropout=0.05):
        super().__init__()

        # input is observations, predictions is output
        self.input_size = input_size
        self.predict_size = predict_size

        # model parameters
        self.d_model = d_model
        self.window_size = window_size
        self.inner_size = 3
        self.d_inner_hid = d_inner_hid
        self.n_heads = n_head
        self.n_layer = n_layer
        self.dropout = dropout

        self.enc_in = 7

        self.d_k = d_k
        self.d_v = d_v

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.CSCM = "Bottleneck_Construct"
        self.d_bottleneck = 128

        self.opt = {
            "input_size": self.input_size,
            "predict_size": self.predict_size,
            "d_model": self.d_model,
            "window_size": self.window_size,
            "inner_size": self.inner_size,
            "d_inner_hid": self.d_inner_hid,
            "n_heads": self.n_heads,
            "n_layer": self.n_layer,
            "dropout": self.dropout,
            "d_k": self.d_k,
            "d_v": self.d_v,
            "device": self.device,
            "CSCM": self.CSCM,
            "d_bottleneck": self.d_bottleneck,
            "enc_in": self.enc_in,
            "model": "Pyraformer",
            "n_head": self.n_heads
        }
        
        # Convert dict to namespace for attribute access
        from types import SimpleNamespace
        self.opt = SimpleNamespace(**self.opt)

        self.encoder = Encoder(self.opt)
        mask = get_subsequent_mask(self.input_size, self.window_size, self.predict_size, False)
        self.decoder = Decoder(self.opt, mask)
        self.predictor = Predictor(self.d_model, self.enc_in)

    def forward(self, x_enc, t_enc, x_dec, t_dec, pretrain):
        enc_output = self.encoder(x_enc, t_enc)
        dec_enc = self.decoder(x_dec, t_dec, enc_output)

        if pretrain:
            dec_enc = torch.cat([enc_output[:, :self.input_size], dec_enc], dim=1)
            pred = self.predictor(dec_enc)
        else:
            pred = self.predictor(dec_enc)

        return pred


