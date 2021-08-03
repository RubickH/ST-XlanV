import torch
import torch.nn as nn

from lib.config import cfg
import lib.utils as utils
from models.att_basic_model_xlanv import AttBasicModel
import blocks

class XLANV(AttBasicModel):
    def __init__(self, pos=True, tag=True):
        super(XLANV, self).__init__(pos,tag)
        self.num_layers = 2

        # First LSTM layer
        rnn_input_size = cfg.MODEL.RNN_SIZE + cfg.MODEL.BILINEAR.DIM #+ cfg.MODEL.BILINEAR.DIM
        self.att_lstm = nn.LSTMCell(rnn_input_size, cfg.MODEL.RNN_SIZE)
        self.ctx_drop = nn.Dropout(cfg.MODEL.DROPOUT_LM)

        self.attention_cross = blocks.create(            
            cfg.MODEL.BILINEAR.DECODE_BLOCK, 
            embed_dim = cfg.MODEL.BILINEAR.DIM, 
            att_type = cfg.MODEL.BILINEAR.ATTTYPE,
            att_heads = cfg.MODEL.BILINEAR.HEAD,
            att_mid_dim = cfg.MODEL.BILINEAR.DECODE_RES152_MID_DIM,
            att_mid_drop = cfg.MODEL.BILINEAR.DECODE_RES152_MID_DROPOUT,
            dropout = cfg.MODEL.BILINEAR.DECODE_DROPOUT, 
            layer_num = cfg.MODEL.BILINEAR.DECODE_LAYERS
        )
       
        
        self.linear_alpha = nn.Linear(cfg.MODEL.BILINEAR.DIM * 3, cfg.MODEL.BILINEAR.DIM)
        self.att2ctx = nn.Sequential(
            nn.Linear(cfg.MODEL.BILINEAR.DIM + cfg.MODEL.RNN_SIZE, 2 * cfg.MODEL.RNN_SIZE), 
            nn.GLU()
        )

    # state[0] -- h, state[1] -- c
    def Forward(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        cross_feats = kwargs[cfg.PARAM.CROSS_FEATS]
        cross_mask = kwargs[cfg.PARAM.CROSS_FEATS_MASK]
        state = kwargs[cfg.PARAM.STATE]
        gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
        p_cross_feats = kwargs[cfg.PARAM.P_CROSS_FEATS]
        
        
        if gv_feat.shape[-1] == 1:  # empty gv_feat
            if cross_mask is not None:
                gv_feat = torch.sum(cross_feats * cross_mask.unsqueeze(-1), 1) / torch.sum(cross_mask.unsqueeze(-1), 1)
            else:
                gv_feat = torch.mean(cross_feats, 1)               
        

        xt = self.word_embed(wt)
        
        h_att, c_att = self.att_lstm(torch.cat([xt, gv_feat + self.ctx_drop(state[0][1])], 1), (state[0][0], state[1][0]))
        cross, _ = self.attention_cross(h_att, cross_feats, cross_mask, p_cross_feats, precompute=True)
       
        ctx_input = torch.cat([cross, h_att], 1)

        output = self.att2ctx(ctx_input)
        state = [torch.stack((h_att, output)), torch.stack((c_att, state[1][1]))]

        return output, state