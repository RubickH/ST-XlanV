import torch
import torch.nn as nn

from lib.config import cfg
import lib.utils as utils
from models.att_basic_model_xlanv import AttBasicModel
import blocks
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb
class XLANV_ENS(AttBasicModel):
    def __init__(self, models, tmp):
        super(XLANV_ENS, self).__init__()
        self.models = nn.ModuleList(models)

    # state[0] -- h, state[1] -- c
    def Forward(self, kwargs_list):
        return zip(*[m.Forward(**kwargs_list[i]) for i, m in enumerate(self.models)])  
        
    def init_hidden(self, bsz):
        return [m.init_hidden(bsz) for m in self.models]
    
    def get_logprobs_state(self, kwargs_list):
        # 'it' contains a word index

        output, state = self.Forward(kwargs_list)
        stacked_prob = torch.stack([F.softmax(m.logit(output[i]), dim=1) for i,m
                                in enumerate(self.models)], 2)

        weight = torch.ones([len(self.models)]).cuda()
        s = sum(weight)
        #weight = weight/sum(weight)
        logprobs = torch.zeros_like(stacked_prob[:,:,0])
        for i in range(len(self.models)):
            tp = weight[i]/s
            if(i == 0):
                logprobs += stacked_prob[:,:,i]*tp
            else:
                logprobs += stacked_prob[:,:,i]*tp
        
        #logprobs = torch.stack([F.softmax(m.logit(output[i]), dim=1) for i,m
        #                        in enumerate(self.models)], 2).mean(2).log()
     
        return logprobs.log(),state 
    
    def preprocess(self, **kwargs):
        gv_feat_cross = []
        cross_feats = []
        cat_mask = []
        p_cross_feats = []
       
        
        for i, m in enumerate(self.models):
            gv_feat_cross_tmp, cross_feats_tmp, p_cross_feats_tmp, cat_mask_tmp = m.preprocess( decode=True, **kwargs)
            gv_feat_cross.append(gv_feat_cross_tmp)
            cross_feats.append(cross_feats_tmp)
            cat_mask.append(cat_mask_tmp)
            p_cross_feats.append(p_cross_feats_tmp)

        return gv_feat_cross, cross_feats, cat_mask, p_cross_feats
            
            
    def make_kwargs(self,wt, gv_feat_cross, cross_feats, p_cross_feats, cat_mask, state):
        kwarg_list = []
        for i,m in enumerate(self.models):
            kwargs_tmp = m.make_kwargs(wt, gv_feat_cross[i], cross_feats[i], p_cross_feats[i], cat_mask[i], state[i])
            kwarg_list.append(kwargs_tmp)
        return kwarg_list
    
    
    def forward(self, **kwargs):
        seq = kwargs[cfg.PARAM.INPUT_SENT] 
        gv_feat_cross, cross_feats, cat_mask, p_cross_feats = self.preprocess(**kwargs)
    
        gv_feat_cross = [utils.expand_tensor(_, cfg.DATA_LOADER.SEQ_PER_IMG) for _ in gv_feat_cross]
        cross_feats = [utils.expand_tensor(_, cfg.DATA_LOADER.SEQ_PER_IMG) for _ in cross_feats]
        cat_mask = [utils.expand_tensor(_, cfg.DATA_LOADER.SEQ_PER_IMG) for _ in cat_mask]
        p_cross_feats = [utils.expand_tensor(_, cfg.DATA_LOADER.SEQ_PER_IMG) for _ in p_cross_feats]
    
        batch_size = gv_feat[0].size(0)
        state = self.init_hidden(batch_size)
        outputs = Variable(torch.zeros(batch_size, seq.size(1), self.vocab_size).cuda())
        
        for t in range(seq.size(1)):
            if self.training and t >=1 and self.ss_prob > 0:
                prob = torch.empty(batch_size).cuda().uniform_(0, 1)
                mask = prob < self.ss_prob
                if mask.sum() == 0:
                    wt = seq[:,t].clone()
                else:
                    ind = mask.nonzero().view(-1)
                    wt = seq[:, t].data.clone()
                    prob_prev = torch.exp(outputs[:, t-1].detach())
                    wt.index_copy_(0, ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, ind))
            else:
                wt = seq[:,t].clone()
    
            if t >= 1 and seq[:, t].max() == 0:
                break
    
            kwarg_list = self.make_kwargs(wt, gv_feat, res152_feats, res152_mask, p_res152_feats, gv_feat_i3d, i3d_feat, i3d_mask, p_i3d_feat, state)
            output, state = self.Forward(kwarg_list)
    
            logit = torch.stack([m.logit(output[i]) for i,m
                                in enumerate(self.models)], 2).mean(2)#.log()
            outputs[:, t] = logit
    
        return outputs
        
    
    def decode_beam(self, **kwargs):
        gv_feat_cross, cross_feats, cat_mask, p_cross_feats = self.preprocess(**kwargs)
    
 
        beam_size = kwargs['BEAM_SIZE']
        batch_size = gv_feat_cross[0].size(0)
        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()

        state = self.init_hidden(batch_size)
        wt = Variable(torch.ones(batch_size, dtype=torch.long).cuda())


        outputs = []
        for t in range(cfg.MODEL.SEQ_LEN):
            cur_beam_size = 1 if t == 0 else beam_size
            kwarg_list = self.make_kwargs(wt, gv_feat_cross, cross_feats, p_cross_feats, cat_mask, state)
 
            word_logprob, state = self.get_logprobs_state(kwarg_list)
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != 2).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob) 
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = (1e-7 + selected_idx / candidate_logprob.shape[-1]).type(torch.int64)
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]
            
            for i, m in enumerate(self.models):
                for s in range(len(state[i])):
                    state[i][s] = self._expand_state(batch_size, beam_size, cur_beam_size, state[i][s], selected_beam)

            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                selected_beam.unsqueeze(-1).expand(batch_size, beam_size, word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            wt = selected_words.squeeze(-1)

            if t == 0:
                cross_feats = [utils.expand_tensor(_, beam_size) for _ in cross_feats]
                gv_feat_cross = [utils.expand_tensor(_, beam_size) for _ in gv_feat_cross]
                cat_mask = [utils.expand_tensor(_, beam_size) for _ in cat_mask]
                p_cross_feats = [utils.expand_tensor(_, beam_size) for _ in p_cross_feats]
                
                 
                
        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))
        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        return outputs, log_probs
