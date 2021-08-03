import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import blocks
import lib.utils as utils
from lib.config import cfg
from models.basic_model import BasicModel

#added in 2021
import os, pdb
from modules.module_bert import BertModel, BertConfig, BertOnlyMLMHead
from torch.nn import CrossEntropyLoss
import logging
logger = logging.getLogger(__name__)

class AttBasicModel(BasicModel):
    def __init__(self):
        super(AttBasicModel, self).__init__()
        self.ss_prob = 0.0                               # Schedule sampling probability
        self.vocab_size = 11734#cfg.MODEL.VOCAB_SIZE + 1       # include <BOS>/<EOS>
        self.att_dim = cfg.MODEL.RES152_FEATS_EMBED_DIM \
            if cfg.MODEL.RES152_FEATS_EMBED_DIM > 0 else cfg.MODEL.RES152_FEATS_DIM

        # BERT define and init
        bert_config, state_dict_bert = BertConfig.get_config(pretrained_model_name="bert-base-uncased", cache_dir='./pretrained_bert', \
                                                             type_vocab_size=2, state_dict=None, task_config=None, bert_dir='./pretrained_bert')
        old_keys = []
        new_keys = []
        for key in state_dict_bert.keys():
            new_key = None
            if key[:5] == "bert.":
                new_key = key[5:]
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict_bert[new_key] = state_dict_bert.pop(old_key)

        self.state_dict_bert = state_dict_bert
        self.bert = BertModel(bert_config)
        self.bert = init_preweight(self.bert, self.state_dict_bert)
        #self.bert_tag = BertModel(bert_config)
        #self.bert_tag = init_preweight(self.bert_tag, self.state_dict_bert)


        self.gv_feat_embed = None
        
        #i3d feat
        sequential = []
        if cfg.MODEL.S3D_FEATS_EMBED_DIM > 0:
            sequential.append(nn.Linear(cfg.MODEL.S3D_FEATS_DIM, cfg.MODEL.S3D_FEATS_EMBED_DIM))
        sequential.append(utils.activation(cfg.MODEL.S3D_FEATS_EMBED_ACT))
        if cfg.MODEL.DROPOUT_S3D_EMBED > 0:
            sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_S3D_EMBED))
        if cfg.MODEL.S3D_FEATS_NORM == True:
            sequential.append(torch.nn.LayerNorm(cfg.MODEL.S3D_FEATS_EMBED_DIM))
        self.s3d_embed = nn.Sequential(*sequential) if len(sequential) > 0 else None

        # attention feats embed
        sequential = []
        if cfg.MODEL.RES152_FEATS_EMBED_DIM > 0:
            sequential.append(nn.Linear(cfg.MODEL.RES152_FEATS_DIM, cfg.MODEL.RES152_FEATS_EMBED_DIM))
        sequential.append(utils.activation(cfg.MODEL.RES152_FEATS_EMBED_ACT))
        if cfg.MODEL.DROPOUT_RES152_EMBED > 0:
            sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_RES152_EMBED))
        if cfg.MODEL.RES152_FEATS_NORM == True:
            sequential.append(torch.nn.LayerNorm(cfg.MODEL.RES152_FEATS_EMBED_DIM))
        self.res152_embed = nn.Sequential(*sequential) if len(sequential) > 0 else None

        self.dropout_lm  = nn.Dropout(cfg.MODEL.DROPOUT_LM) if cfg.MODEL.DROPOUT_LM > 0 else None
        self.logit = nn.Linear(cfg.MODEL.RNN_SIZE, self.vocab_size)
        self.p_res152_feats = nn.Linear(self.att_dim, cfg.MODEL.ATT_HIDDEN_SIZE) \
            if cfg.MODEL.ATT_HIDDEN_SIZE > 0 else None
        
        
        # bilinear
        if cfg.MODEL.BILINEAR.DIM > 0:
            self.p_res152_feats = None
            self.encoder_layers_res152 = blocks.create(
                cfg.MODEL.BILINEAR.ENCODE_BLOCK, 
                embed_dim = cfg.MODEL.BILINEAR.DIM, 
                att_type = cfg.MODEL.BILINEAR.ATTTYPE,
                att_heads = cfg.MODEL.BILINEAR.HEAD,
                att_mid_dim = cfg.MODEL.BILINEAR.ENCODE_RES152_MID_DIM,
                att_mid_drop = cfg.MODEL.BILINEAR.ENCODE_RES152_MID_DROPOUT,
                dropout = cfg.MODEL.BILINEAR.ENCODE_DROPOUT, 
                layer_num = cfg.MODEL.BILINEAR.ENCODE_LAYERS
            )    
            self.encoder_layers_s3d = blocks.create(
                cfg.MODEL.BILINEAR.ENCODE_BLOCK, 
                embed_dim = cfg.MODEL.BILINEAR.DIM, 
                att_type = cfg.MODEL.BILINEAR.ATTTYPE,
                att_heads = cfg.MODEL.BILINEAR.HEAD,
                att_mid_dim = cfg.MODEL.BILINEAR.ENCODE_RES152_MID_DIM,
                att_mid_drop = cfg.MODEL.BILINEAR.ENCODE_RES152_MID_DROPOUT,
                dropout = cfg.MODEL.BILINEAR.ENCODE_DROPOUT, 
                layer_num = cfg.MODEL.BILINEAR.ENCODE_LAYERS
            )  
            self.encoder_layers_cross = blocks.create(
                cfg.MODEL.BILINEAR.ENCODE_BLOCK, 
                embed_dim = cfg.MODEL.BILINEAR.DIM, 
                att_type = cfg.MODEL.BILINEAR.ATTTYPE,
                att_heads = cfg.MODEL.BILINEAR.HEAD,
                att_mid_dim = cfg.MODEL.BILINEAR.ENCODE_RES152_MID_DIM,
                att_mid_drop = cfg.MODEL.BILINEAR.ENCODE_RES152_MID_DROPOUT,
                dropout = cfg.MODEL.BILINEAR.ENCODE_DROPOUT, 
                layer_num = cfg.MODEL.BILINEAR.ENCODE_LAYERS
            )   
            self.encoder_layers_ids = blocks.create(
                cfg.MODEL.BILINEAR.ENCODE_BLOCK, 
                embed_dim = cfg.MODEL.BILINEAR.DIM, 
                att_type = cfg.MODEL.BILINEAR.ATTTYPE,
                att_heads = cfg.MODEL.BILINEAR.HEAD,
                att_mid_dim = cfg.MODEL.BILINEAR.ENCODE_RES152_MID_DIM,
                att_mid_drop = cfg.MODEL.BILINEAR.ENCODE_RES152_MID_DROPOUT,
                dropout = cfg.MODEL.BILINEAR.ENCODE_DROPOUT, 
                layer_num = cfg.MODEL.BILINEAR.ENCODE_LAYERS
            ) 
            self.encoder_layers_tags = blocks.create(
                cfg.MODEL.BILINEAR.ENCODE_BLOCK, 
                embed_dim = cfg.MODEL.BILINEAR.DIM, 
                att_type = cfg.MODEL.BILINEAR.ATTTYPE,
                att_heads = cfg.MODEL.BILINEAR.HEAD,
                att_mid_dim = cfg.MODEL.BILINEAR.ENCODE_RES152_MID_DIM,
                att_mid_drop = cfg.MODEL.BILINEAR.ENCODE_RES152_MID_DROPOUT,
                dropout = cfg.MODEL.BILINEAR.ENCODE_DROPOUT, 
                layer_num = cfg.MODEL.BILINEAR.ENCODE_LAYERS
            ) 
            
        #pre-training codes
        bert_word_embeddings_weight = self.bert.embeddings.word_embeddings.weight
        self.mlm_head = BertOnlyMLMHead(bert_config, bert_word_embeddings_weight)
        self.alm_loss_fct = CrossEntropyLoss(ignore_index=-1)
        self.align_dense = nn.Linear(cfg.MODEL.BILINEAR.DIM, 2)

        self.logit.weight = bert_word_embeddings_weight
        self.logit.bias.data.zero_()

        emb = nn.Embedding(self.vocab_size, cfg.MODEL.WORD_EMBED_DIM)
        #emb.weight = bert_word_embeddings_weight
        sequential = [emb]
        sequential.append(utils.activation(cfg.MODEL.WORD_EMBED_ACT))
        if cfg.MODEL.WORD_EMBED_NORM == True:
            sequential.append(nn.LayerNorm(cfg.MODEL.WORD_EMBED_DIM))
        if cfg.MODEL.DROPOUT_WORD_EMBED > 0:
            sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_WORD_EMBED))
        self.word_embed = nn.Sequential(*sequential)

        emb = nn.Embedding(self.vocab_size, cfg.MODEL.WORD_EMBED_DIM)
        emb.weight = bert_word_embeddings_weight
        sequential = [emb]
        sequential.append(utils.activation(cfg.MODEL.WORD_EMBED_ACT))
        if cfg.MODEL.WORD_EMBED_NORM == True:
            sequential.append(nn.LayerNorm(cfg.MODEL.WORD_EMBED_DIM))
        if cfg.MODEL.DROPOUT_WORD_EMBED > 0:
            sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_WORD_EMBED))
        self.tag_embed = nn.Sequential(*sequential)
        
        emb = nn.Embedding(self.vocab_size, cfg.MODEL.WORD_EMBED_DIM)
        emb.weight = bert_word_embeddings_weight
        sequential = [emb]
        sequential.append(utils.activation(cfg.MODEL.WORD_EMBED_ACT))
        if cfg.MODEL.WORD_EMBED_NORM == True:
            sequential.append(nn.LayerNorm(cfg.MODEL.WORD_EMBED_DIM))
        if cfg.MODEL.DROPOUT_WORD_EMBED > 0:
            sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_WORD_EMBED))
        self.id_embed = nn.Sequential(*sequential)
        
        self.device = torch.device("cuda")

        self.pos_emb_id = nn.Embedding(512, cfg.MODEL.WORD_EMBED_DIM)
        self.pos_emb_tag = nn.Embedding(512, cfg.MODEL.WORD_EMBED_DIM)
        self.pos_emb_s3d = nn.Embedding(512, cfg.MODEL.S3D_FEATS_EMBED_DIM)
        self.pos_emb_res152 = nn.Embedding(512, cfg.MODEL.RES152_FEATS_EMBED_DIM)

        
    def init_hidden(self, batch_size):
        return [Variable(torch.zeros(self.num_layers, batch_size, cfg.MODEL.RNN_SIZE).to(self.device)),
                Variable(torch.zeros(self.num_layers, batch_size, cfg.MODEL.RNN_SIZE).to(self.device))]        


    def make_kwargs(self, wt, gv_feat_cross, cross_feats, p_cross_feats, cross_mask, state, **kgs):
        kwargs = kgs
        kwargs[cfg.PARAM.WT] = wt
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat_cross
        kwargs[cfg.PARAM.CROSS_FEATS] = cross_feats
        kwargs[cfg.PARAM.CROSS_FEATS_MASK] = cross_mask
        kwargs[cfg.PARAM.P_CROSS_FEATS] = p_cross_feats
 
        
        kwargs[cfg.PARAM.STATE] = state
        return kwargs

    def preprocess(self, decode = False, **kwargs):
        #gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT] ##here change gv_feat to s3d_feats
        
        if(decode == True):
            expand_num = 1
        else:
            expand_num = cfg.DATA_LOADER.SEQ_PER_IMG
        
        
        s3d_feats =  kwargs[cfg.PARAM.S3D_FEATS]
        s3d_mask = kwargs[cfg.PARAM.S3D_FEATS_MASK]
        
        res152_feats = kwargs[cfg.PARAM.RES152_FEATS]
        res152_mask = kwargs[cfg.PARAM.RES152_FEATS_MASK]
        
        tag_ids = kwargs[cfg.PARAM.TAG_IDS]
        tag_mask = kwargs[cfg.PARAM.TAG_MASK].type(torch.float)
        
        input_ids = kwargs[cfg.PARAM.INPUT_IDS]
        input_mask = (input_ids>0).type(torch.float)
        
        
        #modified gv_feats
        gv_feat_ori = torch.zeros([s3d_feats.shape[0],1]).float()#.cuda()
        
        # embed RES152_FEATS
        if self.res152_embed is not None:    
            res152_feats = self.res152_embed(res152_feats)
            if cfg.MODEL.IF_POS:
                seq_length = res152_feats.size(1)
                position_ids = torch.arange(seq_length, dtype=torch.long, device=res152_feats.device)
                position_ids = position_ids.unsqueeze(0).expand_as(res152_feats[:,:,0])
                res152_feats = res152_feats + self.pos_emb_res152(position_ids)
        
        if self.s3d_embed is not None:    
            s3d_feats = self.s3d_embed(s3d_feats)        
            if cfg.MODEL.IF_POS:
                seq_length = s3d_feats.size(1)
                position_ids = torch.arange(seq_length, dtype=torch.long, device=s3d_feats.device)
                position_ids = position_ids.unsqueeze(0).expand_as(s3d_feats[:,:,0])
                s3d_feats = s3d_feats + self.pos_emb_s3d(position_ids)
        #p_res152_feats = self.p_res152_feats(res152_feats) if self.p_res152_feats is not None else None
        
        # bilinear
        if cfg.MODEL.BILINEAR.DIM > 0:
            gv_feat, res152_feats = self.encoder_layers_res152(gv_feat_ori, res152_feats, res152_mask)
            gv_feat_s3d, s3d_feats = self.encoder_layers_s3d(gv_feat_ori, s3d_feats, s3d_mask)

            tag_feats = self.tag_embed(tag_ids)
            if cfg.MODEL.IF_POS:
                seq_length = tag_feats.size(1)
                position_ids = torch.arange(seq_length, dtype=torch.long, device=tag_feats.device)
                position_ids = position_ids.unsqueeze(0).expand_as(tag_feats[:,:,0])
                tag_feats = tag_feats + self.pos_emb_tag(position_ids)
            gv_feat_tag, tag_feats = self.encoder_layers_tags(gv_feat_ori, tag_feats, tag_mask)

            if(decode == True):
                input_mask = input_mask[:res152_feats.shape[0]]
                #encoded_layers, _ = self.bert(input_ids[:res152_feats.shape[0]], token_type_ids=None, attention_mask=input_mask)
                #language_feats = encoded_layers[-1]
                
                language_feats = self.id_embed(input_ids[:res152_feats.shape[0]])
                if cfg.MODEL.IF_POS:
                    seq_length = language_feats.size(1)
                    position_ids = torch.arange(seq_length, dtype=torch.long, device=language_feats.device)
                    position_ids = position_ids.unsqueeze(0).expand_as(language_feats[:,:,0])
                    language_feats = language_feats + self.pos_emb_id(position_ids)
                gv_feat_id, language_feats = self.encoder_layers_ids(gv_feat_ori, language_feats, input_mask)

            else:
                #encoded_layers, _ = self.bert(input_ids, token_type_ids=None, attention_mask=input_mask)
                #language_feats = encoded_layers[-1]
                language_feats = self.id_embed(input_ids)
                if cfg.MODEL.IF_POS:
                    seq_length = language_feats.size(1)
                    position_ids = torch.arange(seq_length, dtype=torch.long, device=language_feats.device)
                    position_ids = position_ids.unsqueeze(0).expand_as(language_feats[:,:,0])
                    language_feats = language_feats + self.pos_emb_id(position_ids)
                gv_feat_id, language_feats = self.encoder_layers_ids(torch.zeros([language_feats.shape[0],1]).float(), language_feats, input_mask)
                
                res152_feats = utils.expand_tensor(res152_feats, expand_num)
                res152_mask = utils.expand_tensor(res152_mask, expand_num)
       
                s3d_feats = utils.expand_tensor(s3d_feats, expand_num)
                s3d_mask = utils.expand_tensor(s3d_mask, expand_num)
                
                tag_feats = utils.expand_tensor(tag_feats, expand_num)
                tag_mask = utils.expand_tensor(tag_mask, expand_num)
            
            cat_feats = torch.cat([language_feats, tag_feats, res152_feats, s3d_feats], dim=1)
            cat_mask = torch.cat([input_mask, tag_mask, res152_mask, s3d_mask], dim = 1)
            
            gv_feat_cross, cross_feats = self.encoder_layers_cross(gv_feat_ori, cat_feats, cat_mask)
            keys, value2s = self.attention_cross.precompute(cross_feats, cross_feats)
            p_cross_feats = torch.cat([keys, value2s], dim=-1)            
            
        return gv_feat_cross, cross_feats, p_cross_feats, cat_mask

    # gv_feat -- batch_size * cfg.MODEL.I3DFEAT_DIM
    # RES152_FEATS -- batch_size * att_num * RES152_FEATS_dim
    def _calculate_mlm_loss(self, sequence_output_alm, pairs_token_labels):
        alm_scores = self.mlm_head(sequence_output_alm)
        alm_scores = alm_scores.view(-1, self.vocab_size)
        pairs_token_labels = pairs_token_labels.view(-1)
        mask = pairs_token_labels >= 0
        acc = accuracy(alm_scores[mask].detach(), pairs_token_labels[mask].detach())
        alm_loss = self.alm_loss_fct(alm_scores, pairs_token_labels)
        return alm_loss, acc
    
    def forward_pretrain(self, **kwargs):
        loss = 0.
        aligned = kwargs[cfg.PARAM.ALIGNED]
        aligned = aligned.view(-1)
        gv_feat_cross, cross_feats, p_cross_feats, cross_mask = self.preprocess(decode = False, **kwargs)
        language_cross_feats, tag_cross_feats, res152_cross_feats, s3d_cross_feats = \
                torch.split(cross_feats, [kwargs[cfg.PARAM.INPUT_IDS].shape[1], kwargs[cfg.PARAM.TAG_IDS].shape[1], \
                                          kwargs[cfg.PARAM.RES152_FEATS].shape[1], kwargs[cfg.PARAM.S3D_FEATS].shape[1]], dim=1)        

        align_score = self.align_dense(language_cross_feats[:, 0, :])
        match_loss = torch.nn.functional.cross_entropy(align_score, aligned)
        acc_match = accuracy(align_score.detach(), aligned.detach())
        loss += match_loss

        token_labels = kwargs[cfg.PARAM.TOKEN_LABELS_ALL]
        alm_loss, acc = self._calculate_mlm_loss(language_cross_feats[aligned==1], token_labels[aligned==1])

        if token_labels[aligned == 1].shape[0] != 0:
            loss += alm_loss

        loss_info = {'loss_pretrain': loss.item(), 'acc_match': acc_match.item(), 'acc_mlm': acc.item()}
        # loss_info = {'loss_pretrain': loss.item(), 'acc_match': match_loss.item(), 'acc_mlm': alm_loss.item()}
        return loss, loss_info
    
    def forward(self, **kwargs): 
        seq = kwargs[cfg.PARAM.INPUT_SENT] 
        gv_feat_cross, cross_feats, p_cross_feats, cross_mask = self.preprocess(decode = False , **kwargs)
        
        
        batch_size = gv_feat_cross.size(0)
        state = self.init_hidden(batch_size)

        outputs = Variable(torch.zeros(batch_size, seq.size(1), self.vocab_size).to(self.device))
        for t in range(seq.size(1)):
            if self.training and t >=1 and self.ss_prob > 0:
                prob = torch.empty(batch_size).to(self.device).uniform_(0, 1)
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
            
            kwargs = self.make_kwargs(wt, gv_feat_cross, cross_feats, p_cross_feats, cross_mask, state)
            output, state = self.Forward(**kwargs)
            if self.dropout_lm is not None:
                output = self.dropout_lm(output)

            logit = self.logit(output)
            outputs[:, t] = logit

        return outputs

    def get_logprobs_state(self, **kwargs):
        output, state = self.Forward(**kwargs)
        logprobs = F.log_softmax(self.logit(output), dim=1)
        return logprobs, state

    def _expand_state(self, batch_size, beam_size, cur_beam_size, state, selected_beam):
        shape = [int(sh) for sh in state.shape]
        beam = selected_beam
        for _ in shape[2:]:
            beam = beam.unsqueeze(-1)
        beam = beam.unsqueeze(0)
        
        state = torch.gather(
            state.view(*([shape[0], batch_size, cur_beam_size] + shape[2:])), 2,
            beam.expand(*([shape[0], batch_size, beam_size] + shape[2:]))
        )
        state = state.view(*([shape[0], -1, ] + shape[2:]))
        return state

    # the beam search code is inspired by https://github.com/aimagelab/meshed-memory-transformer
    def decode_beam(self, **kwargs):
        gv_feat_cross, cross_feats, p_cross_feats, cross_mask = self.preprocess(decode= True, **kwargs)
        
        beam_size = kwargs['BEAM_SIZE']
        batch_size = cross_feats.size(0)
        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()

        state = self.init_hidden(batch_size)
        wt = Variable(torch.ones(batch_size, dtype=torch.long).cuda())

        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat_cross
        kwargs[cfg.PARAM.CROSS_FEATS] = cross_feats
        kwargs[cfg.PARAM.CROSS_FEATS_MASK] = cross_mask
        kwargs[cfg.PARAM.P_CROSS_FEATS] = p_cross_feats

        
        outputs = []
        for t in range(cfg.MODEL.SEQ_LEN):
            cur_beam_size = 1 if t == 0 else beam_size

            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            word_logprob, state = self.get_logprobs_state(**kwargs)
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

            for s in range(len(state)):
                state[s] = self._expand_state(batch_size, beam_size, cur_beam_size, state[s], selected_beam)

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
                cross_feats = utils.expand_tensor(cross_feats, beam_size)
                gv_feat_cross = utils.expand_tensor(gv_feat_cross, beam_size)
                cross_mask = utils.expand_tensor(cross_mask, beam_size)
                p_cross_feats = utils.expand_tensor(p_cross_feats, beam_size)
                
                kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat_cross
                kwargs[cfg.PARAM.CROSS_FEATS] = cross_feats
                kwargs[cfg.PARAM.CROSS_FEATS_MASK] = cross_mask
                kwargs[cfg.PARAM.P_CROSS_FEATS] = p_cross_feats              
                
                                
                
        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        return outputs, log_probs

    # For the experiments of X-LAN, we use the following beam search code, 
    # which achieves slightly better results but much slower.
    
    def decode(self, **kwargs):
        greedy_decode = kwargs['GREEDY_DECODE']
 
        gv_feat_cross, cross_feats, p_cross_feats, cross_mask = self.preprocess(decode = greedy_decode, **kwargs)
        batch_size = gv_feat_cross.size(0)
        state = self.init_hidden(batch_size)

        sents = Variable(torch.zeros((batch_size, cfg.MODEL.SEQ_LEN), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(batch_size, cfg.MODEL.SEQ_LEN).cuda())
        wt = Variable(torch.ones(batch_size, dtype=torch.long).cuda())
        unfinished = wt.eq(wt)
        for t in range(cfg.MODEL.SEQ_LEN):
            kwargs = self.make_kwargs(wt, gv_feat_cross, cross_feats, p_cross_feats, cross_mask, state)
            logprobs_t, state = self.get_logprobs_state(**kwargs)
            
            if greedy_decode:
                logP_t, wt = torch.max(logprobs_t, 1)
            else:
                probs_t = torch.exp(logprobs_t)
                wt = torch.multinomial(probs_t, 1)
                logP_t = logprobs_t.gather(1, wt)
            wt = wt.view(-1).long()
            unfinished = unfinished * (wt != 2)
            wt = wt * unfinished.type_as(wt)
            sents[:,t] = wt
            logprobs[:,t] = logP_t.view(-1)

            if unfinished.sum() == 0:
                break
        return sents, logprobs


def init_preweight(model, state_dict, prefix=None, task_config=None):
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if 'gamma' in key:
            new_key = key.replace('gamma', 'weight')
        if 'beta' in key:
            new_key = key.replace('beta', 'bias')
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    if prefix is not None:
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            old_keys.append(key)
            new_keys.append(prefix + key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='')

    if prefix is None and (task_config is None or task_config.local_rank == 0):
        logger.info("-" * 20)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}"
                        .format(model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}"
                        .format(model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)))
        if len(error_msgs) > 0:
            logger.error("Weights from pretrained model cause errors in {}: {}"
                         .format(model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)))
    return model

def accuracy(pred, target, topk=1):
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        if pred.size(0) != 0:
            res.append(correct_k.mul_(100.0 / pred.size(0)))
        else:
            res.append(correct_k.mul_(100.0))
    return res[0] if return_single else res
