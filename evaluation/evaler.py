import os
import sys
import numpy as np
import torch
import tqdm
import json
import evaluation
import lib.utils as utils
import datasets.data_loader as data_loader
from lib.config import cfg
from modules.tokenization import BertTokenizer, WordTokenizer

class Evaler(object):
    def __init__(
        self,
        eval_ids,
        s3d_feats,
        s3d_logits,
        res152_feats,
        res152_logits,
        eval_annfile
    ):
        super(Evaler, self).__init__()
        self.vocab = utils.load_vocab(cfg.INFERENCE.VOCAB)

        self.eval_ids = np.array(utils.load_ids(eval_ids))
        self.eval_loader = data_loader.load_val(eval_ids, s3d_feats, s3d_logits, res152_feats, res152_logits)
        self.evaler = evaluation.create(cfg.INFERENCE.EVAL, eval_annfile)
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.tokenizer = WordTokenizer("./data_msrvtt/msrvtt_vocab.txt")
        
    def make_kwargs(self, indices, input_ids_all, s3d_feats, s3d_mask, res152_feats, res152_mask, tag_ids, tag_mask):
        kwargs = {}
        kwargs[cfg.PARAM.INDICES] = indices
        kwargs[cfg.PARAM.INPUT_IDS] = input_ids_all
        kwargs[cfg.PARAM.S3D_FEATS] = s3d_feats
        kwargs[cfg.PARAM.S3D_FEATS_MASK] = s3d_mask
        kwargs[cfg.PARAM.RES152_FEATS] = res152_feats
        kwargs[cfg.PARAM.RES152_FEATS_MASK] = res152_mask            
        kwargs[cfg.PARAM.TAG_IDS] = tag_ids
        kwargs[cfg.PARAM.TAG_MASK] = tag_mask  
        kwargs['BEAM_SIZE'] = cfg.INFERENCE.BEAM_SIZE
        kwargs['GREEDY_DECODE'] = cfg.INFERENCE.GREEDY_DECODE
        
        return kwargs
        
    def __call__(self, model, rname):
        model.eval()
        
        results = []
        with torch.no_grad():
            for _, (indices, input_ids_all, input_seq, target_seq, s3d_feats, s3d_mask, res152_feats, res152_mask, tag_ids, tag_mask, aligned_act, aligned_con) in tqdm.tqdm(enumerate(self.eval_loader)):
                ids = self.eval_ids[indices]
                input_ids_all = input_ids_all.cuda()
                s3d_feats = s3d_feats.cuda()
                s3d_mask = s3d_mask.cuda()
                res152_feats = res152_feats.cuda()
                res152_mask = res152_mask.cuda()
                tag_ids = torch.from_numpy(np.array(tag_ids)).cuda()
                tag_mask = torch.from_numpy(np.array(tag_mask)).cuda()
                aligned_act = torch.from_numpy(np.array(aligned_act)).cuda()
                aligned_con = torch.from_numpy(np.array(aligned_con)).cuda()
                
                kwargs = self.make_kwargs(indices, input_ids_all, s3d_feats, s3d_mask, res152_feats, res152_mask, tag_ids, tag_mask)
                if kwargs['BEAM_SIZE'] > 1:
                    seq, _ = model.module.decode_beam(**kwargs)
                else:
                    seq, _ = model.module.decode(**kwargs)
                sents = seq.cpu().numpy()
                for sid, sent in enumerate(sents):
                    decode_text_list = self.tokenizer.convert_ids_to_tokens(sent)
                    if "[SEP]" in decode_text_list:
                        SEP_index = decode_text_list.index("[SEP]")
                        decode_text_list = decode_text_list[:SEP_index]
                    if "[PAD]" in decode_text_list:
                        PAD_index = decode_text_list.index("[PAD]")
                        decode_text_list = decode_text_list[:PAD_index]
                    decode_text = ' '.join(decode_text_list)
                    decode_text = decode_text.replace(" ##", "").strip("##").strip()
                                  
                    result = {cfg.INFERENCE.ID_KEY: int(ids[sid]), cfg.INFERENCE.CAP_KEY: decode_text}
                    results.append(result)
        eval_res = self.evaler.eval(results)

        result_folder = os.path.join(cfg.ROOT_DIR, 'result')
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        json.dump(results, open(os.path.join(result_folder, 'result_' + rname +'.json'), 'w'))

        model.train()
        return eval_res
