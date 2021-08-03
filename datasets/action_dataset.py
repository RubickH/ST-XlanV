import os
import random
import numpy as np
import torch
import torch.utils.data as data
import lib.utils as utils
import pickle
#added in 2021
from modules.tokenization import BertTokenizer, WordTokenizer
#from pytorch_pretrained_bert import BertModel, BertTokenizer
import json

class ActionDataset(data.Dataset):
    def __init__(
        self,
        image_ids_path, 
        input_seq, 
        s3d_feats_path, 
        s3d_logits_path, 
        res152_feats_path, 
        res152_logits_path, 
        seq_per_img,
        max_feats_num,
        pretrain=False,
    ):
        self.max_feats_num = max_feats_num
        self.seq_per_img = seq_per_img
        assert self.seq_per_img == 1
        # self.image_ids = utils.load_lines(image_ids_path)
        self.res152_feats_path = res152_feats_path if len(res152_feats_path) > 0 else None
        self.gv_feats = None#pickle.load(open(s3d_feats_path, 'rb'), encoding='bytes') if len(s3d_feats_path) > 0 else None
        
        #added in 2021
        self.s3d_feats_path = s3d_feats_path if len(s3d_feats_path) > 0 else None
        self.s3d_logits_path = s3d_logits_path if len(s3d_logits_path) > 0 else None
        self.res152_logits_path = res152_logits_path if len(res152_logits_path) > 0 else None
        
        if input_seq is not None:
            self.input_seq = json.load(open(input_seq))
            self.seq_len = 20
        else:
            self.seq_len = 20
            self.input_seq = None
            self.target_seq = None
        
        #added in 2021
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.tokenizer = WordTokenizer("./data_msrvtt/msrvtt_vocab.txt")
        self.max_len_tag = 11
        self.s3d_names = [c.strip() for c in open('./data_common/s3d_label.txt')]
        self.imagenet_names = json.load(open("./data_common/imagenet-simple-labels.json"))          
        self.pre_train = pretrain
        
    def set_seq_per_img(self, seq_per_img):
        self.seq_per_img = seq_per_img

    def __len__(self):
        return len(self.input_seq)



    def _mask_tokens(self, words):
        token_labels = []
        masked_tokens = words.copy()

        for token_id, token in enumerate(masked_tokens):
            if token_id == 0 or token_id == len(masked_tokens) - 1:
                token_labels.append(-1)
                continue
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                if prob < 0.8:
                    masked_tokens[token_id] = "[MASK]"
                elif prob < 0.9:
                    masked_tokens[token_id] = random.choice(list(self.tokenizer.vocab.items()))[0]
                try:
                    token_labels.append(self.tokenizer.vocab[token])
                except KeyError:
                    token_labels.append(self.tokenizer.vocab["[UNK]"])
            else:
                token_labels.append(-1)

        return masked_tokens, token_labels
    
    def __getitem__(self, index):
        indices = np.array([index]).astype('int')

        #if self.gv_feats is not None:
            #gv_feats = self.gv_feat[image_id]
            #gv_feats = np.array(gv_feat).astype('float32')
        #else:
            #gv_feats = np.zeros((1,1))

        if self.res152_feats_path is not None:
            res152_feats = np.load(os.path.join(self.res152_feats_path, str(index) + '.npy'))
            res152_feats = np.array(res152_feats).astype('float32')
        else:
            res152_feats = np.zeros((1,1))
            
        if self.s3d_feats_path is not None:
            s3d_feats = np.load(os.path.join(self.s3d_feats_path, str(index) + '.npy'))
            s3d_feats = np.array(s3d_feats).astype('float32')
        else:
            s3d_feats = np.zeros((1,1))
        
              
        if self.max_feats_num > 0 and res152_feats.shape[0] > self.max_feats_num:
            res152_feats = res152_feats[:self.max_feats_num, :]


        s3d_logits = np.load(os.path.join(self.s3d_logits_path, str(index) + '.npy'))
        res152_logits = np.load(os.path.join(self.res152_logits_path, str(index) + '.npy'))
            
        res152_logits = np.array(res152_logits).astype('float32')
        s3d_logits = np.array(s3d_logits).astype('float32')

        s3d_logits = s3d_logits.mean(0)
        res_logits = res152_logits.mean(0)
        s3d_sort_idx = np.argsort(s3d_logits)[::-1][:5]
        res_sort_idx = np.argsort(res_logits)[::-1][:5]
        action_tag = self.s3d_names[s3d_sort_idx[0]]
        content_tag = self.imagenet_names[res_sort_idx[0]]

        prob = random.random()
        if self.pre_train and prob < 0.5:
            action_tag = self.s3d_names[np.random.randint(0, len(self.s3d_names))]
            aligned_act = 0
        else:
            aligned_act = 1

        prob = random.random()
        if self.pre_train and prob < 0.5:
            content_tag = self.imagenet_names[np.random.randint(0, len(self.imagenet_names))]
            aligned_con = 0
        else:
            aligned_con = 1
                
        tags = ",".join([content_tag, action_tag])
        tags = self.tokenizer.tokenize(tags)
        tags = tags[:self.max_len_tag-1] + ["[SEP]"]
        tag_ids = self.tokenizer.convert_tokens_to_ids(tags)
        tag_mask = [1] * len(tag_ids)
        while len(tag_ids) < self.max_len_tag:
            tag_ids.append(0)
            tag_mask.append(0)
        assert len(tag_ids) == self.max_len_tag
        assert len(tag_mask) == self.max_len_tag        
       
        tag_ids = np.array(tag_ids)
        tag_mask = np.array(tag_mask)

       
        input_ids_all = np.zeros((self.seq_per_img, self.seq_len), dtype='int')
        input_masks_all = np.zeros((self.seq_per_img, self.seq_len), dtype='int')
        input_seq = np.zeros((self.seq_per_img, self.seq_len), dtype='int')
        target_seq = np.zeros((self.seq_per_img, self.seq_len), dtype='int')
        token_labels_all = np.zeros((self.seq_per_img, self.seq_len), dtype='int')
        aligned_all_act = []
        aligned_all_con = []
        
        for i in range(self.seq_per_img):
            if(self.input_seq is not None):
                sentences = self.input_seq[str(index)]["sentence"]
                sentence = random.choice(sentences)
                sentence = self.tokenizer.tokenize(sentence)
            else:
                sentence = []
            
            if(self.pre_train == True):
                words = sentence
            else:
                words = []
          
            words = ["[CLS]"] + words
            total_length_with_CLS = self.seq_len - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + ["[SEP]"]

            masked_tokens, token_labels = self._mask_tokens(words)
    
            input_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
            input_mask = [1] * len(input_ids)

            while len(input_ids) < self.seq_len:
                input_ids.append(0)
                input_mask.append(0)
                token_labels.append(-1)

            assert len(input_ids) == self.seq_len
            assert len(input_mask) == self.seq_len
            assert len(token_labels) == self.seq_len
            
            if len(sentence) > total_length_with_CLS:
                sentence = sentence[:total_length_with_CLS]                
            input_caption_words = ["[CLS]"] + sentence
            output_caption_words = sentence + ["[SEP]"]
    
            # For generate captions
            input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
            output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)

            while len(input_caption_ids) < self.seq_len:
                input_caption_ids.append(0)
                output_caption_ids.append(0)

            assert len(input_caption_ids) == self.seq_len
            assert len(output_caption_ids) == self.seq_len
            
            input_seq[i] = input_caption_ids
            target_seq[i] = output_caption_ids
            token_labels_all[i] = token_labels
            input_ids_all[i] = input_ids
            input_masks_all[i] = input_mask
            aligned_all_act.append(aligned_act)
            aligned_all_con.append(aligned_con)

        if(not self.pre_train):
            input_ids_all = input_ids_all[:, :2]
            input_masks_all = input_masks_all[:, :2]
            
        return indices, input_ids_all, input_seq, target_seq, s3d_feats, res152_feats, tag_ids, tag_mask, input_masks_all, token_labels_all, aligned_all_act, aligned_all_con
