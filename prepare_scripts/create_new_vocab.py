import json
#msrvtt_vocab = json.load(open('info_alltrain_vlp.json'))['ix_to_word']
#debug = 1
vlp_vocab = json.load(open('vlp_data.json'))['ix_to_word']
debug = 1

all_words = []
#for i in range(len(msrvtt_vocab)):
#    all_words.append(msrvtt_vocab[str(i)])

#all_words = sorted(list(set(all_words)))
for i in range(1,len(vlp_vocab)+1):
    word = vlp_vocab[str(i)]
    all_words.append(word)

all_words = sorted(list(set(all_words)))#[23:]
ix_to_word = {}
word_to_ix = {}

curr_idx = 1
for i in range(len(all_words)):
    if(all_words[i] == '<eos>' or all_words[i] == '<sos>' ):
        continue
    else:
        word = all_words[i]
        idx = curr_idx
        ix_to_word[idx] = word
        word_to_ix[word] = idx
        curr_idx += 1

#ix_to_word[str(len(ix_to_word)+1)] = 'UNK'
#word_to_ix['UNK'] = str(len(word_to_ix)+1)

vocab = {'word_to_ix':word_to_ix,'ix_to_word':ix_to_word}
json.dump(vocab,open('vocab_all.json','w'))

with open('../data/vlp_vocab.txt','w') as f: #vocab_len 11730
    for i in range(1,len(ix_to_word)+1):
        word = ix_to_word[i]+'\n'
        f.write(word)
        