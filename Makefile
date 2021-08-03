################# Tools and config ####################
file_parts=train

BPE_TOKENS=40000

ROOT=/home/v-yiqhuang/mycontainer/v-yiqhuang/pre_process_files
FAST_BPEROOT=$(ROOT)/fastBPE
BPEROOT=$(ROOT)/subword-nmt/subword_nmt
MOSES_ROOT=$(ROOT)/mosesdecoder
FASTALIGN_ROOT=$(ROOT)/fastalign
FASTALIGN=$(FASTALIGN_ROOT)/build
SCRIPTS=$(MOSES_ROOT)/scripts

REPLACE_UNICODE_PUNCT=$(SCRIPTS)/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$(SCRIPTS)/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$(SCRIPTS)/tokenizer/remove-non-printing-char.perl
TOKENIZER=$(SCRIPTS)/tokenizer/tokenizer.perl
TOK_AND_GEN_NE=$(ROOT)/script/tok_and_gen_ne.py
ASSING_NE=$(ROOT)/script/assign_ne.py
EXTRACT_NE=$(ROOT)/script/extract_ne.py
BERTFY_ENTITY=$(ROOT)/script/bertfy_entity.py
CLEAN=$(ROOT)/script/filter.py
JIEBA=$(ROOT)/script/zh_jieba.py
BERT_ENTITY=$(ROOT)/script/extract_and_bertfy_entity.py
EXTRACT_ALL_ALIGN=$(ROOT)/script/extract_all_ne_align.py

$(BPEROOT):
	git clone https://github.com/rsennrich/subword-nmt.git $(BPEROOT)

$(MOSES_ROOT):
	git clone https://github.com/moses-smt/mosesdecoder.git $(MOSES_ROOT)

$(FAST_BPEROOT):
	git clone https://github.com/glample/fastBPE.git $(FAST_BPEROOT)
	cd $(FAST_BPEROOT) && g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast

$(FASTALIGN_ROOT):
	git clone https://github.com/clab/fast_align.git $(FASTALIGN_ROOT)
	mkdir $(FASTALIGN_ROOT)/build
	cd $(FASTALIGN_ROOT)/build && cmake .. && make
.PHONY: tools DeepPavlov
tools: $(BPEROOT) $(MOSES_ROOT) DeepPavlov $(FAST_BPEROOT) $(FASTALIGN_ROOT)
DeepPavlov:
	python -m deeppavlov install 'ner_ontonotes_bert_mult'

################# Download data ####################

%.processed.en: %.filtered.en
	cat $< | perl $(REPLACE_UNICODE_PUNCT) | perl $(NORM_PUNC) -l en | perl $(REM_NON_PRINT_CHAR) > $@

%.processed.de: %.filtered.de
	cat $< | perl $(REPLACE_UNICODE_PUNCT) | perl $(NORM_PUNC) -l de | perl $(REM_NON_PRINT_CHAR)  > $@

.PHONY: process_all
process_all: $(file_parts:%=%.processed.en) $(file_parts:%=%.processed.de)
	     wc -l $^

#################### Moses tokenzie #################
%.mtok.en: %.processed.en
	        perl $(TOKENIZER) -threads "$(shell nproc)" -no-escape -l en < $< > $@

%.mtok.de: %.processed.de
	        perl $(TOKENIZER) -threads "$(shell nproc)" -no-escape -l de < $< > $@

.PHONY: mtok_all
mtok_all: $(file_parts:%=%.mtok.en) $(file_parts:%=%.mtok.de)
	        wc -l $^


#################### Train and apply BPE ###########################
train.bpe.%: train.mtok.% BPE_files/result_length/BPE_F1_aligned_1.04.src
	$(FAST_BPEROOT)/fast applybpe $@ $^

valid.bpe.%: valid.mtok.% BPE_files/result_length/BPE_F1_aligned_1.04.src
	$(FAST_BPEROOT)/fast applybpe $@ $^

test.bpe.%: test.mtok.% BPE_files/result_length/BPE_F1_aligned_1.04.src
	$(FAST_BPEROOT)/fast applybpe $@ $^

.PHONY: bpe_all
bpe_all: $(file_parts:%=%.bpe.en) $(file_parts:%=%.bpe.de)
	#cat train.bpe.en valid.bpe.en > train.bpe.all.en
	#cat train.bpe.de valid.bpe.de > train.bpe.all.de
	wc -l $^

BPE.en:train.mtok.en train.mtok.de
	$(FAST_BPEROOT)/fast learnbpe $(BPE_TOKENS) $^ > $@
BPE.de: BPE.en
	cp $< $@

train.bpe.ori.%: train.mtok.% BPE.%
	$(FAST_BPEROOT)/fast applybpe $@ $^

valid.bpe.ori.%: valid.mtok.% BPE.%
	$(FAST_BPEROOT)/fast applybpe $@ $^

test.bpe.ori.%: test.mtok.% BPE.%
	$(FAST_BPEROOT)/fast applybpe $@ $^

.PHONY: bpe_code_all bpe_all_ori
bpe_code_all: BPE.en BPE.de
bpe_ori_all: $(file_parts:%=%.bpe.ori.en) $(file_parts:%=%.bpe.ori.de)
	wc -l $^

################### Make Fair-Seq bin ############################

data-bin-102/:
	fairseq-preprocess --source-lang en --target-lang de \
		--trainpref train.bpe --validpref test.bpe --testpref test.bpe \
		--destdir $@ \
		--workers $(shell nproc)\
	        --joined-dictionary \
 
data-bin-104-dataset/:
	fairseq-preprocess --source-lang en --target-lang de \
		--testpref train.bpe \
		--destdir $@ \
		--workers $(shell nproc)\
	        --srcdict ./data-bin-104/dict.en.txt\
                --tgtdict ./data-bin-104/dict.de.txt 

data-bin/:
	fairseq-preprocess --source-lang en --target-lang de \
		--trainpref train.bpe.ori --validpref test.bpe.ori --testpref test.bpe.ori \
		--destdir $@ \
		--workers $(shell nproc)\
	        --joined-dictionary


################  Extract entity #################################
%.extracted_entity.bpe: train.bpe.% train.bpe.%.ne
	python $(EXTRACT_NE) $^ $@

en-zh.extracted_ne_align: train.bpe.en train.bpe.en.ne train.bpe.zh train.bpe.zh.ne train.fast_align
	python $(EXTRACT_ALL_ALIGN) $^ $@

zh-en.extracted_ne_align: en-zh.extracted_ne_align
	awk '-F' '\t' '{print $$1 "\t" $$3 "\t" $$2}' $< > $@

zh.external_entity.bpe: BPE.zh
	cut -f 1 $(ROOT)/zh-en.entity.txt | python $(JIEBA) | $(FAST_BPEROOT)/fast applybpe_stream $< > $@

en.external_entity.bpe: BPE.en
	cut -f 2 $(ROOT)/zh-en.entity.txt | $(FAST_BPEROOT)/fast applybpe_stream $< > $@

zh-en.external_entity.mapping.bpe: zh.external_entity.bpe en.external_entity.bpe
	paste $^ > $@

en-zh.external_entity.mapping.bpe: en.external_entity.bpe zh.external_entity.bpe
	paste $^ > $@

en-zh.%.ent.bpe : %.extracted_entity.bpe %.external_entity.bpe
	python $(BERTFY_ENTITY) $@ en-zh.$*.ent_emb $^

en-zh.%.ent_emb.npy: en-zh.%.ent.bpe
	echo "BERT"

fast_align_input: train.bpe.en train.bpe.zh # Is de ok to use BPE? or we need to use mtok data
	paste $^ | sed 's/ *\t */ ||| /g' > $@

train.fast_align:	fast_align_input
	$(FASTALIGN)/fast_align -i $< -d -o -v > forward.align
	$(FASTALIGN)/fast_align -i $< -d -o -v -r > reverse.align
	$(FASTALIGN)/atools -i forward.align -j reverse.align -c grow-diag-final-and > $@


.PHONY: extract_entity_all extract_entity_mapping extract_entity_emb
extract_entity_mapping: zh-en.external_entity.mapping.bpe en-zh.external_entity.mapping.bpe en-zh.extracted_ne_align zh-en.extracted_ne_align
extract_entity_emb: en-zh.en.ent_emb.npy en-zh.zh.ent_emb.npy en-zh.zh.ent.bpe en-zh.en.ent.bpe
extract_entity_all: extract_entity_mapping extract_entity_emb
