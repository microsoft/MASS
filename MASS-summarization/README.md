# MASS-SUM

<!---
The source sentence in summarization tasks is usually long. To handle the long sequence, we use document-level corpus to extract long-continuous sequence for pre-training. The max sequence length is set as 512 for each iteration. For each sequence, we randomly mask a continuous segment for every 64-tokens at the encoder and predict it at the decoder. 
-->

## Dependency
```
pip install torch==1.0.0 
pip install fairseq==0.8.0
```

## MODEL
MASS uses default Transformer structure. We denote L, H, A as the number of layers, the hidden size and the number of attention heads. 

| Model | Encoder | Decoder | Download |
| :------| :-----|:-----|:-----|
| MASS-base-uncased | 6L-768H-12A | 6L-768H-12A | [MODEL](https://modelrelease.blob.core.windows.net/mass/mass-base-uncased.tar.gz) | 

## Results on Abstractive Summarization (9/27/2019)

| Dataset | Params | RG-1 | RG-2 | RG-L |
| ------| -----  | ---- | ---- | ---- |
| CNN/Daily Mail | 123M | 42.12 | 19.50 | 39.01 |  
| Gigaword | 123M | 38.73 | 19.71| 35.96 |
| XSum | 123M | 39.75 | 17.24 | 31.95 |

Evaluated by [files2rouge](https://github.com/pltrdy/files2rouge). 

## Pipeline for Pre-Training
### Download data
Our model is trained on Wikipekia + BookCorpus. Here we use wikitext-103 to demonstrate how to process data.
```
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip
```

### Tokenize corpus
We use wordpiece vocabuary (from bert) to tokenize the original text data directly. We provide a [script](encode.py) to deal with data. You need to `pip install pytorch_transformers` first to generate tokenized data. 
```
mkdir -p mono
for SPLIT in train valid test; do 
    python encode.py \
        --inputs wikitext-103-raw/wiki.${SPLIT}.raw \
        --outputs mono/${SPLIT}.txt \
        --workers 60; \
done 
```

### Binarized data
```
wget -c https://modelrelease.blob.core.windows.net/mass/mass-base-uncased.tar.gz
tar -zxvf mass-base-uncased.tar.gz
# Move dict.txt from tar file to the data directory 

fairseq-preprocess \
    --user-dir mass --only-source --task masked_s2s \
    --trainpref mono/train.txt --validpref mono/valid.txt --testpref mono/test.txt \
    --destdir processed --srcdict dict.txt --workers 60
```

### Pre-training
```
TOKENS_PER_SAMPLE=512
WARMUP_UPDATES=10000
PEAK_LR=0.0005
TOTAL_UPDATES=125000
MAX_SENTENCES=8
UPDATE_FREQ=16

fairseq-train processed \
    --user-dir mass --task masked_s2s --arch transformer_mass_base \
    --sample-break-mode none \
    --tokens-per-sample $TOKENS_PER_SAMPLE \
    --criterion masked_lm \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --ddp-backend=no_c10d \
```
## Pipeline for Fine-tuning (CNN / Daily Mail)

### Data 
Download, tokenize and truncate data from this [link](https://github.com/abisee/cnn-dailymail), and use the above [tokenization](#tokenize-corpus) to generate wordpiece-level data. Rename the shuffix `article` and `title` as `src` and `tgt`. Assume the tokenized data is under `cnndm/para`
```
fairseq-preprocess \
    --user-dir mass --task masked_s2s \
    --source-lang src --target-lang tgt \
    --trainpref cnndm/para/train --validpref cnndm/para/valid --testpref cnndm/para/test \
    --destdir cnndm/processed --srcdict dict.txt --tgtdict dict.txt \
    --workers 20
```
`dict.txt` is included in `mass-base-uncased.tar.gz`. A copy of binarized data can be obtained from [here](https://modelrelease.blob.core.windows.net/mass/cnndm.tar.gz).


### Running
```
fairseq-train cnndm/processed/ \
    --user-dir mass --task translation_mass --arch transformer_mass_base \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --min-lr 1e-09 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --update-freq 8 --max-tokens 4096 \
    --ddp-backend=no_c10d --max-epoch 25 \
    --max-source-positions 512 --max-target-positions 512 \
    --skip-invalid-size-inputs-valid-test \
    --load-from-pretrained-model mass-base-uncased.pt \
```
`lr=0.0005` is not the optimal choice for any task. It is tuned on the dev set (among 1e-4, 2e-4, 5e-4). 
### Inference
```
MODEL=checkpoints/checkpoint_best.pt
fairseq-generate $DATADIR --path $MODEL \
    --user-dir mass --task translation_mass \
    --batch-size 64 --beam 5 --min-len 50 --no-repeat-ngram-size 3 \
    --lenpen 1.0 \
```
`min-len` is sensitive for different tasks, `lenpen` needs to be tuned on the dev set. Restore the results to the word-level data by using `sed 's/ ##//g'`.


## Other questions
1. Q: I have met error like `ModuleNotFouldError: No module named 'mass'` in multi-GPUs or multi-nodes, how to solve it?   
   A: It seems like a bug in python `multiprocessing/spawn.py`. A direct solution is to move these three files to its corresponding folder in the fairseq. For example:
```
  mv bert_dictionary.py fairseq/fairseq/data/
  mv masked_dataset.py fairseq/fairseq/data/
  mv learned_positional_embedding.py fairseq/fairseq/modules/
  modify fairseq/fairseq/data/__init__.py to import the above files.
```

<!---
## Training Details 

`MASS-base-uncased` uses 32x NVIDIA 32GB V100 GPUs and trains on (Wikipekia + BookCorpus, 16GB) for 20 epochs (float32), batch size is simulated as 4096.

## Other questions
> 1. Q: When i run this program in multi-gpus or multi-nodes, the program reports errors like `ModuleNotFouldError: No module named 'mass'`.    
  A: This seems a bug in python `multiprocessing/spawn.py`, a direct solution is to move these files into each relative folder under fairseq. Do not forget to modify the import path in the code.
-->
