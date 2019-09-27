# MASS with Supervised Pre-training

We implement MASS on [fairseq](https://github.com/pytorch/fairseq), in order to support the pre-training and fine-tuning for large scale supervised tasks, such as neural machine translation, text summarization, grammatical error correction. Unsupervised pre-training usually works better in zero-resource or low-resource downstream tasks. However, there are plenty of supervised data in these tasks, which brings challenges for conventional unsupervised pre-training. Therefore, we design new pre-training loss to support large scale supervised tasks.

We extend the MASS to supervised setting where the supervised sentence pair (X, Y) is leveraged for pre-training. The sentence X is masked and feed into the encoder, and the decoder predicts the whole sentence Y. Some discret tokens in the decoder input are also masked, to encourage the decoder to extract more informaiton from the encoder side.   
![img](archi_mass_sup_md.png)

During pre-training, we combine the orignal MASS pre-training loss and the new supervised pre-training loss together. During fine-tuning, we directly use supervised sentence pairs to fine-tune the pre-trained model.  

MASS on fairseq contains the following codes:
* [Neural Machine Translation](#neural-machine-translation)
* [Text Summarization](#text-summarization)
* [Grammatical Error Correction](#grammatical-error-correction)



## Prerequisites
After download the repository, you need to install `fairseq` by `pip`:
```
pip install fairseq==0.7.1
```


## Neural Machine Translation 

| Languages | Pre-trained Model | BPE codes | English-Dict | Chinese-Dict |
|:-----------:|:-----------------:| :---------:| :------------:| :------------:|
|En - Zh      | [MODEL](https://modelrelease.blob.core.windows.net/mass/zhen_mass_pre-training.pt) | [CODE](https://modelrelease.blob.core.windows.net/mass/bpecode.zip) | [VOCAB](https://modelrelease.blob.core.windows.net/mass/dict.en.txt) | [VOCAB](https://modelrelease.blob.core.windows.net/mass/dict.zh.txt)

We provide an example of how to pre-train and fine-tune on WMT English<->Chinese (En<->Zh) translation.


### Data Ready
We first prepare the monolingual and bilingual sentences for Chinese and English respectively. The data directory looks like:

```
- data/
  ├─ mono/
  |  ├─ train.en
  |  ├─ train.zh
  |  ├─ valid.en
  |  ├─ valid.zh
  |  ├─ dict.en.txt
  |  └─ dict.zh.txt
  └─ para/
     ├─ train.en
     ├─ train.zh
     ├─ valid.en
     ├─ valid.zh
     ├─ dict.en.txt
     └─ dict.zh.txt
```
The files under `mono` are monolingual data, while under `para` are bilingual data. `dict.en(zh).txt` in different directory should be identical. The dictionary for different language can be different. Running the following command can generate the binarized data:

```
# Ensure the output directory exists
data_dir=data/
mono_data_dir=$data_dir/mono/
para_data_dir=$data_dir/para/
save_dir=$data_dir/processed/

# set this relative path of MASS in your server
user_dir=mass

mkdir -p $data_dir $save_dir $mono_data_dir $para_data_dir


# Generate Monolingual Data
for lg in en zh
do

  fairseq-preprocess \
  --task cross_lingual_lm \
  --srcdict $mono_data_dir/dict.$lg.txt \
  --only-source \
  --trainpref $mono_data_dir/train --validpref $mono_data_dir/valid \
  --destdir $save_dir \
  --workers 20 \
  --source-lang $lg

  # Since we only have a source language, the output file has a None for the
  # target language. Remove this

  for stage in train valid
  do
    mv $save_dir/$stage.$lg-None.$lg.bin $save_dir/$stage.$lg.bin
    mv $save_dir/$stage.$lg-None.$lg.idx $save_dir/$stage.$lg.idx
  done
done

# Generate Bilingual Data
fairseq-preprocess \
  --user-dir $mass_dir \
  --task xmasked_seq2seq \
  --source-lang en --target-lang zh \
  --trainpref $para_data_dir/train --validpref $para_data_dir/valid \
  --destdir $save_dir \
  --srcdict $para_data_dir/dict.en.txt \
  --tgtdict $para_data_dir/dict.zh.txt
```


### Pre-training
We provide a simple demo code to demonstrate how to deploy mass pre-training.
```
save_dir=checkpoints/mass/pre-training/
user_dir=mass
data_dir=data/processed/

mkdir -p $save_dir

fairseq-train $data_dir \
    --user-dir $user_dir \
    --save-dir $save_dir \
    --task xmasked_seq2seq \
    --source-langs en,zh \
    --target-langs en,zh \
    --langs en,zh \
    --arch xtransformer \
    --mass_steps en-en,zh-zh \
    --memt_steps en-zh,zh-en \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --lr 0.00005 --min-lr 1e-09 \
    --criterion label_smoothed_cross_entropy \
    --max-tokens 4096 \
    --dropout 0.1 --relu-dropout 0.1 --attention-dropout 0.1 \
    --max-update 100000 \
    --share-decoder-input-output-embed \
    --valid-lang-pairs en-zh \
```
We also provide a pre-training [script](run_mass_enzh.sh) which is used for our released model.

### Fine-tuning
After pre-training stage, we fine-tune the model on bilingual sentence pairs:
```
data_dir=data/processed
save_dir=checkpoints/mass/fine_tune/
user_dir=mass
model=checkpoint/mass/pre-training/checkpoint_last.pt # The path of pre-trained model

mkdir -p $save_dir

fairseq-train $data_dir \
    --user-dir $user_dir \
    --task xmasked_seq2seq \
    --source-langs zh --target-langs en \
    --langs en,zh \
    --arch xtransformer \
    --mt_steps zh-en \
    --save-dir $save_dir \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --lr-shrink 0.5 --lr 0.00005 --min-lr 1e-09 \
    --criterion label_smoothed_cross_entropy \
    --max-tokens 4096 \
    --max-update 100000 --max-epoch 50 \
    --dropout 0.1 --relu-dropout 0.1 --attention-dropout 0.1 \
    --share-decoder-input-output-embed \
    --valid-lang-pairs zh-en \
    --reload_checkpoint $model
```
We also provide a fine-tuning [script](ft_mass_enzh.sh) which is used for our pre-trained model.

### Inference
After the fine-tuning stage, you can generate translation results by using the below [script](translate.sh):
```
model=checkpoints/mass/fine_tune/checkpoint_best.pt
data_dir=data/processed
user_dir=mass

fairseq-generate $data_dir \
    --user-dir $user_dir \
    -s zh -t en \
    --langs en,zh \
    --source-langs zh --target-langs en \
    --mt_steps zh-en \
    --gen-subset valid \
    --task xmasked_seq2seq \
    --path $model \
    --beam 5 --remove-bpe 
```

## Text Summarization

### Data Ready
Download [CNN/Daily dataset](https://github.com/abisee/cnn-dailymail) and use Stanford CoreNLP to tokenize the data. We truncate the length of article as 400 tokens and use BPE to process data. 
We use the similar data process pipeline as NMT to generate the binarized data.  

### Pre-training
Here is a demo code about how to run pre-training in text summarization:
```
save_dir=checkpoints/mass/pre-training/
user_dir=mass
data_dir=data/processed/

mkdir -p $save_dir

fairseq-train $data_dir \
    --user-dir $user_dir \
    --save-dir $save_dir \
    --task xmasked_seq2seq \
    --source-langs ar,ti \
    --target-langs ar,ti \
    --langs ar,ti \
    --arch xtransformer \
    --mass_steps ar-ar,ti-ti \
    --memt_steps ar-ti \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --lr 0.0001 --min-lr 1e-09 \
    --criterion label_smoothed_cross_entropy \
    --max-tokens 4096 \
    --dropout 0.1 --relu-dropout 0.1 --attention-dropout 0.1 \
    --max-update 300000 \
    --share-decoder-input-output-embed \
    --valid-lang-pairs ar-ti \
    --word_mask 0.15 
```
Our experiments are still ongoing, we will summarize a better experiment setting in the future.

## Grammatical Error Correction
To be updated soon

## Paper
Paper for supervised pre-training will be available soon
