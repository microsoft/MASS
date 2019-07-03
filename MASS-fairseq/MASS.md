# MASS on fairseq

This is an internal implementation of MASS on fairseq

## Pre-train Data Ready

For monolingual data, we assume that the directory of monolingual data is like that:

```
data/
|-- mono/
    |-- train.en
    |-- valid.en
    |-- dict.en.txt
```
while `train.en`, `valid.en`, `dict.en.txt` correspond to the training data, valid data and dictionary. So the script for data generation is as:

```
data_dir=data/mono/
save_dir=data/processed
mkdir -p $data_dir $save_dir

for lg in en
do
  python preprocess.py \
  --task cross_lingual_lm \
  --srcdict $data_dir/dict.$lg.txt \
  --only-source \
  --trainpref $data_dir/train \
  --validpref $data_dir/valid \
  --destdir $save_dir \
  --workers 20 \
  --source-lang $lg

  for stage in train valid
  do
    mv $save_dir/$stage.$lg-None.$lg.bin $save_dir/$stage.$lg.bin
    mv $save_dir/$stage.$lg-None.$lg.idx $save_dir/$stage.$lg.idx
  done
done
```
After that, the files under binary data directory will be liked that:
```
data/
|-- processed/
    |-- train.en.bin
    |-- train.en.idx
    |-- valid.en.bin
    |-- valid.en.idx
    |-- dict.en.txt
```

## Pre-training Stage
After data processing, we start to deploy MASS on monolingual data, and this is a demo code to for MASS on english monolingual data.
```
save_dir=checkpoints/mass_en

mkdir -p $save_dir

CUDA_VISIBLE_DEVICES=0 python train.py ./data/processed \
	--task xmasked_seq2seq \
	--mono_langs en \
	--langs en \
	--save-dir $save_dir \
	--max-update 30000 \
	--no-epoch-checkpoints \
	--optimizer adam --lr-scheduler reduce_lr_on_plateau \
	--lr-shrink 0.5 --lr 0.0001 --min-lr 1e-09 \
	--dropout 0.1 \
	--criterion label_smoothed_cross_entropy \
	--max-tokens 4096 \
	--ddp-backend=no_c10d \
	--lm-bias --lazy-load --seed 0 \
	--arch xtransformer \
	--log-format json \
	--log-interval 1000 \
	--mass_steps en-en \
	--valid-lang-pairs en-en \
	--save-interval-updates 1000 \
	--no-epoch-checkpoints \
```

## Fine-tune Data Ready
We take text summarization as a example. We name `train.article (valid.article)` as `train.en (valid.en)` and `train.title (valid.title)` as `train.fr (valid.fr)`. We use same dictionary in english monolingual dataset, and copy it as `dict.en.txt (dict.fr.txt)`. So the directory of dataset can be viewed as:
```
data/
|-- para/
    |-- train.en
    |-- train.fr
    |-- valid.en
    |-- valid.fr
    |-- dict.fr.txt
    |-- dict.en.txt
```
The data generation command is as:
```
data_dir=data/para/
save_dir=data/processed

python preprocess.py \
    --task xmasked_seq2seq \
    --source-lang en --target-lang fr \
    --trainpref $data_dir/train --validpref $data_dir/valid \
    --destdir $save_dir \
    --srcdict $data_dir/dict.en.txt \
    --tgtdict $data_dir/dict.fr.txt
```
After data generation, our data directory is as:
```
data/
|-- processed/
    |-- train.en-fr.en.bin
    |-- train.en-fr.en.idx
    |-- train.en-fr.fr.bin
    |-- train.en-fr.fr.idx
    |-- valid.en-fr.en.bin
    |-- valid.en-fr.en.idx
    |-- valid.en-fr.fr.bin
    |-- valid.en-fr.fr.idx
    |-- dict.en.txt
    |-- dict.fr.txt
```

## Fine-tuning 
```
save_dir=checkpoints/text_summarization

python train.py ./data/processed/ \
   	--task xmasked_seq2seq \
	--mono_langs en,fr \
	--langs en,fr \
	--para_lang_pairs en-fr \
	--save-dir $save_dir \
	--max-update 100000 \
	--no-epoch-checkpoints \
	--optimizer adam --lr-scheduler reduce_lr_on_plateau \
	--lr-shrink 0.5 --lr 0.0001 --min-lr 1e-09 \
	--dropout 0.1 \
	--criterion label_smoothed_cross_entropy \
	--max-tokens 4096 \
	--ddp-backend=no_c10d \
	--lm-bias --lazy-load --seed 0 \
	--arch xtransformer \
	--log-format json \
	--log-interval 1000 \
	--mt_steps en-fr \
	--valid-lang-pairs en-fr \
	--no-epoch-checkpoints \
```
## Inference
After pre-training, we inference our results by using the following scripts:
```
data_dir=data/processed/
MODEL=checkpoints/checkpoint_best.pt

python generate.py $data_dir \
  --langs en,fr \
  --source-lang en --target-lang fr \
  --para_lang_pairs en-fr \
  --mt_steps en-fr \
  --gen-subset test \
  --task xmasked_seq2seq \
  --path $MODEL \ 
  --batch-size 128 --beam 5 --remove-bpe \
  --min-len 4 \
```
## Tips
The data of all of our experiments are mainly in translation domain (NewsCrawl). You can also use in-domain data from downstream tasks to obtain better performance.  
