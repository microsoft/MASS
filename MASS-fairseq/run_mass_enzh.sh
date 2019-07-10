data_dir=data/processed
save_dir=checkpoints/mass/pretraining
user_dir=mass

seed=1234
max_tokens=2048 # for 16GB GPUs 
update_freq=1
dropout=0.1
attention_heads=16
embed_dim=1024
ffn_embed_dim=4096
encoder_layers=10
decoder_layers=6
word_mask=0.3

mkdir -p $save_dir

fairseq-train $data_dir \
	--user-dir $user_dir \
    --task xmasked_seq2seq \
	--source-langs en,zh \
	--target-langs en,zh \
    --langs en,zh \
	--arch xtransformer \
    --mass_steps en-en,zh-zh \
    --memt_steps en-zh,zh-en \
    --save-dir $save_dir \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --lr 0.00005 --min-lr 1e-09 \
    --criterion label_smoothed_cross_entropy \
    --lm-bias --lazy-load --seed ${seed} \
    --log-format json \
    --max-tokens ${max_tokens} --update-freq ${update_freq} \
    --encoder-normalize-before  --decoder-normalize-before \
    --dropout ${dropout} --relu-dropout 0.1 --attention-dropout 0.1 \
    --decoder-attention-heads ${attention_heads} --encoder-attention-heads ${attention_heads} \
    --decoder-embed-dim ${embed_dim} --encoder-embed-dim ${embed_dim} \
    --decoder-ffn-embed-dim ${ffn_embed_dim} --encoder-ffn-embed-dim ${ffn_embed_dim} \
    --encoder-layers ${encoder_layers} --decoder-layers ${decoder_layers} \
    --max-update 100000000 --max-epoch 50 \
    --keep-interval-updates 100 --save-interval-updates 3000  --log-interval 50 \
    --share-decoder-input-output-embed \
    --valid-lang-pairs en-zh \
	--word_mask ${word_mask} \
	--ddp-backend=no_c10d
