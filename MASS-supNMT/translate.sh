MODEL=zhen_mass_pre-training.pt

fairseq-generate ./data/processed \
	-s zh -t en \
	--user-dir mass \
	--langs en,zh \
	--source-langs zh --target-langs en \
	--mt_steps zh-en \
	--gen-subset valid \
	--task xmasked_seq2seq \
	--path $MODEL \
	--beam 5 \
	--remove-bpe
