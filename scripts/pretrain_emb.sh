python pretrain/pretrain_emb.py \
    --config_name decapoda-research/llama-13b-hf \
    --tokenizer_name output/merged_tokenizer_hf \
    --train_file data/sample_train.txt \
    --validation_file data/sample_eval.txt \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir output/checkpoint_emb
