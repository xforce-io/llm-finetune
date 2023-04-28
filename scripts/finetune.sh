python finetune/finetune.py \
    --model_name_or_path bigscience/bloomz-560m \
    --train_file data/sample_train.txt \
    --validation_file data/sample_eval.txt \
    --block_size 2048 \
    --do_train \
    --do_eval \
    --deepspeed_config conf/deepspeed_config.json \
    --output_dir output/checkpoint_emb