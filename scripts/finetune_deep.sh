deepspeed --num_gpus=8 \
    finetune/finetune.py \
    --deepspeed_config conf/deepspeed_config.json \
    --model_name_or_path bigscience/bloomz-560m \
    --train_file data/sample_train.txt \
    --validation_file data/sample_eval.txt \
    --block_size 2048 \
    --do_train \
    --do_eval \
    --output_dir output/checkpoint_emb
