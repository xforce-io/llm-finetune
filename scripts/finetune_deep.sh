deepspeed --num_gpus=8 \
    finetune/finetune.py \
    --deepspeed_config conf/deepspeed_config.json \
    --model_name_or_path bigscience/bloom-560m \
    --train_file data/sample_train.txt \
    --validation_file data/sample_eval.txt \
    --per_device_train_batch_size 16 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --logging_steps 50 \
    --output_dir output/checkpoint
