deepspeed --num_gpus=8 \
    finetune/finetune.py \
    --deepspeed_config conf/deepspeed_config.json \
    --model_name_or_path bigscience/bloom-560m \
    --train_file data/sample_train.txt \
    --validation_file data/sample_eval.txt \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --block_size 1024 \
    --weight_decay 1e-4 \
    --do_train \
    --do_eval \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --output_dir output/checkpoint \
    --save_steps 2000 \
    --save_total_limit 3 \
    --resume_from_checkpoint false
