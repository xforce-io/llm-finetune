# activate conda env
eval "$(conda shell.bash hook)"
conda activate xfc

CMD="python pretrain/lit/run_trainer.py \
        --precision bf16 \
        --devices $DEVICES \
        accumulate_grad_batches=$ACCU_GRAD_BATCH \
        train_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
        deepspeed_config=\"conf/deepspeed_config.json\" \
        train_file=\"$TRAIN_FILE\" \
        validation_file=\"/mnt/data1/dev/github/llm-finetune/data/sample_eval.txt\" \
        model_name_or_path=\"$MODEL_NAME_OR_PATH\" \
        config_name=\"$BASE_MODEL\" \
        tokenizer_name=\"$BASE_MODEL\" \
        block_size=$BLOCK_SIZE \
        preprocessing_num_workers=128 \
        warmup_min_lr=1e-6 \
        warmup_max_lr=3e-4 \
        warmup_num_steps=200 \
        num_train_epochs=1 \
        default_root_dir=\"/mnt/data1/dev/github/llm-finetune/\" \
        output_dir=\"$OUTPUT_DIR\" \
        use_auth_token=\"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\" \
        do_train=true \
        do_eval=true"
srun $CMD