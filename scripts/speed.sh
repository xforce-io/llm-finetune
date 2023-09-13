export MASTER_ADDR=10.4.119.109
export MASTER_PORT=12314

lightning run model  pretrain/lit/run_trainer.py \
        --precision bf16 \
        --devices 8 \
        accumulate_grad_batches=256 \
        train_micro_batch_size_per_gpu=1 \
        deepspeed_config="conf/deepspeed_config.json" \
        train_file="/mnt/data1/dev/github/llm-finetune/data/sample_train.txt" \
        validation_file="/mnt/data1/dev/github/llm-finetune/data/sample_eval.txt" \
        model_name_or_path="/mnt/data1/dev/github/llm-finetune/output/lm13-230819/" \
        config_name="meta-llama/Llama-2-13b-hf" \
        tokenizer_name="meta-llama/Llama-2-13b-hf" \
        block_size=4096 \
        preprocessing_num_workers=128 \
        warmup_min_lr=1e-6 \
        warmup_max_lr=3e-4 \
        warmup_num_steps=200 \
        num_train_epochs=1 \
        default_root_dir="/mnt/data1/dev/github/llm-finetune/" \
        output_dir="/mnt/data1/dev/github/llm-finetune/output/lm13-230905/" \
        use_auth_token="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" \
        do_train=true \
        do_eval=true