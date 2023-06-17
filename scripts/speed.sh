CUDA_VISIBLE_DEVICES=7 lightning run model pretrain/lit/run_trainer.py \
        --precision bf16-mixed \
        --main_port 29500 \
        --devices 1 \
        port=5001 \
        train_micro_batch_size_per_gpu=4 \
        accumulate_grad_batches=16 \
        deepspeed_config="conf/deepspeed_config.json" \
        train_file="data/sample_train_10w.txt" \
        validation_file="data/sample_eval.txt" \
        model_name_or_path="bigscience/bloom-560m" \
        config_name="bigscience/bloom-560m" \
        tokenizer_name="bigscience/bloom-560m" \
        block_size=1024 \
        preprocessing_num_workers=64 \
        warmup_min_lr=1e-6 \
        warmup_max_lr=3e-4 \
        warmup_num_steps=200 \
        num_train_epochs=1 \
        default_root_dir="/data/mnt/data1/dev/github/llm-finetune/" \
        output_dir="/data/mnt/data1/dev/github/llm-finetune/output/speed/" \
        do_train=true \
        do_eval=true \
        low_cpu_mem_usage=true
