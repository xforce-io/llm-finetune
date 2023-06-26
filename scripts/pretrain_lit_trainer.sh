export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=12314

CUDA_LAUNCH_BLOCKING=1 lightning run model pretrain/lit/run_trainer.py \
        --precision bf16-mixed \
        --devices 8 \
        port=5000 \
        train_micro_batch_size_per_gpu=4 \
        accumulate_grad_batches=16 \
        deepspeed_config="conf/deepspeed_config.json" \
        train_file="data/sample_train.txt" \
        validation_file="data/sample_eval.txt" \
        model_name_or_path="bigscience/bloom-560m" \
        block_size=1024 \
        preprocessing_num_workers=8 \
        warmup_min_lr=1e-6 \
        warmup_max_lr=3e-4 \
        warmup_num_steps=1000 \
        num_train_epochs=2 \
        default_root_dir="" \ 
        output_dir="output/checkpoint" \
        do_train=true \
        do_eval=true