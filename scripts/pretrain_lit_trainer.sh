export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=12314

CUDA_LAUNCH_BLOCKING=1 lightning run model pretrain/lit/run_trainer.py \
        --precision bf16-mixed \
        --devices 8 \
        accumulate_grad_batches=2 \
        deepspeed_config="conf/deepspeed_config.json" \
        train_file="data/sample_train.txt" \
        validation_file="data/sample_eval.txt" \
        model_name_or_path="bigscience/bloom-560m" \
        block_size=1024 \
        num_train_epochs=2 \
        output_dir="output/checkpoint" \
        do_train=true \
        preprocessing_num_workers=8 \
        do_eval=true