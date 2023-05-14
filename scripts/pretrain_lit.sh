export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=12314

lightning run model pretrain/lit/run.py \
        --precision bf16-mixed \
        --devices 8 \
        accumulate_grad_batches=4 \
        deepspeed_config="conf/deepspeed_config.json" \
        train_file="data/sample_train.txt" \
        validation_file="data/sample_eval.txt" \
        model_name_or_path="bigscience/bloom-560m" \
        block_size=1024 \
        num_train_epochs=2 \
        do_train=true \
        do_eval=true