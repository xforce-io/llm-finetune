#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8

# activate conda env
eval "$(conda shell.bash hook)"
conda activate xfc

export NCCL_SOCKET_IFNAME="eno1"
export NCCL_IB_PCI_RELAXED_ORDERING="1"
export NCCL_DEBUG="INFO"
export MASTER_ADDR=10.4.119.108
export MASTER_PORT=12314
export TRANSFORMERS_CACHE=/mnt/data1/.cache/huggingface/hub
export HF_HOME=/mnt/data1/.cache/huggingface
export HF_DATASETS_CACHE=/mnt/data1/.cache/huggingface/hub/dataset
export TORCH_EXTENSIONS_DIR=/mnt/data1/.cache/torch_extensions
export TORCH_DISTRIBUTED_DEBUG=INFO

CMD="python pretrain/lit/run_trainer.py --precision bf16 \
        --devices 8 \
        train_micro_batch_size_per_gpu=4 \
        accumulate_grad_batches=16 \
        deepspeed_config=\"conf/deepspeed_config.json\" \
        train_file=\"data/sample_train_10w.txt\" \
        validation_file=\"data/sample_eval.txt\" \
        model_name_or_path=\"/mnt/data1/dev/github/llm-finetune/output/lm7_ns_bk/\" \
        config_name=\"decapoda-research/llama-7b-hf\" \
        tokenizer_name=\"decapoda-research/llama-7b-hf\" \
        block_size=1024 \
        preprocessing_num_workers=64 \
        warmup_min_lr=1e-6 \
        warmup_max_lr=3e-4 \
        warmup_num_steps=200 \
        num_train_epochs=1 \
        default_root_dir=\"/mnt/data1/dev/github/llm-finetune/\" \
        output_dir=\"/mnt/data1/dev/github/llm-finetune/output/tmp/\" \
        do_train=true \
        do_eval=true"
srun $CMD
