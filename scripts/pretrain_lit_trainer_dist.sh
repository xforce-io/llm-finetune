#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8

# activate conda env
eval "$(conda shell.bash hook)"
conda activate xfc

export MASTER_ADDR=10.4.119.108
export MASTER_PORT=12314
export TRANSFORMERS_CACHE=/mnt/data1/.cache/huggingface/hub
export HF_HOME=/mnt/data1/.cache/huggingface
export HF_DATASETS_CACHE=/mnt/data1/.cache/huggingface/hub/dataset
export TORCH_EXTENSIONS_DIR=/mnt/data1/.cache/torch_extensions
export TRITON_CACHE_DIR=/mnt/data1/.cache/triton_$user

CMD="python pretrain/lit/run_trainer.py \
        --precision bf16 \
        --devices 8 \
        accumulate_grad_batches=256 \
        train_micro_batch_size_per_gpu=1 \
        deepspeed_config=\"conf/deepspeed_config.json\" \
        train_file=\"/mnt/data1/dev/data/nb_sample.txt\" \
        validation_file=\"/mnt/data1/dev/github/llm-finetune/data/sample_eval.txt\" \
        model_name_or_path=\"meta-llama/Llama-2-13b-hf\" \
        config_name=\"meta-llama/Llama-2-13b-hf\" \
        tokenizer_name=\"meta-llama/Llama-2-13b-hf\" \
        block_size=4096 \
        preprocessing_num_workers=128 \
        warmup_min_lr=1e-6 \
        warmup_max_lr=3e-4 \
        warmup_num_steps=200 \
        num_train_epochs=1 \
        default_root_dir=\"/mnt/data1/dev/github/llm-finetune/\" \
        output_dir=\"/mnt/data1/dev/github/llm-finetune/output/tmp\" \
        use_auth_token=\"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\" \
        do_train=true \
        do_eval=true"
srun $CMD