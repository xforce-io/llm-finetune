export MASTER_ADDR=10.4.119.108
export MASTER_PORT=12314
export TRANSFORMERS_CACHE=/mnt/data1/.cache/huggingface/hub
export HF_HOME=/mnt/data1/.cache/huggingface
export HF_DATASETS_CACHE=/mnt/data1/.cache/huggingface/hub/dataset

export DEVICES=8
export ACCU_GRAD_BATCH=256
export MICRO_BATCH_SIZE=1
export BLOCK_SIZE=4096
export TRAIN_FILE="/mnt/data1/dev/data/book/book_05.txt"
export BASE_MODEL="meta-llama/Llama-2-13b-hf"
export MODEL_NAME_OR_PATH="/mnt/data1/dev/github/llm-finetune/output/lm13-230819/"
export OUTPUT_DIR="/mnt/data1/dev/github/llm-finetune/output/lm13-230901/"

bash scripts/base_pretrain_lit.sh