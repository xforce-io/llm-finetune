MASTER_ADDR=10.4.119.108
MASTER_PORT=12314
TRANSFORMERS_CACHE=/mnt/data1/.cache/huggingface/hub
HF_HOME=/mnt/data1/.cache/huggingface
HF_DATASETS_CACHE=/mnt/data1/.cache/huggingface/hub/dataset
TORCH_EXTENSIONS_DIR=/mnt/data1/.cache/torch_extensions
TRITON_CACHE_DIR=/mnt/data1/.cache/triton_$user

DEVICES=8
ACCU_GRAD_BATCH=256
MICRO_BATCH_SIZE=1
BLOCK_SIZE=4096
TRAIN_FILE="/mnt/data1/dev/data/nb_sample.txt"
BASE_MODEL="meta-llama/Llama-2-13b-hf"
MODEL_NAME_OR_PATH=$BASE_MODEL
OUTPUT_DIR="/mnt/data1/dev/github/llm-finetune/output/tmp/"

bash scripts/base_pretrain_lit_dist.sh