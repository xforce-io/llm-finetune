#!/usr/bin/env python
# coding=utf-8
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.
https://huggingface.co/models?filter=text-generation
"""

import math
import os
import sys
from dataclasses import dataclass, field

import evaluate
import torch
import deepspeed

from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from data import loadDataset, tokenizeDataset, makeDataset
from args import Args, ModelArguments, DataTrainingArguments
from logger import logger, initLogging

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
#check_min_version("4.29.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

def getLastCheckpoint(training_args) -> str:
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint
 
def loadTokenizer(model_args):
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    return tokenizer

def loadPretrain(model_args :ModelArguments) :
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )

        MAX_MEM_PER_GPU = "78GB"
        max_mem = {}
        for i in range(8):
            max_mem[i] = MAX_MEM_PER_GPU
        
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            max_memory= max_mem
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    return model

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def training(
        last_checkpoint, 
        trainer, 
        args,
        dataset):
    if args.training_args.do_train:
        checkpoint = None
        if args.training_args.resume_from_checkpoint is not None:
            checkpoint = args.training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        trainer.model, _, _, _ = deepspeed.initialize(model=trainer.model, config=args.data_args.deepspeed_config)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            args.data_args.max_train_samples \
                if args.data_args.max_train_samples is not None else \
                len(dataset.train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(dataset.train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

def evaluate(trainer, args, dataset):
    if args.training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        max_eval_samples = args.data_args.max_eval_samples \
            if args.data_args.max_eval_samples is not None \
            else len(dataset.eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(dataset.eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": args.model_args.model_name_or_path, "tasks": "text-generation"}
    if args.data_args.dataset_name is not None:
        kwargs["dataset_tags"] = args.data_args.dataset_name
        if args.data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = args.data_args.dataset_config_name
            kwargs["dataset"] = f"{args.data_args.dataset_name} {args.data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = args.data_args.dataset_name

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)

    metric = evaluate.load("accuracy")
    return metric.compute(predictions=preds, references=labels)

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    args = Args(model_args, data_args, training_args)

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    initLogging(args.training_args)

    logger.info("get_last_checkpoint")
    last_checkpoint = getLastCheckpoint(args.training_args)
    set_seed(training_args.seed)

    logger.info("start_load_dataset")
    raw_datasets = loadDataset(args.data_args, args.model_args)

    logger.info("start_load_tokenizer")
    tokenizer = loadTokenizer(args.model_args)

    logger.info("start_load_pretrain")
    model = loadPretrain(args.model_args)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    logger.info("start_tokenizer_dataset")
    tokenized_datasets = tokenizeDataset(training_args, data_args, tokenizer, raw_datasets)

    logger.info("start_make_dataset")
    dataset = makeDataset(tokenizer, args, tokenized_datasets)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=\
            preprocess_logits_for_metrics \
                if training_args.do_eval and not is_torch_tpu_available() \
                else None 
    )

    if torch.__version__ >= "2":
        print("torch compile")
        model = torch.compile(model)

    logger.info("start_training")
    training(last_checkpoint, trainer, args, dataset)

    logger.info("start_saving_pretrained")
    model.save_pretrained(training_args.output_dir)
    
    logger.info("start_evaluating")
    evaluate(trainer, args, dataset)

if __name__ == "__main__":
    main()