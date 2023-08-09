import torch

from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from pretrain.args import ModelArguments

from pretrain.logger import log

DEFAULT_EOS_TOKEN = "</s>"

def loadTokenizer(modelArgs):
    tokenizer_kwargs = {
        "cache_dir": modelArgs.cache_dir,
        "use_fast": modelArgs.use_fast_tokenizer,
        "revision": modelArgs.model_revision,
        "token": True if modelArgs.use_auth_token else None,
        "trust_remote_code": True
    }
    if modelArgs.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            modelArgs.tokenizer_name, 
            **tokenizer_kwargs)
    elif modelArgs.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            modelArgs.model_name_or_path, 
            **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    special_tokens_dict = dict()
    special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    tokenizer.add_special_tokens(special_tokens_dict)
    log.info(f"model_max_length for tokenization is {tokenizer.model_max_length}")
    return tokenizer

def loadPretrain(modelArgs :ModelArguments) :
    config_kwargs = {
        "cache_dir": modelArgs.cache_dir,
        "revision": modelArgs.model_revision,
        "token": True if modelArgs.use_auth_token else None,
        "trust_remote_code": True
    }
    if modelArgs.config_name:
        config = AutoConfig.from_pretrained(modelArgs.config_name, **config_kwargs)
    elif modelArgs.model_name_or_path:
        config = AutoConfig.from_pretrained(modelArgs.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[modelArgs.model_type]()
        log.warning("You are instantiating a new config instance from scratch.")
        if modelArgs.config_overrides is not None:
            log.info(f"Overriding config: {modelArgs.config_overrides}")
            config.update_from_string(modelArgs.config_overrides)
            log.info(f"New config: {config}")

    if modelArgs.model_name_or_path:
        torch_dtype = (
            modelArgs.torch_dtype
            if modelArgs.torch_dtype in ["auto", None]
            else getattr(torch, modelArgs.torch_dtype)
        )

        if modelArgs.load_in_8bit:
            model = AutoModelForCausalLM.from_pretrained(
                modelArgs.model_name_or_path,
                from_tf=bool(".ckpt" in modelArgs.model_name_or_path),
                config=config,
                cache_dir=modelArgs.cache_dir,
                revision=modelArgs.model_revision,
                token=True if modelArgs.use_auth_token else None,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True)
        else:
             model = AutoModelForCausalLM.from_pretrained(
                modelArgs.model_name_or_path,
                from_tf=bool(".ckpt" in modelArgs.model_name_or_path),
                config=config,
                cache_dir=modelArgs.cache_dir,
                revision=modelArgs.model_revision,
                token=True if modelArgs.use_auth_token else None,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=modelArgs.low_cpu_mem_usage,
                trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        log.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    return model
