import torch
import transformers
from transformers.testing_utils import CaptureLogger
from datasets import load_dataset
from itertools import chain
from pretrain.logger import log

ID_IGNORED = -100

class Dataset:
    def __init__(self) -> None:
        pass

    def set_train_dataset(self, train_dataset, dataArgs) -> None:
        self.train_dataset = train_dataset
        if dataArgs.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), dataArgs.max_train_samples)
            self.train_dataset = train_dataset.select(range(max_train_samples))

    def set_eval_dataset(self, eval_dataset, dataArgs) -> None:
        self.eval_dataset = eval_dataset
        if dataArgs.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), dataArgs.max_eval_samples)
            self.eval_dataset = eval_dataset.select(range(max_eval_samples))

def loadDataset(dataArgs, modelArgs) :
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    dataset_args = {}
    if dataArgs.train_file is not None:
        data_files["train"] = dataArgs.train_file
    if dataArgs.validation_file is not None:
        data_files["validation"] = dataArgs.validation_file
    extension = (
        dataArgs.train_file.split(".")[-1]
        if dataArgs.train_file is not None
        else dataArgs.validation_file.split(".")[-1]
    )

    if extension == "txt":
        extension = "text"
        dataset_args["keep_linebreaks"] = dataArgs.keep_linebreaks
    elif extension == "jsonl":
        extension = "json"
        
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=modelArgs.cache_dir,
        use_auth_token=True if modelArgs.use_auth_token else None,
        **dataset_args,
    )
    # If no validation data is there, validation_split_percentage will be used to divide the dataset.
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{dataArgs.validation_split_percentage}%]",
            cache_dir=modelArgs.cache_dir,
            use_auth_token=True if modelArgs.use_auth_token else None,
            **dataset_args,
        )
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{dataArgs.validation_split_percentage}%:]",
            cache_dir=modelArgs.cache_dir,
            use_auth_token=True if modelArgs.use_auth_token else None,
            **dataset_args,
        )
    return raw_datasets

def tokenizeDataset(trainArgs, dataArgs, tokenizer, raw_datasets):
    if trainArgs.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]
    label_column_name = "labels"

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            text = examples[text_column_name]
            size = len(text)
            
            tok_text = tokenizer(text)
            tok_label = None
            if label_column_name in examples:
                tok_label = tokenizer(examples[label_column_name])

            if label_column_name in examples:
                tok_text["labels"] = []

            for i in range(size):
                if label_column_name in examples:
                    tok_text["labels"].append(len(tok_text["input_ids"][i]) * [ID_IGNORED])
                    #tok_text["labels"].append(tok_text["input_ids"][i].copy())
                    tok_text["labels"][i] += tok_label["input_ids"][i]
                    tok_text["labels"][i].append(tokenizer.eos_token_id)

                    tok_text["input_ids"][i] += tok_label["input_ids"][i]
                    tok_text["attention_mask"][i] += tok_label["attention_mask"][i]
                
                tok_text["input_ids"][i].append(tokenizer.eos_token_id)
                tok_text["attention_mask"][i].append(1)

        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return tok_text

    with trainArgs.main_process_first(desc="dataset map tokenization"):
        if not dataArgs.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=dataArgs.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not dataArgs.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )
    return tokenized_datasets

def makeDataset(tokenizer, args, tokenized_datasets) :
    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length-block_size, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    if args.dataArgs.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            log.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if args.dataArgs.block_size > tokenizer.model_max_length:
            log.warning(
                f"The block_size passed ({args.dataArgs.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.dataArgs.block_size, tokenizer.model_max_length)

    with args.trainArgs.main_process_first(desc="grouping texts together"):
        if not args.dataArgs.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=args.dataArgs.preprocessing_num_workers,
                load_from_cache_file=not args.dataArgs.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )
    
    dataset = Dataset()
    if args.trainArgs.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        dataset.set_train_dataset(lm_datasets["train"], args.dataArgs)
        log.info("token_size_to_be_trained[%d]" % (len(lm_datasets["train"]) * block_size))

    if args.trainArgs.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        dataset.set_eval_dataset(lm_datasets["validation"], args.dataArgs)
    return dataset

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from pretrain.load_pretrain import loadTokenizer

def customCollate(data) :
    inputIds = []
    attentionMask = []
    labels = []
    for item in data:
        inputIds += [item["input_ids"]]
        attentionMask += [item["attention_mask"]]
        labels += [item["labels"]]
    return {
        "input_ids" : torch.LongTensor(inputIds),
        "attention_mask" : torch.LongTensor(attentionMask),
        "labels" : torch.LongTensor(labels)
}

class DataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()

        log.info("start_load_dataset")
        raw_datasets = loadDataset(args.dataArgs, args.modelArgs)

        log.info("start_load_tokenizer")
        tokenizer = loadTokenizer(args.modelArgs)

        log.info("start_tokenizer_dataset")
        tokenized_datasets = tokenizeDataset(args.trainArgs, args.dataArgs, tokenizer, raw_datasets)

        log.info("start_make_dataset")
        self.dataset = makeDataset(tokenizer, args, tokenized_datasets)
        self.trainDataloader = DataLoader(
            self.dataset.train_dataset, 
            collate_fn=customCollate,
            batch_size=args.trainArgs.train_micro_batch_size_per_gpu,
            num_workers=args.dataArgs.preprocessing_num_workers,
            shuffle=True)
        self.evalDataloader = DataLoader(
            self.dataset.eval_dataset, 
            collate_fn=customCollate,
            batch_size=args.trainArgs.train_micro_batch_size_per_gpu,
            num_workers=args.dataArgs.preprocessing_num_workers)

        self.tokenizer = tokenizer

    def train_dataloader(self):
        return self.trainDataloader

    def val_dataloader(self):
        return self.evalDataloader