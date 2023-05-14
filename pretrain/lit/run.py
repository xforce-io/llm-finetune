import torch

from torch.utils.data import DataLoader
from torch import optim
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam

import sys
sys.path.append("./")

from pretrain.data import loadDataset, tokenizeDataset, makeDataset
from pretrain.logger import logger
from pretrain.lit.args_lit import ArgsLit
from pretrain.load_pretrain import loadTokenizer, loadPretrain

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

        logger.info("start_load_dataset")
        raw_datasets = loadDataset(args.dataArgs, args.modelArgs)

        logger.info("start_load_tokenizer")
        tokenizer = loadTokenizer(args.modelArgs)

        logger.info("start_tokenizer_dataset")
        tokenized_datasets = tokenizeDataset(args.trainArgs, args.dataArgs, tokenizer, raw_datasets)

        logger.info("start_make_dataset")
        self.dataset = makeDataset(tokenizer, args, tokenized_datasets)
        self.trainDataloader = DataLoader(
            self.dataset.train_dataset, 
            collate_fn=customCollate,
            batch_size=args.trainArgs.train_micro_batch_size_per_gpu,
            shuffle=True)
        self.evalDataloader = DataLoader(
            self.dataset.eval_dataset, 
            collate_fn=customCollate,
            batch_size=args.trainArgs.train_micro_batch_size_per_gpu,
            shuffle=True)

    def train_dataloader(self):
        return self.trainDataloader

    def val_dataloader(self):
        return self.evalDataloader

class LlmModule(LightningModule):
    def __init__(self, model, modelArgs):
        super().__init__()
        self.model = model
        self.modelArgs = modelArgs
        
    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def configure_optimizers(self):
        return DeepSpeedCPUAdam(self.parameters())

    def save_pretrained(self, outputDir):
        self.model.save_pretrained(outputDir)

def cli_main():
    args = ArgsLit()

    deepspeedStrategy = DeepSpeedStrategy(config=args.dataArgs.deepspeed_config)
    args.trainArgs.train_micro_batch_size_per_gpu = deepspeedStrategy.config["train_micro_batch_size_per_gpu"]
    
    dataModule = DataModule(args)
    llmModule = LlmModule(loadPretrain(args.modelArgs), args.modelArgs)
    compiledModule = torch.compile(llmModule)
    
    trainer = Trainer(
        max_epochs=args.trainArgs.num_train_epochs,
        accelerator="auto",
        accumulate_grad_batches=args.trainArgs.accumulate_grad_batches,
        strategy=deepspeedStrategy)
    trainer.fit(compiledModule, datamodule=dataModule)
    trainer.test(
        model=compiledModule,
        dataLoaders=dataModule.val_dataloader(),
        datamodule=dataModule)
    llmModule.save_pretrained(args.trainArgs.output_dir)

if __name__ == "__main__":
    cli_main()
