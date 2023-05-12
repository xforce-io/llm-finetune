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
        self.trainDataloader = DataLoader(self.dataset.train_dataset, batch_size=args.trainArgs.train_batch_size)
        self.evalDataloader = DataLoader(self.dataset.eval_dataset, batch_size=args.trainArgs.train_batch_size)

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

def cli_main():
    args = ArgsLit()

    deepspeedStrategy = DeepSpeedStrategy(config=args.dataArgs.deepspeed_config)
    args.trainArgs.train_batch_size = deepspeedStrategy.config["train_batch_size"]
    
    dataModule = DataModule(args)
    llmModule = LlmModule(loadPretrain(args.modelArgs), args.modelArgs)
    trainer = Trainer(
        max_epochs=args.trainArgs.num_train_epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        strategy=deepspeedStrategy)
    trainer.fit(llmModule, datamodule=dataModule)
    trainer.test(
        model=llmModule,
        dataLoaders=dataModule.val_dataloader(),
        ckpt_path=args.trainArgs.output_dir, 
        datamodule=dataModule)

if __name__ == "__main__":
    cli_main()