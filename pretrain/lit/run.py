import torch

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy

import sys
sys.path.append("./")

from pretrain.data import loadDataset, tokenizeDataset, makeDataset
from pretrain.logger import logger, initLogging
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

    def train_dataloader(self):
        return DataLoader(self.dataset.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset.eval_dataset, batch_size=self.batch_size)

def cli_main():
    args = ArgsLit()
    dataModule = DataModule(args)

    model = loadPretrain(args.modelArgs)
    
    trainer = Trainer(
        max_epochs=args.trainArgs.num_train_epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        strategy=DeepSpeedStrategy(config=args.dataArgs.deepspeed_config))
    trainer.fit(model, datamodule=dataModule)
    trainer.test(
        model=model,
        dataLoaders=dataModule.val_dataloader(),
        ckpt_path=args.trainArgs.output_dir, 
        datamodule=dataModule)


if __name__ == "__main__":
    cli_main()