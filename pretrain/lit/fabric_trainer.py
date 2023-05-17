import torch

from typing import Any, Iterable

from torch.utils.data import DataLoader
from torch import optim
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from lightning.fabric.strategies import DeepSpeedStrategy
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from deepspeed.ops.adam import DeepSpeedCPUAdam
from lightning import Fabric
import lightning as L

from tqdm import tqdm

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
            num_workers=args.dataArgs.preprocessing_num_workers,
            shuffle=True)
        self.evalDataloader = DataLoader(
            self.dataset.eval_dataset, 
            collate_fn=customCollate,
            batch_size=args.trainArgs.train_micro_batch_size_per_gpu,
            num_workers=args.dataArgs.preprocessing_num_workers)

    def train_dataloader(self):
        return self.trainDataloader

    def val_dataloader(self):
        return self.evalDataloader

class FabricTrainer:
    def __init__(self) -> None:
        self.args = ArgsLit()

        deepspeedStrategy = DeepSpeedStrategy(
            config=self.args.dataArgs.deepspeed_config)
        self.args.trainArgs.train_micro_batch_size_per_gpu = deepspeedStrategy.config["train_micro_batch_size_per_gpu"]

        earlyStopCallback = EarlyStopping(
            monitor="val_accuracy",
            min_delta=0.1,
            patience=3,
            verbose=False,
            mode="max")

        self.fabric = L.Fabric(
            accelerator="cuda", 
            strategy=deepspeedStrategy,
            callbacks=[earlyStopCallback])
 
        self.dataModule = DataModule(self.args)
        self.currentEpoch = 0
    
    def train(self):
        dataloader = self.fabric.setup_dataloaders(self.dataModule.train_dataloader())
        self.model.train()

        iterable = self._progBarWrapper(
            dataloader,
            total=len(dataloader),
            desc=f"Trainning epochs {self.currentEpoch}")
        for epoch in range(self.args.trainArgs.num_train_epochs):
            i = 0
            for batchIdx, batch in enumerate(iterable):
                if i % len(dataloader) == 0:
                    self.eval(self.dataModule, self.model, fabric)
                    
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs[0]

                print("train loss[%f]" % loss)

                self.fabric.backward(loss)
                self.optimizer.step()

    def eval(self):
        dataloader = self.fabric.setup_dataloaders(self.dataModule.val_dataloader())
        self.model.eval()

        iterable = self._progBarWrapper(
            dataloader,
            total=len(dataloader),
            desc=f"evaluation")
        with torch.no_grad():
            for batchIdx, batch in enumerate(iterable):
                outputs = self.model(**batch)
                loss = outputs[0]
            print("eval loss[%f]" % loss)

    def train(self):
        self.model = loadPretrain(self.args.modelArgs)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5)
        self.model, self.optimizer = self.fabric.setup(self.model, optimizer)
        self.train()

    def _progBarWrapper(
            self, 
            iterable :Iterable, 
            total :int, 
            **kwargs :Any):
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)        
        return iterable

if __name__ == "__main__":
    pass