import torch

from torch.utils.data import DataLoader
from torch import optim
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from lightning.fabric.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam
from lightning import Fabric
import lightning as L

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
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log("validation_loss", loss)

    def configure_optimizers(self):
        return DeepSpeedCPUAdam(self.parameters())

    def save_pretrained(self, outputDir):
        self.model.save_pretrained(outputDir)

def trainerMain():
    args = ArgsLit()

    deepspeedStrategy = DeepSpeedStrategy(config=args.dataArgs.deepspeed_config)
    args.trainArgs.train_micro_batch_size_per_gpu = deepspeedStrategy.config["train_micro_batch_size_per_gpu"]
    
    dataModule = DataModule(args)
    model = loadPretrain(args.modelArgs)
    llmModule = LlmModule(model, args.modelArgs)
    
    trainer = Trainer(
        max_epochs=args.trainArgs.num_train_epochs,
        accelerator="auto",
        accumulate_grad_batches=args.trainArgs.accumulate_grad_batches,
        strategy=deepspeedStrategy,
        val_check_interval=0.05)
    trainer.fit(llmModule, datamodule=dataModule)
    trainer.validate(model=llmModule, datamodule=dataModule)
    llmModule.save_pretrained(args.trainArgs.output_dir)

def fabricTrain(
        args,
        dataModule, 
        model, 
        optimizer,
        fabric):
    dataloader = fabric.setup_dataloaders(dataModule.train_dataloader)
    model.train()
    for epoch in range(args.trainArgs.num_train_epochs):
        i = 0
        for batch in dataloader:
            if i % len(dataloader) == 0:
                fabricEval(dataModule, model, fabric)
                
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs[0]

            print("train loss[%f]" % loss)

            fabric.backward(loss)
            optimizer.step()

def fabricEval(
        dataModule,
        model,
        fabric):
    dataloader = fabric.setup_dataloaders(dataModule.val_dataloader)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            loss = outputs[0]
        print("eval loss[%f]" % loss)

def fabricMain():
    args = ArgsLit()

    deepspeedStrategy = DeepSpeedStrategy(
        config=args.dataArgs.deepspeed_config)
    args.trainArgs.train_micro_batch_size_per_gpu = deepspeedStrategy.config["train_micro_batch_size_per_gpu"]

    fabric = L.Fabric(
        accelerator="cuda", 
        strategy=deepspeedStrategy)
    fabric.launch()
    
    dataModule = DataModule(args)
    model = loadPretrain(args.modelArgs)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    model, optimizer = fabric.setup(model, optimizer)
    
    fabricTrain(args, dataModule, model, optimizer, fabric)

if __name__ == "__main__":
    fabricMain()