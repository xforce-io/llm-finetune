import torch

from typing import Any, Iterable

from lightning.fabric.strategies import DeepSpeedStrategy
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning as L
from deepspeed.ops.adam import DeepSpeedCPUAdam

from tqdm import tqdm

import sys
sys.path.append("./")

from pretrain.data import DataModule
from pretrain.lit.args_lit import ArgsLit
from pretrain.load_pretrain import loadPretrain

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
        optimizer = DeepSpeedCPUAdam(self.model.parameters())
        self.model, self.optimizer = self.fabric.setup(self.model, optimizer)
        self._train()

    def _train(self):
        dataloader = self.fabric.setup_dataloaders(self.dataModule.train_dataloader())
        self.model.train()

        iterable = self._progBarWrapper(
            dataloader,
            total=len(dataloader),
            desc=f"Trainning epochs {self.currentEpoch}")
        for epoch in range(self.args.trainArgs.num_train_epochs):
            for batchIdx, batch in enumerate(iterable):
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs[0]
                if batchIdx % 1000 == 0 and batchIdx != 0:
                    print("train loss[%f]" % loss)
                    self.eval()

                self.fabric.backward(loss)
                self.optimizer.step()

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