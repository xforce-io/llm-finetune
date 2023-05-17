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
        self.trainDataloader, self.evalDataloader = self.fabric.setup_dataloaders(
            self.dataModule.train_dataloader(),
            self.dataModule.evalDataloader())

    def fit(self):
        self.model = loadPretrain(self.args.modelArgs)
        optimizer = DeepSpeedCPUAdam(self.model.parameters())
        self.model, self.optimizer = self.fabric.setup(self.model, optimizer)
        for epoch in range(self.args.trainArgs.num_train_epochs):
            self._trainStep()
            self._evalStep()

    def _trainStep(self):
        self.model.train()
        curLoss = None
        iterable = self._progBarWrapper(
            self.trainDataloader,
            total=len(self.trainDataloader),
            desc=f"Trainning epochs {self.currentEpoch}/{curLoss}")
        for batchIdx, batch in enumerate(iterable):
            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            curLoss = outputs[0]
            self.fabric.backward(curLoss)
            self.optimizer.step()

    def _evalStep(self):
        self.model.eval()
        iterable = self._progBarWrapper(
            self.evalDataloader,
            total=len(self.evalDataloader),
            desc=f"evaluation")
        with torch.no_grad():
            for batchIdx, batch in enumerate(iterable):
                outputs = self.model(**batch)
                loss = outputs[0]
            print("eval loss[%f]" % loss)

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