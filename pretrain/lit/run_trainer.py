import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy, DDPStrategy
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.profiler import SimpleProfiler
from torch.optim import Adam
from deepspeed.ops.adam import DeepSpeedCPUAdam
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

torch.cuda.device_count()

import sys
sys.path.append("./")

from pretrain.data import DataModule
from pretrain.lit.args_lit import ArgsLit
from pretrain.load_pretrain import loadPretrain

class LlmModule(LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        
    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log("train_loss", loss, sync_dict=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log("validation_loss", loss, sync_dict=True)

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(), 
            lr=self.args.trainArgs.warmup_max_lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer = optimizer,
            eta_min = self.args.trainArgs.warmup_min_lr,
            warmup_epochs = self.args.trainArgs.warmup_num_steps)
        return optimizer, scheduler

    def save_pretrained(self, outputDir):
        self.model.save_pretrained(outputDir)

def makeDDPStrategy(*args):
    strategy = DDPStrategy()
    return strategy

def makeDeepSpeedStrategy(args) :
    strategy = DeepSpeedStrategy(config=args.dataArgs.deepspeed_config)
    args.trainArgs.train_micro_batch_size_per_gpu = strategy.config["train_micro_batch_size_per_gpu"]
    return strategy

def trainerMain(strategy):
    args = ArgsLit()

    dataModule = DataModule(args)
    model = loadPretrain(args.modelArgs)
    llmModule = LlmModule(model, args.args)
    
    lrLogger = LearningRateLogger(logging_interval="step")
    profiler = SimpleProfiler()
    trainer = Trainer(
        max_epochs=args.trainArgs.num_train_epochs,
        accelerator="auto",
        accumulate_grad_batches=args.trainArgs.accumulate_grad_batches,
        strategy=strategy,
        val_check_interval=0.5,
        gradient_clip_val=0.5,
        callbacks=[lrLogger],
        profiler=profiler)
    trainer.fit(llmModule, datamodule=dataModule)
    trainer.validate(model=llmModule, datamodule=dataModule)
    llmModule.save_pretrained(args.trainArgs.output_dir)

if __name__ == "__main__":
    strategy = makeDDPStrategy()
    trainerMain(strategy)