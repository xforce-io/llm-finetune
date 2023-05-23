import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy, DDPStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam
from pytorch_lightning.callbacks import LearningRateMonitor
from lightning.pytorch.profilers.simple import SimpleProfiler
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers.trainer_utils import SchedulerType

torch.cuda.device_count()

import sys
sys.path.append("./")

from pretrain.data import DataModule
from pretrain.lit.args_lit import ArgsLit
from pretrain.load_pretrain import loadPretrain

class LlmModule(LightningModule):
    def __init__(self, model, framework, args):
        super().__init__()
        self.model = model
        self.framework = framework
        self.args = args
        
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
        return self.framework.makeOptimizer(self.parameters())

    def save_pretrained(self, outputDir):
        self.model.save_pretrained(outputDir)

class Framework(object):
    def __init__(self, args):
        self.args = args

    def makeStrategy(self, args):
        pass

    def makeOptimizer(self, parameters):
        pass

class FrameworkDeepSpeed(Framework):
    def makeStrategy(self, args):
        strategy = DeepSpeedStrategy(config=args.dataArgs.deepspeed_config)
        args.trainArgs.train_micro_batch_size_per_gpu = strategy.config["train_micro_batch_size_per_gpu"]
        return strategy

    def makeOptimizer(self, parameters):
        optimizer = DeepSpeedCPUAdam(
            parameters,
            lr=self.args.trainArgs.warmup_max_lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer, 
            num_warmup_steps=self.args.trainArgs.warmup_num_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return ([optimizer], [scheduler])

class FrameworkDDP(Framework):
    def makeStrategy(self, args):
        strategy = DDPStrategy()
        return strategy

    def makeOptimizer(self, parameters):
        optimizer = Adam(
            parameters, 
            lr=self.args.trainArgs.warmup_max_lr)

        def lr_foo(epoch):
            if epoch < self.args.trainArgs.warmup_num_steps:
                lr_scale = 0.1 ** (self.args.trainArgs.warmup_num_steps - epoch)
            else:
                lr_scale = 0.95 ** epoch
            return lr_scale

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lr_foo
        )
        return ([optimizer], [scheduler])

def trainerMain(framework, args):
    dataModule = DataModule(args)
    model = loadPretrain(args.modelArgs)
    llmModule = LlmModule(model, framework, args)
    
    lrLogger = LearningRateMonitor(logging_interval="step")
    profiler = SimpleProfiler()
    trainer = Trainer(
        max_epochs=args.trainArgs.num_train_epochs,
        accelerator="auto",
        accumulate_grad_batches=args.trainArgs.accumulate_grad_batches,
        strategy=framework.makeStrategy(args),
        val_check_interval=0.5,
        gradient_clip_val=0.5,
        callbacks=[lrLogger],
        profiler=profiler)
    trainer.fit(llmModule, datamodule=dataModule)
    trainer.validate(model=llmModule, datamodule=dataModule)
    llmModule.save_pretrained(args.trainArgs.output_dir)

if __name__ == "__main__":
    args = ArgsLit()
    framework = FrameworkDDP(args)
    trainerMain(framework, args)