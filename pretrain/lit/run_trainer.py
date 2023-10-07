import os
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy, DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.profilers.simple import SimpleProfiler
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers import TensorBoardLogger
import deepspeed.comm as dist
from datetime import timedelta

torch.cuda.device_count()

import sys
sys.path.append("./")

from pretrain.data import DataModule
from pretrain.lit.args_lit import ArgsLit
from pretrain.load_pretrain import loadPretrain
from pretrain.logger import log, initLogging

from pretrain.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

replace_llama_attn_with_flash_attn()

class LlmModule(LightningModule):
    def __init__(self, model, framework, args):
        super().__init__()
        self.model = model
        self.framework = framework
        self.args = args
        
    def forward(self, **inputs):
        return self.model(**inputs, use_cache=False)

    def training_step(self, batch, batch_idx):
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", lr, prog_bar=True)

        outputs = self(**batch)
        loss = outputs[0]
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return self.framework.makeOptimizer(self.parameters())

    def save_pretrained(self, outputDir):
        self.model.save_pretrained(outputDir)

class Framework(object):
    def __init__(self, args):
        self.args = args

    def makeStrategy(self, args):
        return None

    def makeOptimizer(self, parameters):
        return None

class FrameworkDeepSpeed(Framework):
    def makeStrategy(self, args):
        strategy = DeepSpeedStrategy(config=args.dataArgs.deepspeed_config)
        strategy.config["train_micro_batch_size_per_gpu"] = args.trainArgs.train_micro_batch_size_per_gpu
        return strategy

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
    model.resize_token_embeddings(len(dataModule.tokenizer))
    
    llmModule = LlmModule(model, framework, args)
    
    ckptCallback = ModelCheckpoint(
        dirpath=args.trainArgs.output_dir, 
        save_last=True,
        every_n_train_steps=args.trainArgs.every_n_train_steps)
    
    profiler = SimpleProfiler()
    trainer = Trainer(
        max_epochs=args.trainArgs.num_train_epochs,
        accelerator="auto",
        accumulate_grad_batches=args.trainArgs.accumulate_grad_batches,
        strategy=framework.makeStrategy(args),
        val_check_interval=0.2,
        gradient_clip_val=0.5,
        callbacks=[ckptCallback],
        profiler=profiler,
        enable_checkpointing=True,
        default_root_dir=args.trainArgs.default_root_dir,
        inference_mode=False,
        logger=TensorBoardLogger(
            save_dir=args.trainArgs.default_root_dir, 
            version=1, 
            name="lightning_logs") 
                if args.trainArgs.logger_tensorboard 
                else CSVLogger(f"{args.trainArgs.logger_tensorboard}/lightning_logs/", name="current"))

    trainer.fit(llmModule, datamodule=dataModule)
    trainer.validate(model=llmModule, datamodule=dataModule)
    llmModule.save_pretrained(args.trainArgs.output_dir)

    dist.log_summary()

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    if "SLURM_NTASKS" in os.environ:
        print(f"procid[{os.environ['SLURM_PROCID']}] ntasks[{os.environ['SLURM_NTASKS']}]")
        torch.distributed.init_process_group(
            rank=int(os.environ["SLURM_PROCID"]),
            world_size=int(os.environ["SLURM_NTASKS"]),
            backend="nccl")
    else:
        torch.distributed.init_process_group(backend="nccl")

    args = ArgsLit()
    initLogging(args)

    framework = FrameworkDeepSpeed(args)
    trainerMain(framework, args)