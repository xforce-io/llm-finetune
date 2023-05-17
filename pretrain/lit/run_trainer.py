import torch

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam

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

if __name__ == "__main__":
    trainerMain()