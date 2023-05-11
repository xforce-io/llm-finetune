from omegaconf import OmegaConf
from pretrain.args import ModelArguments, DataArguments
from transformers import TrainingArguments
import dataclasses

class TrainingArgumentsLit(TrainingArguments):
    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

class ArgsLit:
    def __init__(self) -> None:
        conf = OmegaConf.from_cli()
        data = OmegaConf.structured(conf)
        self.modelArgs = ModelArguments(**data)
        self.dataArgs = DataArguments(**data)
        self.trainArgs = TrainingArgumentsLit(**data)

if __name__ == "__main__":
    argLit = ArgsLit()