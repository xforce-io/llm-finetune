from omegaconf import OmegaConf
from pretrain.args import ModelArguments, DataArguments
from transformers import TrainingArguments
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

@dataclass(init=False)
class TrainingArgumentsLit(TrainingArguments):

    train_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "data batch size."
            )
        },
    )

    accumulate_grad_batches: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "accumulate grad batches."
            )
        },
    )

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
        self.trainArgs.deepspeed = self.dataArgs.deepspeed_config

if __name__ == "__main__":
    argLit = ArgsLit()