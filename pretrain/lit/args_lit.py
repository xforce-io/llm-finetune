from omegaconf import OmegaConf
from pretrain.args import ModelArguments, DataArguments
from transformers import TrainingArguments
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

@dataclass(init=False)
class TrainingArgumentsLit(TrainingArguments):

    train_micro_batch_size_per_gpu: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "train micro batch size per gpu."
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

    warmup_min_lr: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "warmup min lr."
            )
        },
    )

    warmup_max_lr: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "warmup max lr."
            )
        },
    )

    warmup_num_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "warmup num steps."
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