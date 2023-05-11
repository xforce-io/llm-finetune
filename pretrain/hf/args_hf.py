from transformers import TrainingArguments
from transformers.deepspeed import HfTrainerDeepSpeedConfig
from ..args import ModelArguments, DataTrainingArguments

class ArgsHf:
    def __init__(self, modelArgs, dataArgs, trainArgs) -> None:
        self.modelArgs = modelArgs
        self.dataArgs = dataArgs
        self.trainArgs = trainArgs
        self.trainArgs.deepspeed = dataArgs.deepspeed_config
        self.trainArgs.hf_deepspeed_config = HfTrainerDeepSpeedConfig(dataArgs.deepspeed_config)
        self.trainArgs.remove_unused_columns = False
        self.trainArgs.warmup_steps = 1000

    def modelArgs(self) -> ModelArguments:
        return self.modelArgs

    def dataArgs(self) -> DataTrainingArguments:
        return self.dataArgs
    
    def trainArgs(self) -> TrainingArguments:
        return self.trainArgs