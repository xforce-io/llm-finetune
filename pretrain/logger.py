import sys
import logging
import transformers
import datasets

log = logging.getLogger(__name__)

def initLogging(trainArgs):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if trainArgs.should_log:
        # The default of trainArgs.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = trainArgs.get_process_log_level()
    log.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    log.warning(
        f"Process rank: {trainArgs.local_rank}, device: {trainArgs.device}, n_gpu: {trainArgs.n_gpu}"
        + f"distributed training: {bool(trainArgs.local_rank != -1)}, 16-bits training: {trainArgs.fp16}"
    )
    log.info(f"Training/evaluation parameters {trainArgs}")