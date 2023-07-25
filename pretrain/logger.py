import sys
import logging
import transformers
import datasets

log = logging.getLogger(__name__)

def initLogging(argsList):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    transformers.utils.logging.set_verbosity_info()

    log_level = argsList.trainArgs.get_process_log_level()
    log.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    log.warning(
        f"Process rank: {argsList.trainArgs.local_rank}, device: {argsList.trainArgs.device}, n_gpu: {argsList.trainArgs.n_gpu}"
        + f"distributed training: {bool(argsList.trainArgs.local_rank != -1)}, 16-bits training: {argsList.trainArgs.fp16}"
    )
    log.info(f"Training/evaluation parameters {argsList.trainArgs}")
    log.info(f"Model parameters {argsList.modelArgs}")
    log.info(f"Data parameters {argsList.dataArgs}")