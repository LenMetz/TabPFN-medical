from pathlib import Path

import pandas as pd
import torch
import torch.multiprocessing as mp
from loguru import logger
from sklearn.base import BaseEstimator

from tabularbench.config.config_pretrain import ConfigPretrain
from tabularbench.core.enums import BenchmarkName, DataSplit, Phase
from tabularbench.core.get_model import get_model_pretrain
from tabularbench.core.get_optimizer import get_optimizer_pretrain
from tabularbench.core.get_scheduler import get_scheduler_pretrain
from tabularbench.core.losses import CrossEntropyLossExtraBatch
from tabularbench.core.metrics import MetricsTraining, MetricsValidation
from tabularbench.core.trainer_pretrain_evaluate import create_config_benchmark_sweep
from tabularbench.core.trainer_pretrain import TrainerPretrain
from tabularbench.core.trainer_pretrain_init import (create_synthetic_dataloader, create_synthetic_dataset,
                                                     log_parameter_count, prepare_ddp_model)
from tabularbench.data.benchmarks import BENCHMARKS
from tabularbench.sweeps.run_sweep import run_sweep
from tabularbench.utils.paths_and_filenames import DEFAULT_RESULTS_TEST_FILE_NAME, DEFAULT_RESULTS_VAL_FILE_NAME


class TrainerPretrainMicrobiome(TrainerPretrain):

    def validate():
        print("No validation implemented yet!")