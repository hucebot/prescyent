"""Interface for the library's Predictors
The predictor can be trained and predict
"""
import copy
from typing import Dict, Iterable, List, Union

import torch
from pydantic import BaseModel
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

from prescyent.evaluator.eval_summary import EvaluationSummary
from prescyent.evaluator.metrics import get_ade, get_fde, get_mpjpe
from prescyent.utils.logger import logger, PREDICTOR


class BasePredictor:
    """base class for any predictor
    methods _build_from_config, train, predict, save must be overridden by a child class
    This class initialize a tensorboard logger and, a test loop and a default run loop
    """

    log_root_path: str
    name: str
    version: int

    def __init__(
        self,
        log_root_path: str,
        name: str = None,
        version: Union[str, int] = None,
        no_sub_dir_log: bool = False,
    ) -> None:
        self.log_root_path = log_root_path
        if name is None:
            name = self.__class__.__name__
        self.name = name
        self.version = version
        self._init_logger(no_sub_dir_log)
        logger.info(f"Loaded {self} at path {self.log_path}", group=PREDICTOR)

    def _init_logger(self, no_sub_dir_log=False):
        if no_sub_dir_log:
            name = ""
            version = ""
        else:
            name = self.name
            version = self.version
        self.tb_logger = TensorBoardLogger(
            self.log_root_path, name=name, version=version, default_hp_metric=False
        )
        # redetermine version from tb logger logic if None
        if self.version is None:
            self.version = copy.deepcopy(self.tb_logger.version)

    @property
    def log_path(self) -> str:
        return self.tb_logger.log_dir

    def __call__(
        self,
        input_batch,
        future_size: int = None,
        history_size: int = None,
        history_step: int = 1,
    ):
        return self.run(input_batch, future_size, history_size, history_step)

    def __str__(self) -> str:
        return f"{self.name}_v{self.version}"

    def _build_from_config(self, config: Dict):
        """build predictor from a config"""
        raise NotImplementedError(
            "This method must be overriden by the inherited predictor"
        )

    def train(
        self,
        train_dataloader: Iterable,
        train_config: BaseModel = None,
        val_dataloader: Iterable = None,
    ):
        """train predictor"""
        raise NotImplementedError(
            "This method must be overriden by the inherited predictor"
        )

    def finetune(
        self,
        train_dataloader: Iterable,
        train_config: BaseModel = None,
        val_dataloader: Iterable = None,
    ):
        """finetune predictor"""
        raise NotImplementedError(
            "This method must be overriden by the inherited predictor"
        )

    def predict(self, input_t: torch.Tensor, future_size: int) -> torch.Tensor:
        """run the model / algorithm for one input"""
        raise NotImplementedError(
            "This method must be overriden by the inherited predictor"
        )

    def save(self, save_path: str):
        """save predictor"""
        raise NotImplementedError(
            "This method must be overriden by the inherited predictor"
        )

    def test(self, test_dataloader: Iterable):
        """test predictor"""
        # log in tensorboard
        losses, ades, fdes, mpjpes = [], [], [], []
        for sample, truth in test_dataloader:
            # eval step
            pred = self.predict(sample, len(truth[0]))
            losses.append(torch.nn.MSELoss()(truth, pred).detach())
            ades.append(get_ade(truth, pred).detach())
            fdes.append(get_fde(truth, pred).detach())
            mpjpes.append(get_mpjpe(truth, pred).detach()[-1])
        # eval epoch
        mean_loss = torch.stack(losses).mean()
        ade = torch.stack(ades).mean()
        fde = torch.stack(fdes).mean()
        mpjpe = torch.stack(mpjpes).mean()
        self.tb_logger.experiment.add_scalar("Test/epoch_loss", mean_loss, 0)
        self.tb_logger.experiment.add_scalar("Test/ADE", ade, 0)
        self.tb_logger.experiment.add_scalar("Test/FDE", fde, 0)
        self.tb_logger.experiment.add_scalar("Test/MPJPE", mpjpe, 0)
        return mean_loss, ade, fde, mpjpe

    def run(
        self,
        input_batch: Iterable,
        future_size: int = None,
        history_size: int = None,
        history_step: int = 1,
    ) -> List[torch.Tensor]:
        """run method/model inference over the input batch
        The run method outputs a List of prediction because it can iterate over the input_batch
        according to the history_size and history step values.

        Args:
            input_batch (Iterable): Input for the predictor's model. We expect the first
                dimension of the array to be the temporal axis: len(input_batch) == sequence_len
            future_size (int|None, optional): If an future size is provided, the input batch will
                be splitted sequences of len == future_size. Defaults to None.
            history_size (int, optional): If an history_size is provided, the input batch will
                be splitted sequences of len == history_size. Defaults to None means no split.
            history_step (int, optional): When splitting the input_batch (history_size != None)
                defines the step of the iteration. Defaults to 1.
        Returns:
            List[torch.Tensor]: the list of model predictions
        """
        prediction_list = []
        input_len = len(input_batch)
        # if we don't split the input, the history_size is the size of input
        if history_size is None:
            history_size = input_len
        # default future_size would be the size of input in the general case
        if future_size is None:
            future_size = input_len

        for i in tqdm(
            range(0, input_len - history_size + 1, history_step),
            desc="Iterate over input_batch",
        ):
            input_sub_batch = input_batch[i : i + history_size]
            prediction = self.predict(input_sub_batch, future_size)
            prediction_list.append(prediction)
        return prediction_list

    def log_evaluation_summary(self, evaluation_summary: EvaluationSummary):
        self.tb_logger.experiment.add_scalar(
            "Eval/mean_ade", evaluation_summary.mean_ade, 0
        )
        self.tb_logger.experiment.add_scalar(
            "Eval/mean_fde", evaluation_summary.mean_fde, 0
        )
        self.tb_logger.experiment.add_scalar(
            "Eval/mean_mpjpe", evaluation_summary.mean_mpjpe, 0
        )
        self.tb_logger.experiment.add_scalar(
            "Eval/mean_inference_time_ms", evaluation_summary.mean_inference_time_ms, 0
        )
        self.tb_logger.experiment.add_scalar(
            "Eval/max_ade", evaluation_summary.max_ade, 0
        )
        self.tb_logger.experiment.add_scalar(
            "Eval/max_fde", evaluation_summary.max_fde, 0
        )
        self.tb_logger.experiment.add_scalar(
            "Eval/max_mpjpe", evaluation_summary.max_mpjpe, 0
        )
        self.tb_logger.experiment.add_scalar(
            "Eval/max_inference_time_ms", evaluation_summary.max_inference_time_ms, 0
        )

    def log_metrics(self, metrics: dict, pre_key=""):
        for key, value in metrics.items():
            if isinstance(value, dict):
                self.log_metrics(value, pre_key=pre_key + key)
                continue
            if isinstance(value, list):
                for i, j in enumerate(value):
                    self.tb_logger.experiment.add_scalar(pre_key + key, j, i)
                continue
            self.tb_logger.experiment.add_scalar(pre_key + key, value)
