"""Interface for the library's Predictors
The predictor can be trained and predict
"""
import copy
from typing import Dict, Iterable, List, Union

import torch
from pydantic import BaseModel
from pytorch_lightning.loggers import TensorBoardLogger
from prescyent.evaluator.eval_result import EvaluationSummary

from prescyent.evaluator.metrics import get_ade, get_fde


class BasePredictor():
    """ base class for any predictor
        methods _build_from_config, train, get_prediction, save must be overridden by a child class
        This class initialize a tensorboard logger and, a test loop and a default run loop
    """
    log_root_path: str
    name: str
    version: int

    def __init__(self, log_root_path: str,
                 name: str = None, version: Union[str, int] = None,
                 no_sub_dir_log: bool = False) -> None:
        self.log_root_path = log_root_path
        if name is None:
            name = self.__class__.__name__
        self.name = name
        self.version = version
        self._init_logger(no_sub_dir_log)

    def _init_logger(self, no_sub_dir_log=False):
        if no_sub_dir_log:
            name = ""
            version = ""
        else:
            name = self.name
            version = self.version
        self.tb_logger = TensorBoardLogger(self.log_root_path,
                                           name=name,
                                           version=version)
        # redetermine version from tb logger logic if None
        if self.version is None:
            self.version = copy.deepcopy(self.tb_logger.version)

    def __call__(self, input_batch, history_size: int = None, history_step: int = 1,
                 future_size: int = None, output_only_future: bool = True):
        return self.run(input_batch, history_size, history_step, future_size, output_only_future)

    def __str__(self) -> str:
        return f"{self.name}_v{self.version}"

    def _build_from_config(self, config: Dict):
        """build predictor from a config"""
        raise NotImplementedError("This method must be overriden by the inherited predictor")

    def train(self, train_dataloader: Iterable,
              train_config: BaseModel = None,
              val_dataloader: Iterable = None):
        """train predictor"""
        raise NotImplementedError("This method must be overriden by the inherited predictor")

    def get_prediction(self, input_t: torch.Tensor, future_size: int):
        """run the model / algorithm for one input"""
        raise NotImplementedError("This method must be overriden by the inherited predictor")

    def save(self, save_path: str):
        """save predictor"""
        raise NotImplementedError("This method must be overriden by the inherited predictor")

    def test(self, test_dataloader: Iterable):
        """test predictor"""
        # log in tensorboard
        losses, ades, fdes = [], [], []
        for sample, truth in test_dataloader:
            # eval step
            pred = self.get_prediction(sample, len(truth[0]))
            losses.append(torch.nn.MSELoss()(pred, truth))
            ades.append(get_ade(truth, pred))
            fdes.append(get_fde(truth, pred))
        # eval epoch
        mean_loss = torch.stack(losses).mean()
        ade = torch.stack(ades).mean()
        fde = torch.stack(fdes).mean()
        self.tb_logger.experiment.add_scalar("Test/epoch_loss", mean_loss, 0)
        self.tb_logger.experiment.add_scalar("Test/ADE", ade, 0)
        self.tb_logger.experiment.add_scalar("Test/FDE", fde, 0)
        return mean_loss, ade, fde

    def run(self, input_batch: Iterable, history_size: int = None,
            history_step: int = 1, future_size: int = None,
            output_only_future: bool = True) -> List[torch.Tensor]:
        """run method/model inference on the input batch
        The output is the list of predictions for each defined subpart of the input batch,
        or the single prediction for the whole input

        Args:
            input_batch (Iterable): Input for the predictor's model
            history_size (int, optional): If an input size is provided, the input batch will
                be splitted sequences of len == history_size. Defaults to None
            history_step (int, optional): When splitting the input_batch (history_size != None)
                defines the step of the iteration. Defaults to 1.
            future_size (int|None, optional): If an input size is provided, the input batch will
                be splitted sequences of len == future_size. Defaults to None.
            output_only_future (bool, optional): If the model also outputs more than future,
                we keep only the last future_size_values. Defaults to True
        Returns:
            List[torch.Tensor]: the list of model predictions
        """
        prediction_list = []
        # if we don't split the input, the history_size is the size of input
        if history_size is None:
            history_size = input_batch.shape[0]
        # default future_size would be the size of input in the general case
        if future_size is None:
            future_size = input_batch.shape[0]

        for i in range(0, input_batch.shape[0] - history_size + 1, history_step):
            input_sub_batch = input_batch[i:i + history_size]
            prediction = self.get_prediction(input_sub_batch, future_size)
            if output_only_future and future_size and \
                    len(prediction) == history_size + future_size - 1:
                prediction = prediction[-future_size:]
            prediction_list.append(prediction)
        return prediction_list

    def log_evaluation_summary(self, evaluation_summary: EvaluationSummary):
        self.tb_logger.experiment.add_scalar("Eval/mean_ade", evaluation_summary.mean_ade, 0)
        self.tb_logger.experiment.add_scalar("Eval/mean_fde", evaluation_summary.mean_fde, 0)
        self.tb_logger.experiment.add_scalar("Eval/mean_inference_time_ms",
                                             evaluation_summary.mean_inference_time_ms, 0)
        self.tb_logger.experiment.add_scalar("Eval/max_ade", evaluation_summary.max_ade, 0)
        self.tb_logger.experiment.add_scalar("Eval/max_fde", evaluation_summary.max_fde, 0)
        self.tb_logger.experiment.add_scalar("Eval/max_inference_time_ms",
                                             evaluation_summary.max_inference_time_ms, 0)
