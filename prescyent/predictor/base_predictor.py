"""Interface for the library's Predictors
The predictor can be trained and predict
"""
import copy
import functools
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from pydantic import BaseModel
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

from prescyent.dataset import Trajectory
from prescyent.dataset.features import Features
from prescyent.dataset.features.feature_manipulation import (
    cal_distance_for_feat,
    convert_tensor_features_to,
)
from prescyent.evaluator.eval_summary import EvaluationSummary
from prescyent.predictor.config import PredictorConfig
from prescyent.scaler.scaler import Scaler
from prescyent.utils.enums import LearningTypes
from prescyent.utils.logger import logger, PREDICTOR
from prescyent.utils.dataset_manipulation import update_parent_ids
from prescyent.utils.tensor_manipulation import (
    is_tensor_is_batched,
    cat_list_with_seq_idx,
)


class BasePredictor:
    """base class for any predictor
    methods _build_from_config, train, predict, save must be overridden by a child class
    This class initialize a tensorboard logger, an optionnal scaler, a test loop and a default run loop
    """

    config: Optional[PredictorConfig]
    """Configuration of the dataset, it is used to shape input and outputs
    If not provided, we cannot perform a fair evaluation of the predictor
    so the test method will raise an error
    """
    log_root_path: str
    """root path where the predictor should log"""
    name: str
    """name of the predictor"""
    version: int
    """version number to describe a given instance of the model"""
    tb_logger: TensorBoardLogger
    """Instance of a TensorBoardLogger create .event files for tests"""
    scaler: Scaler
    """Instance of a prescyent.Scaler"""

    def __init__(
        self,
        config: Optional[PredictorConfig],
        no_sub_dir_log: bool = False,
    ) -> None:
        self.config = config
        self.log_root_path = config.save_path
        if self.config.name is None:
            self.config.name = self.__class__.__name__
        self.name = self.config.name
        self.version = self.config.version
        self._init_logger(no_sub_dir_log)
        if (
            not hasattr(self, "scaler") or self.scaler is None
        ):  # If scaler wasn't _init by child
            self._init_scaler()

    def describe(self):
        _str = f"""\n{2*"    "}Predictor {self} is initialized with the following parameters:
            - Log path: {self.log_path}\n"""
        if self.scaler:
            _str += f"""{3*"    "}- Scaler:
                {self.scaler.describe()}\n"""
        logger.getChild(PREDICTOR).info(_str)

    def _init_scaler(self):
        """create instance of prescyent.Scaler"""
        if self.config and self.config.scaler_config:
            self.scaler = Scaler(self.config.scaler_config)
        else:
            self.scaler = None

    def _init_logger(self, no_sub_dir_log=False):
        """create instance of TensorBoardLogger for the predictor

        Args:
            no_sub_dir_log (bool, optional): If True, log files will be written at root_path
            If False. log files will be written at root_path/predictor.name/predictor.version
            Defaults to False.
        """
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
        input_tensor: torch.Tensor,
        future_size: Optional[int] = None,
        history_size: Optional[int] = None,
        history_step: int = 1,
        input_tensor_features: Optional[Features] = None,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Call on the predictor calls the run() method

        Args:
            input_tensor (torch.Tensor): Input for the predictor's model. Can be batched or unbatched
            future_size (int, optional): Defines the output sequence size of the model.
                Defaults to predictor's value from config if None.
            history_size (int, optional): The input batch will be splitted sequences of len == history_size. Defaults to predictor's value from config if None.
            history_step (int, optional): When splitting the input_tensor (history_step != None)
                defines the step of the iteration over the input_tensor's frames. Defaults to 1.
            input_tensor_features: (Features, optional) Features describing the input_tensor. If not None, it is used to convert input to self.config.in_features. Defaults to None,
            context: (Dict[str, torch.Tensor], optional) Additional context alongside input_tensor. Defaults to None,
        Returns:
            List[torch.Tensor]: the list of model predictions
        """
        return self.run(
            input_tensor=input_tensor,
            future_size=future_size,
            history_size=history_size,
            history_step=history_step,
            input_tensor_features=input_tensor_features,
            context=context,
        )

    def __str__(self) -> str:
        return f"{self.name}_v{self.version}"

    def _build_from_config(self, config: Dict):
        """build predictor from a config"""
        raise NotImplementedError(
            "This method must be overriden by the inherited predictor"
        )

    def train(
        self,
        datamodule: LightningDataModule,
        train_config: BaseModel = None,
    ):
        """train scaler in base class, you'll need to train specific predictors into child classes

        Args:
            datamodule (LightningDataModule): instance of a TrajectoriesDataset
            train_config (BaseModel, optional): configuration for the training. Defaults to None.
        """
        if self.scaler:
            logger.getChild(PREDICTOR).info("Training Scaler")
            self.train_scaler(datamodule)
            logger.getChild(PREDICTOR).info("Scaler Trained")

    def train_scaler(self, datamodule: LightningDataModule):
        """train the scaler given a dataloader over all train trajectories and the predictor config

        Args:
            datamodule (LightningDataModule): instance of a TrajectoriesDataset
        """
        # cat all dataset's frames
        dataset_tensor = torch.cat(
            [traj.tensor for traj in datamodule.trajectories.train], dim=0
        )
        self.scaler.train(
            DataLoader(
                dataset_tensor,
                batch_size=datamodule.config.batch_size,
                num_workers=datamodule.config.num_workers,
            ),
            dataset_features=datamodule.tensor_features,
        )

    def finetune(
        self,
        datamodule: LightningDataModule,
        train_config: BaseModel = None,
    ):
        """finetune predictor

        Args:
            datamodule (LightningDataModule): TrajectoriesDataset
            train_config (BaseModel, optional): config for the training. Defaults to None.

        Raises:
            NotImplementedError: override this method in predictors that can be finetuned
        """
        raise NotImplementedError(
            "This method must be overriden by the inherited predictor"
        )

    def predict(
        self,
        input_t: torch.Tensor,
        future_size: int,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """run the model / algorithm for one input

        Args:
            input_t (torch.Tensor): tensor to predict over
            future_size (int): number of the expected predicted frames
            context (Optional[Dict[str, torch.Tensor]], optional): additional context. Defaults to None.

        Raises:
            NotImplementedError: must ovveride in child predictor classes

        Returns:
            torch.Tensor: predicted tensor
        """
        raise NotImplementedError(
            "This method must be overriden by the inherited predictor"
        )

    def save(
        self,
        save_path: Union[str, Path, None] = None,
        rm_log_path: bool = True,
    ):
        """save predictor, must be rewriten for more complex classes, here we just move the logs if any

        Args:
            save_path (Union[str, Path, None]): save path
            rm_log_path (bool) if True, remove previous log files after copy:
        """
        if (
            save_path is None
            or Path(self.log_path).resolve() == Path(save_path).resolve()
        ):
            return
        save_path = str(save_path)
        while True:
            try:
                shutil.copytree(
                    self.log_path,
                    save_path,
                    ignore=shutil.ignore_patterns("checkpoints"),
                )
                break
            except FileExistsError:
                # do not erase previous logs, add "_" to name instead
                save_path += "_"
            except FileNotFoundError:
                # Nothing to copy
                break
        if rm_log_path:
            shutil.rmtree(self.log_path, ignore_errors=True)

    def test(self, datamodule: LightningDataModule) -> Dict[str, torch.Tensor]:
        """test predictor over the datamodule's test set

        Args:
            datamodule (LightningDataModule): TrajectoryDataset instance

        Raises:
            NotImplementedError: if the predictor hasn't config attribute we may not perform a fair evaluation

        Returns:
            Dict[str, torch.Tensor]: dict with metric name and value
        """
        if self.config is None:
            raise NotImplementedError(
                "We cannot perform a fair evaluation of this predictor without the config"
            )
        distances = list()
        features = self.config.dataset_config.out_features
        for sample, context, truth in tqdm(
            datamodule.test_dataloader(),
            desc=f"Testing {self}:",
            colour="yellow",
        ):
            # eval step
            feat2distances = dict()
            pred = self.predict(sample, self.config.dataset_config.future_size, context)
            feat2distances["mse_loss"] = torch.nn.functional.mse_loss(pred, truth)
            for feat in features:
                feat2distances[feat.name] = cal_distance_for_feat(
                    pred[..., feat.ids], truth[..., feat.ids], feat
                ).detach()
            distances.append(feat2distances)
        # eval epoch
        losses = dict()
        mean_loss = torch.stack([x["mse_loss"] for x in distances]).mean().detach()
        losses["Test/mse_loss_epoch"] = mean_loss
        output_seq_size = (
            self.config.dataset_config.future_size
            if self.config.dataset_config.learning_type != LearningTypes.SEQ2ONE
            else 1
        )
        for feat in features:
            batch_feat_distances = torch.cat(
                [feat2distances[feat.name] for feat2distances in distances]
            )
            ade = batch_feat_distances.mean()
            fde = batch_feat_distances[:, -1].mean()
            mpjpe = (
                batch_feat_distances.transpose(0, 1)
                .reshape(output_seq_size, -1)
                .mean(-1)
            )
            losses[f"Test/{feat.name}/ADE"] = ade
            losses[f"Test/{feat.name}/FDE"] = fde
            losses[f"Test/{feat.name}/MPJPE"] = mpjpe
        for key, value in losses.items():
            if value.ndim > 0 and len(value) > 1:  # if tensor is iterable
                for v, val in enumerate(value):
                    self.tb_logger.experiment.add_scalar(key, val, v + 1)
            else:
                self.tb_logger.experiment.add_scalar(key, value, 0)
        return losses

    def run(
        self,
        input_tensor: torch.Tensor,
        future_size: Optional[int] = None,
        history_size: Optional[int] = None,
        history_step: int = 1,
        input_tensor_features: Optional[Features] = None,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """run method/model inference over an unbatched or batched input sequence
        The run method outputs a List of prediction because it can iterate over the input_tensor
        according to the history_size and history step values.

        Args:
            input_tensor (torch.Tensor): Input for the predictor's model. Can be batched or unbatched
            future_size (int, optional): Defines the output sequence size of the model.
                Defaults to predictor's value from config if None.
            history_size (int, optional): The input batch will be splitted sequences of len == history_size. Defaults to predictor's value from config if None.
            history_step (int, optional): When splitting the input_tensor (history_step != None)
                defines the step of the iteration over the input_tensor's frames. Defaults to 1.
            input_tensor_features: (Features, optional) Features describing the input_tensor. If not None, it is used to convert input to self.config.in_features. Defaults to None,
            context: (Dict[str, torch.Tensor], optional) Additional context alongside input_tensor. Defaults to None,
        Returns:
            List[torch.Tensor]: the list of model predictions
        """
        prediction_list = []
        if context is None:
            context = {}
        # allows batched or unbatched inputs, returns batched or unbatched prediction accordingly
        unbatch = False
        if not is_tensor_is_batched(input_tensor):
            unbatch = True
            input_tensor = torch.unsqueeze(input_tensor, 0)
            context = {
                c_name: c_tensor.unsqueeze(0) for c_name, c_tensor in context.items()
            }
        # Keep only model input points if the input shape doesn't match
        if input_tensor.shape[2] != len(self.config.dataset_config.in_points):
            input_tensor = input_tensor[:, :, self.config.dataset_config.in_points]
        # Else we assume the tensor was already reshaped
        # If we know tensor feats, we make them fit expected inputs.
        if input_tensor_features is not None:
            input_tensor = convert_tensor_features_to(
                input_tensor,
                input_tensor_features,
                self.config.dataset_config.in_features,
            )
        # Else we assume feats are in the right format
        context_sub_batch = None  # init sub_batch of the context to None
        input_len = input_tensor.shape[1]
        if future_size is None:
            future_size = self.config.dataset_config.future_size
        if history_size is None:
            history_size = self.config.dataset_config.history_size
        max_iter = (
            input_len - history_size + 1
            if not self.config.dataset_config.loop_over_traj
            else input_len
        )
        # If we don't step over input, output only last pred
        if history_step is None:
            input_sub_batch = input_tensor[:, -history_size:]
            if context:  # If context we iterate over it along the input
                context_sub_batch = {
                    c_name: c_tensor[:, -history_size:]
                    for c_name, c_tensor in context.items()
                }
            prediction = self.predict(
                input_sub_batch, future_size=future_size, context=context_sub_batch
            )
            if unbatch:
                prediction = prediction.squeeze(0)
            prediction_list.append(prediction)
            return prediction_list
        for i in tqdm(
            range(0, max_iter, history_step),
            desc=f"{self} iterating over input_tensor",
            colour="yellow",
        ):
            if (
                i + history_size > input_len
                and self.config.dataset_config.loop_over_traj
            ):
                input_sub_batch = torch.cat(
                    (
                        input_tensor[:, i:],
                        input_tensor[:, : history_size - input_tensor[:, i:].shape[1]],
                    ),
                    1,
                )
                if context:  # If context we iterate over it along the input
                    context_sub_batch = {
                        c_name: torch.cat(
                            (
                                c_tensor[:, i:],
                                c_tensor[:, : history_size - c_tensor[:, i:].shape[1]],
                            ),
                            1,
                        )
                        for c_name, c_tensor in context.items()
                    }
            else:
                input_sub_batch = input_tensor[:, i : i + history_size]
                if context:  # If context we iterate over it along the input
                    context_sub_batch = {
                        c_name: c_tensor[:, i : i + history_size]
                        for c_name, c_tensor in context.items()
                    }
            if self.config.dataset_config.learning_type == LearningTypes.SEQ2ONE:
                future_size = 1
            prediction = self.predict(
                input_sub_batch, future_size=future_size, context=context_sub_batch
            )
            if unbatch:
                prediction = prediction.squeeze(0)
            prediction_list.append(prediction)
        return prediction_list

    def log_evaluation_summary(self, evaluation_summary: EvaluationSummary):
        """save evaluation summary to tensorboard

        Args:
            evaluation_summary (EvaluationSummary): results of an evaluation
        """
        for feat in evaluation_summary.features:
            self.tb_logger.experiment.add_scalar(
                f"Eval/{feat.name}/average_prediction_error",
                evaluation_summary.average_prediction_error[feat.name],
                0,
            )
            self.tb_logger.experiment.add_scalar(
                f"Eval/{feat.name}/max_prediction_error",
                evaluation_summary.max_prediction_error[feat.name],
                0,
            )
        self.tb_logger.experiment.add_scalar(
            "Eval/mean_rtf", evaluation_summary.mean_rtf, 0
        )
        self.tb_logger.experiment.add_scalar(
            "Eval/max_rtf", evaluation_summary.max_rtf, 0
        )
        self.tb_logger.save()

    def log_metrics(self, metrics: dict, pre_key=""):
        """log dict metrics to tensorboard, even nested values

        Args:
            metrics (dict): dict of metrics
            pre_key (str, optional): key to add as a prefix to each logged metric. Defaults to "".
        """
        for key, value in metrics.items():
            if isinstance(value, dict):
                self.log_metrics(value, pre_key=pre_key + key)
                continue
            if isinstance(value, list):
                for i, j in enumerate(value):
                    self.tb_logger.experiment.add_scalar(pre_key + key, j, i)
                continue
            self.tb_logger.experiment.add_scalar(pre_key + key, value)

    def predict_trajectory(
        self, traj: Trajectory, future_size=None
    ) -> Tuple[Trajectory, int]:
        """Returns new traj with predicted tensor and offset that can be used to compare new traj and base traj
            Behavior here is to run the predictor with a history_step of 1 over the input
            and turn the list of predictions into one predicted trajectory from latest frame predicted at each prediction
        Args:
            traj (Trajectory): the trajectory to predict over
            future_size (_type_, optional): _description_. Defaults to None.

        Returns:
            Tuple[Trajectory, int]: Predicted Trajectory and the offset between the input traj and predicted traj to perform some evaluation
        """
        list_pred_tensor = self.run(
            input_tensor=traj.tensor,
            future_size=future_size,
            history_size=None,
            history_step=1,
            input_tensor_features=traj.tensor_features,
            context=traj.context,
        )
        pred_tensor = cat_list_with_seq_idx(list_pred_tensor, -1)
        offset = (
            self.config.dataset_config.history_size
            + self.config.dataset_config.future_size
            - 1
        )
        pred_traj = Trajectory(
            tensor=pred_tensor,
            tensor_features=self.config.dataset_config.out_features,
            context={},
            frequency=traj.frequency,
            file_path=traj.file_path,
            title=f"{traj.title}_pred_{self.name}",
            point_parents=update_parent_ids(
                self.config.dataset_config.out_points, traj.point_parents
            ),
            point_names=[
                traj.point_names[i] for i in self.config.dataset_config.out_points
            ],
        )
        return (pred_traj, offset)

    @staticmethod
    def use_scaler(function):
        """decorator for predictor level normalization using configured self.scaler"""

        @functools.wraps(function)
        def scale(self, input_t, *args, **kwargs):
            if not self.scaler:
                return function(self, input_t, *args, **kwargs)
            unbatched = False
            if len(input_t.shape) == 3:
                unbatched = True
                input_t = input_t.unsqueeze(0)
            input_t = self.scaler.scale(
                input_t.clone(),
                self.config.dataset_config.in_points,
                self.config.dataset_config.in_features,
            )
            output_t = function(self, input_t, *args, **kwargs)
            output_t = self.scaler.unscale(
                output_t.clone(),
                self.config.dataset_config.out_points,
                self.config.dataset_config.out_features,
            )
            if unbatched:
                return output_t.squeeze(0)
            return output_t

        return scale
