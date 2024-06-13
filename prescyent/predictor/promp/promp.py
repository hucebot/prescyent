# for a more complete implementation : https://pypi.org/project/mp-pytorch/
from typing import Dict, Optional,Union
from tqdm import tqdm
import torch

from prescyent.dataset.config import MotionDatasetConfig
from prescyent.dataset.dataset import MotionDataset
from prescyent.predictor.base_predictor import BasePredictor
from prescyent.utils.logger import logger, PREDICTOR
from prescyent.utils.tensor_manipulation import is_tensor_is_batched

from prescyent.predictor.promp.single_promp import SinglePromp

#plt.style.use('_mpl-gallery')


class PrompPredictor(BasePredictor):
    """ Probabilistic Motion Primitives (1-dimensional per dimension) [ProMP] """
    def __init__( self,
        dataset_config: MotionDatasetConfig,
        log_root_path: str = 'data/models',
        name: str = None,
        version: Union[str, int, None] = None,
        no_sub_dir_log: bool = False,
    ) -> None:
        super().__init__(dataset_config, log_root_path, name, version, no_sub_dir_log)
        self.promps = []
        self.length = -1

    def train(self, data: MotionDataset, train_config: Dict = None):
        # we make a single PromMP for each value to predict
        num_points = data.trajectories.train[0].tensor.size(1)
        self.promps = []
        for p in tqdm(range(num_points)): # num of points (left_hand, right_hand, etc)
            num_dim_per_point = data.trajectories.train[0].tensor.size(2)
            promp_list = []
            for d in tqdm(range(num_dim_per_point),colour='blue',leave=True): # dim of each point
                traj_list = []
                for traj in data.trajectories.train:
                    traj_list += [traj.tensor[:, p, d]]
                promp = SinglePromp()
                promp.train(traj_list)
                promp_list += [promp]
            self.promps += [promp_list]
        self.length = self.promps[0][0].s # all promps should have the same size...

    def plot(self):
        for i, p in enumerate(self.promps):
            for k, pi in enumerate(p):
                pi.plot(str(i) + '_' + str(k) + '_')

    def predict(self, input_t: torch.Tensor, future_size: int):
        raise NotImplementedError

    def predict_by_conditioning(self, input_t: torch.Tensor, future_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = input_t.size(0) + future_size # how much in the future do we predict
        result_mean = torch.zeros(input_t.size(1), input_t.size(2))
        result_std = torch.zeros_like(result_mean)
        for p in range(input_t.size(1)): # num of points (left_hand, right_hand, etc)
            for d in range(input_t.size(2)): # dim of each point
                promp = self.promps[p][d].condition(input_t[:,p,d])
                t_real = min(t, promp.s - 1)
                result_mean[p, d] = promp.mean()[t_real]
                result_std[p, d] = promp.std()[t_real]
        return (result_mean, result_std)
        