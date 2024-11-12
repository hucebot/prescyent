"""Config elements for MLP Pytorch Lightning module usage"""
from typing import Optional
from pydantic import Field, model_validator

from prescyent.predictor.lightning.configs.module_config import ModuleConfig
from prescyent.utils.enums import ActivationFunctions


class MlpConfig(ModuleConfig):
    """Pydantic Basemodel for MLP Module configuration"""

    hidden_size: int = 64
    """Size of the hidden FC Layers in the MLP"""
    num_layers: int = Field(2, gt=0)
    """Number of FC layers in the MLP"""
    activation: Optional[ActivationFunctions] = ActivationFunctions.RELU
    """Activation function used between layers"""
    context_size: Optional[int] = None
    """Number of features of the context tensors used as inputs. See dataset.context_size_sum"""

    @model_validator(mode="after")
    def check_context_attributes(self):
        if (
            self.context_size is None
            and self.dataset_config.context_keys
            or self.context_size is not None
            and not self.dataset_config.context_keys
        ):
            raise AttributeError(
                "If we have a context as input, it keys should be described "
                "for the dataset to filter the inputs, and the shape of the "
                "tensors' features should be given to the MLP to init its first layer"
            )
        return self
