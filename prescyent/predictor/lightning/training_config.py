"""Common config elements for Pytorch Lightning training usage"""
from typing import Union

from pydantic import BaseModel


class TrainingConfig(BaseModel):
    """Pydantic Basemodel for Pytorch Lightning Training configuration"""
    epoch: int = 100
    accelerator: str = "auto"
    devices: Union[str, int] = "auto"
    # TODO see pl trainer args for more
