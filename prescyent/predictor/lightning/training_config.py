from typing import Union

from pydantic import BaseModel


class TrainingConfig(BaseModel):
    epoch: int = 50
    accelerator: str = "gpu"
    devices: Union[str, int] = 2
    # TODO see pl trainer args for more
