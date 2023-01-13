from typing import Union

from pydantic import BaseModel


class TrainingConfig(BaseModel):
    epoch: int = 50
    accelerator: str = "auto"
    devices: Union[str, int] = "auto"
    # TODO see pl trainer args for more
