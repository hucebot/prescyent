from pydantic import BaseModel


class LinearConfig(BaseModel):
    feature_size: int
    input_size: int
    output_size: int
    identifier: str = "linear"
    model_path: str = "data/models"
