from pydantic import BaseModel


class LSTMConfig(BaseModel):
    feature_size: int
    output_size: int
    identifier: str = "lstm"
    hidden_size: int = 10
    model_path: str = "data/models"
    num_layers: int = 1
