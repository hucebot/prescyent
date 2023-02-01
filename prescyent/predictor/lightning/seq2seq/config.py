from pydantic import BaseModel


class Seq2SeqConfig(BaseModel):
    feature_size: int
    output_size: int
    identifier: str = "seq2seq"
    hidden_size: int = 10
    model_path: str = "data/models"
    num_layers: int = 1
