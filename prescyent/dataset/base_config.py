from pydantic import BaseModel


class DatasetConfig(BaseModel):
    data_path: str
    batch_size = 16
    shuffle = True
    # TODO see torch Dataset args for more
