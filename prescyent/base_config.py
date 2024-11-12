"""custom class for all our config classes"""
from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """custom class for all our config classes, raising an error for any extra attribute"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    """Metadata about the fields defined on the model"""
