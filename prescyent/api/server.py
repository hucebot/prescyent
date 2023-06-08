from functools import wraps
from prescyent.utils.errors.custom_exception import CustomException

import torch
from fastapi import FastAPI, APIRouter, HTTPException

from prescyent import get_predictor_from_path
from prescyent.predictor.base_predictor import BasePredictor


def return_http_except(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except CustomException as custom_exception:
            raise HTTPException(
                custom_exception.code,
                f"{custom_exception.__class__.__name__}: {custom_exception}",
            ) from custom_exception
        except Exception as generic_exception:
            raise HTTPException(
                500, f"{generic_exception.__class__.__name__}: {generic_exception}"
            ) from generic_exception

    return wrapper


class PredictorApi:
    def __init__(self, predictor_path: BasePredictor) -> None:
        self._app = FastAPI()
        self.predictor = get_predictor_from_path(predictor_path)
        self.router = APIRouter()
        self.router.add_api_route("/", self.predict, methods=["GET"])
        self.router.add_api_route(
            "/update_predictor", self.update_predictor, methods=["GET"]
        )
        self._app.include_router(self.router)

    @return_http_except
    async def predict(self, tensor: list, future: int):
        tensor = torch.tensor(tensor)
        return self.predictor.predict(tensor, future_size=future).tolist()

    @return_http_except
    async def update_predictor(self, predictor_path=None):
        self.predictor = get_predictor_from_path(predictor_path)
        return str(self.predictor)
