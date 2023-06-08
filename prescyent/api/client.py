import json
import requests
import torch

class APIException(Exception):
    pass

class PredictorClient():
    host: str
    port: int

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    @property
    def url(self):
        return f"http://{self.host}:{self.port}/"

    def _get(self, url, payload=None):
        response = requests.get(url, timeout=10, json=payload)
        try:
            content = json.loads(response.text)
        except json.JSONDecodeError:
            content = str(response.text)
        if response.status_code != 200:
            raise APIException(f"API returned status code {response.status_code} with: `{content}`")
        return content

    def update_predictor(self, model_path):
        url = self.url + f"update_predictor?predictor_path={model_path}"
        self._get(url)

    def get_prediction(self, tensor, future_size):
        payload = tensor.tolist()
        url = self.url + f"?future={future_size}"
        return torch.tensor(self._get(url, payload))
