from argparse import ArgumentParser

import uvicorn

from prescyent.api import PredictorApi


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000)
    args = parser.parse_args()

    app = PredictorApi(args.model_path)
    uvicorn.run(app._app, host=args.host, port=args.port)
