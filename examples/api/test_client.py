from argparse import ArgumentParser

from prescyent.api import PredictorClient
from prescyent.auto_dataset import AutoDataset
from prescyent.evaluator.metrics import get_mpjpe


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", default="data/models/example/LinearPredictor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000)
    args = parser.parse_args()

    dataset = AutoDataset.build_from_config(args.model_path)
    client = PredictorClient(host=args.host, port=args.port)
    input, truth = dataset.test_datasample[0]
    prediction = client.get_prediction(input, future_size=10)
    print(f"MPJPE of returned prediction: {get_mpjpe(truth=truth, pred=prediction)}")
    print(client.update_predictor(args.model_path))
    print(f"Updated predictor to {args.model_path}")
    prediction = client.get_prediction(input, future_size=10)
    print(f"MPJPE of returned prediction: {get_mpjpe(truth=truth, pred=prediction)}")
