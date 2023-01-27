import torch

from prescyent.dataset.motion.teleop_icub.config import TeleopIcubDatasetConfig
from prescyent.predictor import LSTMPredictor, LSTMConfig, TrainingConfig
from prescyent.dataset import TeleopIcubDataset
from prescyent.evaluator.plotting import plot_prediction
from prescyent.evaluator.metrics import get_ade, get_fde


if __name__ == "__main__":

    subsampling_step: int = 10     # subsampling -> 100 Hz to 10Hz
    input_size = 10  # 1 second
    output_size = 10  # 1 second
    dataset_config = TeleopIcubDatasetConfig(input_size=input_size,
                                             output_size=output_size,
                                             dimensions=[1, 2, 3],
                                             subsampling_step=subsampling_step)
    dataset = TeleopIcubDataset(dataset_config)

    predictor = LSTMPredictor(model_path="data/example/lstm_ver1")

    predictor.test(dataset.test_dataloader)
    input = dataset.test_datasample[0][0]
    truth = dataset.test_datasample[0][1]
    prediction = predictor(input)
    plot_prediction(data_sample=dataset.test_datasample[0],
                    pred=prediction,
                    savefig_path="data/example/lstm_ver1/pred_data_1.png")
    truth = dataset.test_datasample[0][1]
    print("ADE: %.5f\nFDE: %.5f" % (get_ade(truth, prediction).item(),
                                    get_fde(truth, prediction).item()))
