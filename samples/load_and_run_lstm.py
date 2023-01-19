from prescyent.predictor import LSTMPredictor, LSTMConfig, TrainingConfig
from prescyent.dataset import TeleopIcubDataset
from prescyent.evaluator.plotting import plot_prediction
from prescyent.evaluator.metrics import get_ade, get_fde


if __name__ == "__main__":

    dataset = TeleopIcubDataset()
    config = LSTMConfig(input_size=1,
                        output_size=10)
    predictor = LSTMPredictor(
        model_path="data/example/lstm_ver1")

    predictor.test(dataset.test_dataloader)

    predictor = LSTMPredictor("data/example/lstm_baseline_ver1")
    prediction = predictor(dataset.test_datasample[0][0])
    plot_prediction(dataset.test_datasample[0], prediction, "data/eval/pred_data_1.png")
    truth = dataset.test_datasample[0][1]
    print("ADE: %.5f\nFDE: %.5f" % (get_ade(truth, prediction), get_fde(truth, prediction)))
