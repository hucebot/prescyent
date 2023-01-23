from pathlib import Path
from prescyent.evaluator.metrics import get_ade, get_fde
from prescyent.evaluator.plotting import plot_prediction
from prescyent.predictor import LSTMPredictor, LSTMConfig, TrainingConfig
from prescyent.dataset import TeleopIcubDataset, TeleopIcubDatasetConfig


if __name__ == "__main__":
    # -- Init dataset
    subsampling_step: int = 10      # subsampling -> 100 Hz to 10Hz
    input_size = 10                 # 1 second
    output_size = 10                # 1 second
    dimensions = [1, 2, 3]          # right hand x, right hand y, right hand z
    batch_size = 128
    dataset_config = TeleopIcubDatasetConfig(input_size=input_size,
                                             output_size=output_size,
                                             dimensions=dimensions,
                                             subsampling_step=subsampling_step,
                                             batch_size=batch_size)
    dataset = TeleopIcubDataset(dataset_config)

    # -- Init predictor
    hidden_size = 100
    feature_size = dataset.feature_size
    config = LSTMConfig(feature_size=feature_size,
                        output_size=output_size,
                        hidden_size=hidden_size,)
    predictor = LSTMPredictor(config=config)

    # Train, Test and Save
    training_config = TrainingConfig(epoch=50,
                                     accelerator="gpu")
    predictor.train(dataset.train_dataloader, training_config)
    predictor.test(dataset.test_dataloader)
    predictor.save()
    # plot some test episodes
    input = dataset.test_datasample[0][0]
    truth = dataset.test_datasample[0][1]
    prediction = predictor(input)
    input = dataset.unscale(input)
    truth = dataset.unscale(truth)
    prediction = dataset.unscale(prediction)
    plot_prediction((input, truth), prediction,
                    savefig_path=Path(predictor.config.model_path) / "pred_data_1.png")
    truth = dataset.test_datasample[0][1]
    print("ADE: %.5f\nFDE: %.5f" % (get_ade(truth, prediction), get_fde(truth, prediction)))
