from prescyent.predictor import LSTMPredictor, LSTMConfig, TrainingConfig
from prescyent.dataset import TeleopIcubDataset


if __name__ == "__main__":
    # -- Init dataset and predictor
    dataset = TeleopIcubDataset()
    config = LSTMConfig(input_size=1,
                        output_size=10)
    predictor = LSTMPredictor(config=config)

    # Train, Test and Save
    training_config = TrainingConfig(epoch=2)
    predictor.train(dataset.train_dataloader, training_config)
    predictor.test(dataset.test_dataloader)
    save_path = "data/example/lstm_ver1"
    predictor.save(save_path)
