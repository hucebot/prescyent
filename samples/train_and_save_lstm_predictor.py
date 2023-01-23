from prescyent.dataset.motion.teleop_icub.config import TeleopIcubDatasetConfig
from prescyent.predictor import LSTMPredictor, LSTMConfig, TrainingConfig
from prescyent.dataset import TeleopIcubDataset


if __name__ == "__main__":
    # -- Init dataset
    subsampling_step: int = 10     # subsampling -> 100 Hz to 10Hz
    input_size = 10  # 1 second
    output_size = 10  # 1 second
    dataset_config = TeleopIcubDatasetConfig(input_size=input_size,
                                             output_size=output_size,
                                             # right hand x, right hand y, right hand z
                                             dimensions=[1, 2, 3],
                                             subsampling_step=subsampling_step)
    dataset = TeleopIcubDataset(dataset_config)

    # -- Init predictor
    model_path = "data/models"  # this is default value also
    # model_path is where the trained model and training infos will be stored by default
    hidden_size = 32
    feature_size = dataset.feature_size
    config = LSTMConfig(feature_size=feature_size,
                        output_size=output_size,
                        hidden_size=hidden_size,
                        model_path=model_path)
    predictor = LSTMPredictor(config=config)

    # Train, Test and Save
    training_config = TrainingConfig(epoch=10)
    predictor.train(dataset.train_dataloader, training_config)
    predictor.test(dataset.test_dataloader)
    # the save method here allow to export all infos of the training and model to a given dir path
    save_path = "data/example/lstm_ver1"
    predictor.save(save_path)
