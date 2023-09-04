# this example shows how to learn a MLP on the TeleopIcubDataset
from prescyent.predictor import MlpPredictor, MlpConfig, TrainingConfig
from prescyent.dataset import TeleopIcubDataset, TeleopIcubDatasetConfig


if __name__ == "__main__":
    # -- Init dataset
    print("Initializing dataset...", end="")
    subsampling_step: int = 10  # subsampling -> 100 Hz to 10Hz
    history_size = 10  # 1 second
    future_size = 10  # 1 second
    dimensions = None  # None equals ALL dimensions !
    # for TeleopIcub dimension = [0, 1, 2] is waist, right_hand, left_hand
    batch_size = 256
    dataset_config = TeleopIcubDatasetConfig(
        history_size=history_size,
        future_size=future_size,
        dimensions=dimensions,
        subsampling_step=subsampling_step,
        batch_size=batch_size,
    )
    dataset = TeleopIcubDataset(dataset_config)
    print("OK")

    # -- Init predictor
    print("Initializing predictor...", end="")
    feature_size = dataset.feature_size
    config = MlpConfig(
        output_size=future_size,
        input_size=history_size,
        hidden_size=512,
        norm_on_last_input=True,
    )
    predictor = MlpPredictor(config=config)
    print("OK")

    # Train
    training_config = TrainingConfig(epoch=100, devices=1, use_auto_lr=True)
    predictor.train(dataset.train_dataloader, training_config, dataset.val_dataloader)

    # Save the predictor
    model_dir = (
        f"data/models/teleopicub/all/{predictor.name}/version_{predictor.version}"
    )
    print("model directory:", model_dir)
    predictor.save(model_dir)
    # We save also the config so that we can load it later if needed
    dataset.save_config(model_dir + "/dataset.config")

    # Test so that we know how good we are
    predictor.test(dataset.test_dataloader)
