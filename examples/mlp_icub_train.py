# this example shows how to learn a MLP on the TeleopIcubDataset
from prescyent.dataset import TeleopIcubDataset, TeleopIcubDatasetConfig
from prescyent.dataset.features import CoordinateXYZ
from prescyent.predictor import (
    MlpPredictor,
    MlpConfig,
    TrainingConfig,
    DelayedPredictor,
    PredictorConfig,
)
from prescyent.scaler import ScalerConfig
from prescyent.utils.enums import LossFunctions, Scalers, TrajectoryDimensions


if __name__ == "__main__":
    # -- Init dataset
    print("Initializing dataset...", end=" ")
    frequency: int = 10  # subsampling -> 100 Hz to 10Hz
    history_size = 10  # 1 seconds
    future_size = 10  # 1 seconds
    features = [CoordinateXYZ(range(3))]
    batch_size = 256
    dataset_config = TeleopIcubDatasetConfig(
        history_size=history_size,
        future_size=future_size,
        frequency=frequency,
        batch_size=batch_size,
        in_features=features,
        out_features=features,
    )
    dataset = TeleopIcubDataset(dataset_config)
    print("OK")

    # -- Init scaler
    scaler_config = ScalerConfig(
        do_feature_wise_scaling=False,
        scaler=Scalers.NORMALIZATION,
        scaling_axis=TrajectoryDimensions.TEMPORAL,
    )
    # -- Init predictor
    print("Initializing predictor...", end=" ")
    config = MlpConfig(
        dataset_config=dataset_config,
        scaler_config=scaler_config,
        hidden_size=128,
        num_layers=4,
        norm_on_last_input=True,
        loss_fn=LossFunctions.MTRDLOSS,
    )
    predictor = MlpPredictor(config=config)
    print("OK")

    # Train
    training_config = TrainingConfig(
        max_epochs=200,
        devices="auto",
        accelerator="auto",
        lr=0.0001,
        early_stopping_patience=10,
    )
    predictor.train(dataset, training_config)

    # Save the predictor
    xp_dir = (
        f"examples/data/models/{dataset.DATASET_NAME}"
        f"/h{history_size}_f{future_size}"
        f"_{dataset.frequency}hz"
    )
    model_dir = f"{xp_dir}/{predictor.name}/version_{predictor.version}"
    print("Model directory:", model_dir)
    predictor.save(model_dir, rm_log_path=False)
    # We save also the dataset config so that we can load it later if needed
    dataset.save_config(model_dir + "/dataset_config.json")

    # Test so that we know how good we are
    predictor.test(dataset)
    # Compare with delayed baseline
    delayed_config = PredictorConfig(
        dataset_config=dataset_config, save_path=f"{xp_dir}"
    )
    delayed = DelayedPredictor(config=delayed_config)
    delayed.test(dataset)

    print(
        "You can visualize all logs from this script at xp_dir using tensorboard like this:"
    )
    print(f"tensorboard --logdir {xp_dir}")
