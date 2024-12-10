# this example shows how to learn a MLP on the TeleopIcubDataset
from pathlib import Path

from prescyent.dataset import TeleopIcubDataset, TeleopIcubDatasetConfig
from prescyent.dataset.features import CoordinateXYZ, Features
from prescyent.predictor import (
    MlpPredictor,
    MlpConfig,
    TrainingConfig,
    DelayedPredictor,
    PredictorConfig,
)
from prescyent.scaler import ScalerConfig
from prescyent.utils.enums import LossFunctions, Scalers, TrajectoryDimensions
from prescyent.utils.enums.learning_types import LearningTypes


if __name__ == "__main__":
    # -- Init dataset
    print("Initializing dataset...", end=" ")
    frequency: int = 24  # target frequency
    history_size = 24  # 24 frames at 24Hz => 1 seconds as history
    future_size = 12  # 12 frames at 24Hz => 0.5 seconds as future
    features = Features([CoordinateXYZ(range(3))])
    points_ids = [1, 2]  # ids of the left and right hands
    batch_size = 256
    dataset_config = TeleopIcubDatasetConfig(
        hdf5_path="data/datasets/AndyData-lab-prescientTeleopICub.hdf5",
        context_keys=[
            "center_of_mass"
        ],  # Using center of mass of the robot as additional context input
        subsets=[
            "BottleTable"
        ],  # training and evaluating only on this subset task of the dataset
        learning_type=LearningTypes.SEQ2SEQ,
        history_size=history_size,
        future_size=future_size,
        frequency=frequency,  # subsample -> 100 Hz to 24Hz
        batch_size=batch_size,
        in_features=features,  # we have x,y,z coordinates for each points as input (this is also the default value)
        out_features=features,  # we have x,y,z coordinates for each points as output (this is also the default value)
        out_points=points_ids,  # Predict only hands. Here we don't override defaults for 'in_points' so it will be all points, including waist
    )
    dataset = TeleopIcubDataset(dataset_config)
    print("OK")

    # -- Init scaler
    scaler_config = ScalerConfig(
        do_feature_wise_scaling=True,
        scaler=Scalers.STANDARDIZATION,
        scaling_axis=TrajectoryDimensions.TEMPORAL,
    )
    # -- Init predictor
    print("Initializing predictor...", end=" ")
    config = MlpConfig(
        dataset_config=dataset_config,
        context_size=dataset.context_size_sum,
        scaler_config=scaler_config,
        hidden_size=128,
        num_layers=4,
        deriv_on_last_frame=True,
        loss_fn=LossFunctions.MTRDLOSS,
    )
    predictor = MlpPredictor(config=config)
    print("OK")

    # Train
    training_config = TrainingConfig(
        max_epochs=200,  # Maximum number of training epochs
        devices="auto",  # Chose the best available devices (see lightning documentation for more)
        accelerator="auto",  # Chose the best available accelerator (see lightning documentation for more)
        lr=0.0001,  # The learning rate
        early_stopping_patience=10,  # We'll stop the training before max_epochs if the validation loss doesn't improve for 10 epochs
    )

    # Scaler is also trained by the predictor's method !
    predictor.train(dataset, training_config)

    # Save the predictor
    xp_dir = (
        Path(__file__).parent.resolve()
        / "data"
        / "models"
        / f"{dataset.DATASET_NAME}"
        / f"h{history_size}_f{future_size}_{dataset.frequency}hz"
    )
    model_dir = xp_dir / f"{predictor.name}" / f"version_{predictor.version}"
    print("Model directory:", model_dir)
    predictor.save(model_dir, rm_log_path=False)
    # We can save also the dataset config so that we can load it later if needed
    dataset.save_config(model_dir / "dataset_config.json")

    # Test predictor over the test set so that we know how good we are
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
