# this example shows how to learn a MLP on the TeleopIcubDataset
import copy
from prescyent.predictor import (
    MlpPredictor,
    MlpConfig,
    TrainingConfig,
    DelayedPredictor,
)
from prescyent.dataset import TeleopIcubDataset, TeleopIcubDatasetConfig
from prescyent.dataset.features import CoordinateXYZ
from prescyent.utils.enums import Normalizations, LossFunctions

if __name__ == "__main__":
    # -- Init dataset
    print("Initializing dataset...", end=" ")
    subsampling_step: int = 10  # subsampling -> 100 Hz to 10Hz
    history_size = 10  # 1 second
    future_size = 10  # 1 second
    dimensions = None  # None equals ALL dimensions !
    # for TeleopIcub dimension = [0, 1, 2] is [waist, right_hand, left_hand]
    features = CoordinateXYZ(range(3))
    batch_size = 256
    dataset_config = TeleopIcubDatasetConfig(
        history_size=history_size,
        future_size=future_size,
        subsampling_step=subsampling_step,
        batch_size=batch_size,
        in_features=features,
        out_features=features,
    )
    dataset = TeleopIcubDataset(dataset_config)
    print("OK")

    # -- Init predictor
    print("Initializing predictor...", end=" ")
    config = MlpConfig(
        dataset_config=dataset_config,
        hidden_size=256,
        num_layers=4,
        norm_on_last_input=True,
        used_norm=Normalizations.BATCH,
        loss_fn=LossFunctions.MTRDLOSS,
    )
    sample, truth = dataset.val_datasample[0]
    predictor = MlpPredictor(config=config)
    print("OK")

    # Train
    training_config = TrainingConfig(
        epoch=100,
        devices="auto",
        accelerator="cpu",
        lr=0.0001,
        early_stopping_patience=10,
    )
    predictor.train(dataset.train_dataloader, training_config, dataset.val_dataloader)

    # Save the predictor
    xp_dir = (
        f"data/models/{dataset.DATASET_NAME}"
        f"/h{history_size}_f{future_size}"
        f"_{dataset.frequency}hz"
    )
    model_dir = f"{xp_dir}/{predictor.name}/version_{predictor.version}"
    print("Model directory:", model_dir)
    predictor.save(model_dir, rm_log_path=False)
    # We save also the config so that we can load it later if needed
    dataset.save_config(model_dir + "/dataset.config")

    # Test so that we know how good we are
    predictor.test(dataset.test_dataloader)
    # Compare with delayed baseline
    delayed = DelayedPredictor(f"{xp_dir}")
    delayed.test(dataset.test_dataloader)
