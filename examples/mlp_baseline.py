from prescyent.evaluator import eval_predictors
from prescyent.predictor import MlpPredictor, MlpConfig, TrainingConfig
from prescyent.dataset import TeleopIcubDataset, TeleopIcubDatasetConfig


if __name__ == "__main__":
    # -- Init dataset
    subsampling_step: int = 10      # subsampling -> 100 Hz to 10Hz
    history_size = 10                 # 1 second
    future_size = 10                # 1 second
    dimensions = None               # None equals ALL dimensions !
    # for TeleopIcub dimension = [0, 1, 2] is waist, right_hand, left_hand
    batch_size = 64
    dataset_config = TeleopIcubDatasetConfig(history_size=history_size,
                                             future_size=future_size,
                                             dimensions=dimensions,
                                             subsampling_step=subsampling_step,
                                             batch_size=batch_size,
                                            )
    dataset = TeleopIcubDataset(dataset_config)

    # -- Init predictor
    feature_size = dataset.feature_size
    config = MlpConfig(output_size=future_size,
                          input_size=history_size,
                          norm_on_last_input=True)
    predictor = MlpPredictor(config=config)

    # Train, Test and Save
    training_config = TrainingConfig(epoch=100, use_scheduler=True)
    predictor.train(dataset.train_dataloader, training_config,
                    dataset.val_dataloader)
    predictor.test(dataset.test_dataloader)
    predictor.save("data/models/teleopicub/all/"
                   f"{predictor.name}/version_{predictor.version}")
    # plot some test trajectories
    eval_results = eval_predictors([predictor],
                                   dataset.trajectories.test[0:1],
                                   history_size=history_size,
                                   future_size=future_size,)[0]
    print("ADE:", eval_results.mean_ade, "FDE:", eval_results.mean_fde)
