from prescyent.evaluator import eval_predictors
from prescyent.predictor import LinearPredictor, LinearConfig, TrainingConfig
from prescyent.dataset import TeleopIcubDataset, TeleopIcubDatasetConfig


if __name__ == "__main__":
    # -- Init dataset
    subsampling_step: int = 10      # subsampling -> 100 Hz to 10Hz
    history_size = 10                 # 1 second
    future_size = 10                # 1 second
    dimensions = None               # None equals ALL dimensions !
    # for TeleopIcub dimension = [1, 2, 3] is right hand x, right hand y, right hand z
    batch_size = 64
    persistent_workers = True
    dataset_config = TeleopIcubDatasetConfig(history_size=history_size,
                                             output_windows_size=future_size,
                                             dimensions=dimensions,
                                             subsampling_step=subsampling_step,
                                             batch_size=batch_size)
    dataset = TeleopIcubDataset(dataset_config)

    # -- Init predictor

    feature_size = dataset.feature_size
    config = LinearConfig(feature_size=feature_size,
                          output_size=future_size,
                          input_size=history_size)
    predictor = LinearPredictor(config=config)

    # Train, Test and Save
    training_config = TrainingConfig()
    predictor.train(dataset.train_dataloader, training_config, dataset.val_dataloader)
    predictor.test(dataset.test_dataloader)
    predictor.save(f"data/models/teleopredictoricub/all/{predictor.name}/version_{predictor.version}")
    # plot some test trajectories
    eval_results = eval_predictors([predictor],
                               dataset.trajectories.test[0:1],
                               history_size=history_size,
                               future_size=future_size,
                               unscale_function=dataset.unscale)[0]
    print("ADE:", eval_results.mean_ade, "FDE:", eval_results.mean_fde)