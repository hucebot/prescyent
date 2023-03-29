from prescyent.evaluator import eval_predictors
from prescyent.predictor import SARLSTMPredictor, SARLSTMConfig, TrainingConfig
from prescyent.dataset import SineDataset, SineDatasetConfig, LearningTypes


if __name__ == "__main__":
    # -- Init dataset
    history_size = 999
    future_size = 999
    dimensions = None               # None equals ALL dimensions !
    batch_size = 128
    dataset_config = SineDatasetConfig(history_size=history_size,
                                       future_size=future_size,
                                       dimensions=dimensions,
                                       size=1000,
                                       batch_size=batch_size,
                                       learning_type=LearningTypes.AUTOREG)
    dataset = SineDataset(dataset_config)

    # -- Init predictor
    feature_size = dataset.feature_size
    hidden_size = feature_size * 20
    config = SARLSTMConfig(feature_size=feature_size,
                           hidden_size=hidden_size,)
    predictor = SARLSTMPredictor(config=config)

    # Train, Test and Save
    training_config = TrainingConfig()
    predictor.train(dataset.train_dataloader, training_config, dataset.val_dataloader)
    predictor.test(dataset.test_dataloader)
    predictor.save(f"data/models/sine/{predictor.name}/version_{predictor.version}")
    # plot some test trajectories
    eval_results = eval_predictors([predictor],
                                   dataset.trajectories.test[0:1],
                                   history_size=history_size,
                                   future_size=future_size,)[0]
    print("ADE:", eval_results.mean_ade, "FDE:", eval_results.mean_fde)
