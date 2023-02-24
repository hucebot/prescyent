from prescyent.dataset.config import LearningTypes
from prescyent.evaluator import eval_predictors
from prescyent.predictor import SARLSTMPredictor, SARLSTMConfig, TrainingConfig
from prescyent.dataset import TeleopIcubDataset, TeleopIcubDatasetConfig


if __name__ == "__main__":
    # -- Init dataset
    subsampling_step: int = 10      # subsampling -> 100 Hz to 10Hz
    history_size = 10                 # 1 second
    future_size = 10                # 1 second
    dimensions = None               # None equals ALL dimensions !
    # for TeleopIcub dimension = [1, 2, 3] is right hand x, right hand y, right hand z
    batch_size = 64
    dataset_config = TeleopIcubDatasetConfig(history_size=history_size,
                                             future_size=future_size,
                                             dimensions=dimensions,
                                             subsampling_step=subsampling_step,
                                             batch_size=batch_size,
                                             learning_type=LearningTypes.AUTOREG)
    dataset = TeleopIcubDataset(dataset_config)

    # -- Init predictor
    feature_size = dataset.feature_size
    hidden_size = feature_size * 20
    config = SARLSTMConfig(feature_size=feature_size,
                           output_size=future_size,
                           hidden_size=hidden_size,)
    predictor = SARLSTMPredictor(config=config)

    # Train, Test and Save
    training_config = TrainingConfig()
    predictor.train(dataset.train_dataloader, training_config, dataset.val_dataloader)
    predictor.test(dataset.test_dataloader)
    predictor.save("data/models/teleopicub/all/"
                   f"{predictor.name}/version_{predictor.version}")
    # plot some test trajectories
    eval_results = eval_predictors([predictor],
                                   dataset.trajectories.test[0:1],
                                   history_size=history_size,
                                   future_size=future_size,
                                   unscale_function=dataset.unscale)[0]
    print("ADE:", eval_results.mean_ade, "FDE:", eval_results.mean_fde)
