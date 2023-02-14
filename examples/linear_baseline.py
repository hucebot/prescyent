from prescyent.evaluator import eval_trajectory
from prescyent.predictor import LinearPredictor, LinearConfig, TrainingConfig
from prescyent.dataset import TeleopIcubDataset, TeleopIcubDatasetConfig


if __name__ == "__main__":
    # -- Init dataset
    subsampling_step: int = 10      # subsampling -> 100 Hz to 10Hz
    input_size = 10                 # 1 second
    output_size = 10                # 1 second
    dimensions = None               # None equals ALL dimensions !
    # for TeleopIcub dimension = [1, 2, 3] is right hand x, right hand y, right hand z
    batch_size = 64
    persistent_workers = True
    dataset_config = TeleopIcubDatasetConfig(input_size=input_size,
                                             output_size=output_size,
                                             dimensions=dimensions,
                                             subsampling_step=subsampling_step,
                                             batch_size=batch_size)
    dataset = TeleopIcubDataset(dataset_config)

    # -- Init predictor

    feature_size = dataset.feature_size
    config = LinearConfig(feature_size=feature_size,
                          output_size=output_size,
                          input_size=input_size)
    predictor = LinearPredictor(config=config)

    # Train, Test and Save
    training_config = TrainingConfig()
    predictor.train(dataset.train_dataloader, training_config, dataset.val_dataloader)
    predictor.test(dataset.test_dataloader)
    predictor.save()
    # plot some test trajectories
    trajectory = dataset.trajectories.test[0]
    ade, fde = eval_trajectory(trajectory, predictor, input_size=input_size,
                            savefig_path=f"data/eval/linear_test_trajectory.png",
                            eval_on_last_pred=False, unscale_function=dataset.unscale)
    print("ADE:", ade.item(), "FDE:", fde.item())
