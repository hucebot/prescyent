from prescyent.evaluator import eval_trajectory, eval_trajectory_multiple_predictors
from prescyent.predictor import LinearPredictor, LSTMPredictor, Seq2SeqPredictor, DelayedPredictor
from prescyent.dataset import TeleopIcubDataset, TeleopIcubDatasetConfig


if __name__ == "__main__":
    # -- Init dataset
    subsampling_step: int = 10      # subsampling -> 100 Hz to 10Hz
    input_size = 10                 # 1 second
    output_size = 10                # 1 second
    dimensions = None               # None equals ALL dimensions !
    # for TeleopIcub dimension = [1, 2, 3] is right hand x, right hand y, right hand z
    batch_size = 64
    dataset_config = TeleopIcubDatasetConfig(input_size=input_size,
                                             output_size=output_size,
                                             dimensions=dimensions,
                                             subsampling_step=subsampling_step,
                                             batch_size=batch_size)
    dataset = TeleopIcubDataset(dataset_config)

    # -- Load predictors
    linear_predictor = LinearPredictor("data/models/LinearPredictor/version_0")
    lstm_predictor = LSTMPredictor("data/models/LSTMPredictor/version_0")
    seq2seq_predictor = Seq2SeqPredictor("data/models/Seq2SeqPredictor/version_0")
    delayed_predictor = DelayedPredictor("data/models")
    # Train, Test and Save

    predictors = [
            linear_predictor,
            lstm_predictor,
            seq2seq_predictor,
            delayed_predictor
        ]

    for predictor in predictors:
        # redo the test loop and log in tensorboard
        predictor.test(dataset.test_dataloader)

        # check model behavior with some plots on train trajectories
        for i, trajectory in enumerate(dataset.trajectories.train[:5]):
            ade, fde = eval_trajectory(trajectory, predictor, input_size=input_size,
                                    savefig_path=f"data/eval/train/trajectory_{i}/{predictor}"
                                    "_trajectory_evaluation.png",
                                    eval_on_last_pred=False, unscale_function=dataset.unscale)
            print(f"{predictor}, :\nADE: {ade.item() :.5f}, FDE: {fde.item() :.5f}")
        # check model generalization with some plots on test trajectories
        for i, trajectory in enumerate(dataset.trajectories.test[:5]):
            ade, fde = eval_trajectory(trajectory, predictor, input_size=input_size,
                                    savefig_path=f"data/eval/test/trajectory_{i}/{predictor}"
                                    "_trajectory_evaluation.png",
                                    eval_on_last_pred=False, unscale_function=dataset.unscale)
            print(f"{predictor}, :\nADE: {ade.item() :.5f}, FDE: {fde.item() :.5f}")

    # compare models with some plots on train and test trajectories
    for i, trajectory in enumerate(dataset.trajectories.train[:5]):
        eval_trajectory_multiple_predictors(trajectory, predictors, input_size=input_size,
                                 savefig_path=f"data/eval/train/trajectory_{i}/multi_predictors_evaluation.png",
                                 unscale_function=dataset.unscale)
    for i, trajectory in enumerate(dataset.trajectories.test[:5]):
        eval_trajectory_multiple_predictors(trajectory, predictors, input_size=input_size,
                                 savefig_path=f"data/eval/test/trajectory_{i}/multi_predictors_evaluation.png",
                                 unscale_function=dataset.unscale)
