from prescyent.evaluator import eval_episode, eval_episode_multiple_predictors
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
    num_workers = 8
    persistent_workers = True
    dataset_config = TeleopIcubDatasetConfig(input_size=input_size,
                                             output_size=output_size,
                                             dimensions=dimensions,
                                             subsampling_step=subsampling_step,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             persistent_workers=persistent_workers)
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
        predictor.test(dataset.test_dataloader)
        # plot some test episodes
        for i, episode in enumerate(dataset.episodes.val):
            ade, fde = eval_episode(episode, predictor, input_size=input_size,
                                    savefig_path=f"data/eval/{i}_{predictor}"
                                    "_test_episode.png",
                                    eval_on_last_pred=False, unscale_function=dataset.unscale)
            print(f"{predictor}, :\nADE: {ade.item() :.5f}, FDE: {fde.item() :.5f}")
    for i, episode in enumerate(dataset.episodes.val):
        eval_episode_multiple_predictors(episode, predictors, input_size=input_size,
                                 savefig_path=f"data/eval/{i}_test_episode.png",
                                 unscale_function=dataset.unscale)
