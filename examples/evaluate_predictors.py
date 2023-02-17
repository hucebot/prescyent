from prescyent.evaluator import eval_predictors
from prescyent.predictor import SARLSTMPredictor, LinearPredictor, DelayedPredictor, ConstantPredictor, Seq2SeqPredictor
from prescyent.dataset import TeleopIcubDataset, TeleopIcubDatasetConfig


if __name__ == "__main__":
    # -- Init dataset
    subsampling_step: int = 10      # subsampling -> 100 Hz to 10Hz
    history_size = 10                 # 1 second
    future_size = 10                # 1 second
    dimensions = None               # None equals ALL dimensions !
    # for TeleopIcub dimension = [1, 2, 3] is right hand x, right hand y, right hand z
    batch_size = 64 * 4
    dataset_config = TeleopIcubDatasetConfig(history_size=history_size,
                                             future_size=future_size,
                                             dimensions=dimensions,
                                             subsampling_step=subsampling_step,
                                             batch_size=batch_size)
    dataset = TeleopIcubDataset(dataset_config)

    # -- Load predictors
    linear_predictor = LinearPredictor("data/models/teleopredictoricub/all/"
                                       "LinearPredictor/version_0")
    lstm_predictor = SARLSTMPredictor("data/models/teleopredictoricub/all/"
                                      "SARLSTMPredictor/version_2")
    seq2seq_predictor = Seq2SeqPredictor("data/models/teleopredictoricub/all/"
                                         "Seq2SeqPredictor/version_0")
    delayed_predictor = DelayedPredictor("data/models/teleopredictoricub/all")
    constant_predictor = ConstantPredictor("data/models/teleopredictoricub/all")
    # Train, Test and Save

    predictors = [
            linear_predictor,
            lstm_predictor,
            seq2seq_predictor,
            delayed_predictor,
            constant_predictor
        ]

    eval_results = eval_predictors(predictors,
                                   dataset.trajectories.test[0:5],
                                   history_size=history_size,
                                   future_size=future_size,
                                   unscale_function=dataset.unscale,
                                   saveplot_dir_path="data/eval/teleop/ar")

    for p, predictor in enumerate(predictors):
        print("\n ---", predictor,
              "\nMean ADE: %.6f" % eval_results[p].mean_ade,
              "\nMean FDE: %.6f" % eval_results[p].mean_fde,
              "\nMean Inference Time (ms): %.6f" % eval_results[p].mean_inference_time_ms,
              "\nMax ADE: %.6f" % eval_results[p].max_ade,
              "\nMax FDE: %.6f" % eval_results[p].max_fde,
              "\nMax Inference Time (ms): %.6f" % eval_results[p].max_inference_time_ms,)
