from prescyent.evaluator import eval_episode
from prescyent.predictor import LSTMPredictor, LSTMConfig, TrainingConfig
from prescyent.dataset import SineDataset, SineDatasetConfig


if __name__ == "__main__":
    # -- Init dataset
    input_size = 10                 # 1 second
    output_size = 10                # 1 second
    dimensions = None               # None equals ALL dimensions !
    # for TeleopIcub dimension = [1, 2, 3] is right hand x, right hand y, right hand z
    batch_size = 128
    dataset_config = SineDatasetConfig(input_size=input_size,
                                     output_size=output_size,
                                     dimensions=dimensions,
                                     size=100,
                                     batch_size=batch_size,)
    dataset = SineDataset(dataset_config)

    # -- Init predictor
    feature_size = dataset.feature_size
    hidden_size = feature_size * 20
    config = LSTMConfig(feature_size=feature_size,
                         output_size=output_size,
                        hidden_size=hidden_size,)
    predictor = LSTMPredictor(config=config)

    # Train, Test and Save
    training_config = TrainingConfig(epoch=10)
    predictor.train(dataset.train_dataloader, training_config, dataset.val_dataloader)
    predictor.test(dataset.test_dataloader)
    predictor.save()
    # plot some test episodes
    episode = dataset.episodes.test[0]
    ade, fde = eval_episode(episode, predictor, input_size=input_size,
                            savefig_path=f"data/eval/lstm_test_episode.png",
                            eval_on_last_pred=True, unscale_function=dataset.unscale)
    print("ADE:", ade.item(), "FDE:", fde.item())
