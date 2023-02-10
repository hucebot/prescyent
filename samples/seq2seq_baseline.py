from prescyent.evaluator import eval_episode
from prescyent.predictor import Seq2SeqPredictor, Seq2SeqConfig, TrainingConfig
from prescyent.dataset import TeleopIcubDataset, TeleopIcubDatasetConfig


if __name__ == "__main__":
    # -- Init dataset
    subsampling_step: int = 10      # subsampling -> 100 Hz to 10Hz
    input_size = 10                 # 1 second
    output_size = 10                # 1 second
    dimensions = None               # None equals ALL dimensions !
    # for TeleopIcub dimension = [1, 2, 3] is right hand x, right hand y, right hand z
    batch_size = 64 * 4
    persistent_workers = True
    devices = 2
    dataset_config = TeleopIcubDatasetConfig(input_size=input_size,
                                             output_size=output_size,
                                             dimensions=dimensions,
                                             subsampling_step=subsampling_step,
                                             batch_size=batch_size,
                                             num_workers=devices * 4,
                                             persistent_workers=persistent_workers)
    dataset = TeleopIcubDataset(dataset_config)

    # -- Init predictor

    feature_size = dataset.feature_size
    hidden_size = feature_size * 20
    config = Seq2SeqConfig(feature_size=feature_size,
                           output_size=output_size,
                           hidden_size=hidden_size,
                           input_size=input_size)
    predictor = Seq2SeqPredictor(config=config)

    # Train, Test and Save
    training_config = TrainingConfig(epoch=1000,
                                     accelerator="gpu", devices=devices,
                                     use_scheduler=True)
    predictor.train(dataset.train_dataloader, training_config, dataset.val_dataloader)
    predictor.test(dataset.test_dataloader)
    predictor.save()
    # plot some test episodes
    episode = dataset.episodes.test[0]
    ade, fde = eval_episode(episode, predictor, input_size=input_size,
                            savefig_path=f"data/eval/seq2seq_test_episode.png",
                            eval_on_last_pred=False, unscale_function=dataset.unscale)
    print("ADE:", ade.item(), "FDE:", fde.item())
