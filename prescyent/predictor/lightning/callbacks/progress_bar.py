from pytorch_lightning.callbacks import TQDMProgressBar


def calculate_max_epoch(trainer):
    return trainer.max_steps // trainer.num_training_batches + 1


class LightningProgressBar(TQDMProgressBar):
    """surcharge the base progress bar with total number of epoch infos"""

    def on_train_epoch_start(self, trainer, *_) -> None:
        super().on_train_epoch_start(trainer, *_)
        max_epoch = trainer.max_epochs
        if max_epoch < 0:
            max_epoch = calculate_max_epoch(trainer)
        self.train_progress_bar.set_description(
            f"Epoch {trainer.current_epoch + 1}/{max_epoch}"
        )
