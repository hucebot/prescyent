from pytorch_lightning.callbacks import TQDMProgressBar


class LightningProgressBar(TQDMProgressBar):
    def on_train_epoch_start(self, trainer, *_) -> None:
        super().on_train_epoch_start(trainer, *_)
        self.main_progress_bar.set_description(f"Epoch {trainer.current_epoch}/{trainer.max_epochs}")
