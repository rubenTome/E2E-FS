import lightning as pl
import torch
from .precision import FP_PRECISION

torch.set_default_dtype(FP_PRECISION)


class MyEarlyStopping(pl.pytorch.callbacks.EarlyStopping):

    def __init__(self, feature_importance, wait, **kwargs):
        super(MyEarlyStopping, self).__init__(**kwargs)
        self.feature_importance = feature_importance
        self.wait = wait
        self.counter = 0

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer, pl_module)

    def _run_early_stopping_check(self, trainer, pl_module):
        nfeats = pl_module.e2efs_layer.get_n_alive().item()
        alpha = pl_module.e2efs_layer.moving_factor
        if alpha >= self.feature_importance:
            self.counter += 1
        if nfeats < self.stopping_threshold or self.counter >= self.wait:
            trainer.should_stop = True
        print('\tnfeats {} (threshold {}) alpha {:.4f} (threshold {})'.format(nfeats, self.stopping_threshold, alpha, self.feature_importance))
