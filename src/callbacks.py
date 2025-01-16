import lightning as pl
import math


class MyEarlyStopping(pl.pytorch.callbacks.EarlyStopping):

    def __init__(self, feature_importance, wait, **kwargs):
        super(MyEarlyStopping, self).__init__(**kwargs)
        self.feature_importance = feature_importance
        self.nfeats = 0
        self.wait = wait
        self.wait_counter = 0

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer, pl_module)

    def _run_early_stopping_check(self, trainer, pl_module):
        nfeats = pl_module.e2efs_layer.get_n_alive().item()
        if (self.nfeats == nfeats):
            self.wait_counter += 1
        else:
            self.wait_counter = 0
        self.nfeats = nfeats
        alpha = pl_module.e2efs_layer.get_factor().item()
        if math.isclose(alpha, self.feature_importance, abs_tol=1e-3) and (nfeats < self.stopping_threshold or self.wait_counter >= self.wait):
            trainer.should_stop = True
        print('\tnfeats {} (threshold {}) alpha {:.4f} (threshold {})'.format(nfeats, self.stopping_threshold, alpha, self.feature_importance))
