import lightning as pl
import torch
import sys

if sys.argv[1] == "16":
    torch.set_default_dtype(torch.float16)
if sys.argv[1] == "b16":
    torch.set_default_dtype(torch.bfloat16)
if sys.argv[1] == "32":
    torch.set_default_dtype(torch.float32)
if sys.argv[1] == "64":
    torch.set_default_dtype(torch.float64)


class MyEarlyStopping(pl.pytorch.callbacks.EarlyStopping):

    def __init__(self, feature_importance, **kwargs):
        super(MyEarlyStopping, self).__init__(**kwargs)
        self.feature_importance = feature_importance

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer, pl_module)

    def _run_early_stopping_check(self, trainer, pl_module):
        nfeats = pl_module.e2efs_layer.get_n_alive().item()
        alpha = pl_module.e2efs_layer.moving_factor
        if nfeats < self.stopping_threshold:
            trainer.should_stop = True
        print('\tnfeats {} (threshold {}) alpha {:.4f} (threshold {})'.format(nfeats, self.stopping_threshold, alpha, self.feature_importance))
