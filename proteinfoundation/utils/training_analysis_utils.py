import time

import torch
from lightning.pytorch.callbacks import Callback

# from pytorch_lightning.utilities.rank_zero import rank_zero_only  # Do not use here, breaks clsuter
from loguru import logger


def log_metrics(pl_module, metrics):
    for m in metrics:
        pl_module.log(
            m, metrics[m], on_step=True, on_epoch=True, prog_bar=False, logger=True
        )


class UnusedParametersCallback(Callback):
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        for name, param in pl_module.named_parameters():
            if param.grad is None:
                print(f"Unused parameter: {name}")


class LogEpochTimeCallback(Callback):
    """Simple callback that logs how long each epoch takes, in seconds, to a pytorch lightning log"""

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        curr_time = time.time()
        duration = curr_time  # - self.epoch_start
        pl_module.log(
            "train_info/epoch_duration_secs",
            duration,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        if pl_module.current_epoch % 10 == 0:
            logger.info(
                f"Done training epoch {pl_module.current_epoch}, epoch took {duration} seconds"
            )


class LogSetpTimeCallback(Callback):
    """Simple callback that logs how long each training step takes, in seconds, to a pytorch lightning log"""

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.step_start = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        curr_time = time.time()
        duration = curr_time - self.step_start
        pl_module.log(
            "train_info/step_duration_secs",
            duration,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )


class GradAndWeightAnalysisCallback(Callback):
    """Some functionality to observe how are weights and gradientsbehaving during trainnig."""

    def __init__(self, debug=True, moving_avg_size=100):
        super(GradAndWeightAnalysisCallback, self).__init__()
        self.debug = debug
        self.moving_avg_size = moving_avg_size
        self.avg_grad_history = []
        self.max_grad_history = []

    def _get_avg_and_max_w(self, pl_module):
        """Computes average and max weight in module."""
        with torch.no_grad():
            params = torch.nn.utils.parameters_to_vector(pl_module.parameters()).abs()
            return params.sum() / params.numel(), params.max()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Before updating log max and average weight."""
        avg_w, max_w = self._get_avg_and_max_w(pl_module)
        avg_g, max_g = self._get_avg_and_max_grad(pl_module)
        metrics = {}

        # Gradient metrics
        metrics["avg_w_bef_step"] = avg_w
        metrics["max_w_bef_step"] = max_w
        metrics["avg_g_bef_step"] = avg_g
        metrics["max_g_bef_step"] = max_g

        # Add gradient metrics
        if len(self.max_grad_history) > 1:
            metrics["moving_avg_max_grad_bef_step"] = sum(self.max_grad_history) / len(
                self.max_grad_history
            )
            metrics["moving_avg_avg_grad_bef_step"] = sum(self.avg_grad_history) / len(
                self.avg_grad_history
            )
            metrics["max_g_over_avg_max_g_bef_step"] = (
                max_g / metrics["moving_avg_max_grad_bef_step"]
            )
            metrics["avg_g_over_avg_avg_g_bef_step"] = (
                avg_g / metrics["moving_avg_avg_grad_bef_step"]
            )

        # Update history of gradient statistics
        if len(self.max_grad_history) >= self.moving_avg_size:
            self.max_grad_history.pop(0)
        if not (max_g.isnan().any() or max_g.isinf().any()):
            self.max_grad_history.append(max_g.item())

        if len(self.avg_grad_history) >= self.moving_avg_size:
            self.avg_grad_history.pop(0)
        if not (avg_g.isnan().any() or avg_g.isinf().any()):
            self.avg_grad_history.append(avg_g.item())

        if self.debug and (avg_w.isnan().any() or max_w.isnan().any()):
            params = torch.nn.utils.parameters_to_vector(pl_module.parameters()).abs()

        log_metrics(pl_module, metrics)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """First thing on training step."""
        avg_w, max_w = self._get_avg_and_max_w(pl_module)

    def _get_avg_and_max_grad(self, pl_module):
        """Computes average and max grad in module, if no grad computed zero."""
        grad_sum = torch.tensor(0.0, device=pl_module.device)
        max_grad = torch.tensor(0.0, device=pl_module.device)
        count = 0
        for p in pl_module.parameters():
            if p.grad is not None:
                abs_grad = p.grad.abs()
                grad_sum += abs_grad.sum()
                max_grad = torch.max(max_grad, abs_grad.max())
                count += p.grad.numel()
        if count == 0:
            return torch.tensor(0.0), torch.tensor(0.0)
        return grad_sum / count, max_grad

    def _count_nan_grad(self, pl_module):
        numels, num_nans = 0, 0
        for p in pl_module.parameters():
            if p.grad is not None:
                numels += p.grad.numel()
                num_nans += p.grad.isnan().sum().item()
        return numels, num_nans

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        """Before zero-ing grad log max and average grad
        (should be shifted one back from next)."""
        avg_g, max_g = self._get_avg_and_max_grad(pl_module)

        metrics = {
            "avg_g_bef_zerog": avg_g,
            "max_g_bef_zerog": max_g,
        }
        log_metrics(pl_module, metrics)

    def on_after_backward(self, trainer, pl_module):
        """After computing gradient log max and average grad.
        This same value should be returned by <on_before_zxero_grad>
        in the next iteration."""
        avg_g, max_g = self._get_avg_and_max_grad(pl_module)
        numels, num_nans = self._count_nan_grad(pl_module)

        metrics = {
            "avg_g_after_bwd": avg_g,
            "max_g_after_bwd": max_g,
        }
        log_metrics(pl_module, metrics)


class SkipNanGradCallback(Callback):
    """Callback to skip gradient updates with NaN in them."""

    def __init__(self, debug=True):
        super(SkipNanGradCallback, self).__init__()
        self.count = 0
        self.iter = 0

    def on_after_backward(self, trainer, pl_module):
        nan_flag = False
        self.iter += 1
        for p in pl_module.parameters():
            if p.grad is not None:
                # has_grad = True
                if p.grad.isnan().any():
                    nan_flag = True
        if nan_flag:
            self.count += 1
            pl_module.zero_grad()


class SkipLargeGradients(Callback):
    """
    This callback tracks the average max gradient for the last `moving_avg_size` steps and checks if the current max gradient
    """

    def __init__(
        self,
        moving_avg_size: int = 100,
        factor_threshold: int = 5,
        min_opt_steps: int = 2000,
    ):
        self.max_g_history = []
        self.moving_avg_size = moving_avg_size
        self.factor_threshold = factor_threshold
        self.min_opt_steps = min_opt_steps

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # calculate the maximum gradient
        max_g = max(
            param.grad.data.abs().max()
            for param in pl_module.parameters()
            if param.grad is not None
        )

        # Update the moving average history
        if len(self.max_g_history) >= self.moving_avg_size:
            self.max_g_history.pop(0)
        if not (max_g.isnan().any() or max_g.isinf().any()):
            self.max_g_history.append(max_g.item())

        if trainer.global_step > self.min_opt_steps:
            if len(self.max_g_history) > 1:
                moving_average = sum(self.max_g_history) / len(self.max_g_history)

                # Check if the current max gradient exceeds factor_threshold * moving average
                if max_g > self.factor_threshold * moving_average:
                    # Zero out gradients
                    for param in pl_module.parameters():
                        if param.grad is not None:
                            param.grad.data.zero_()  # Set gradients to zero
