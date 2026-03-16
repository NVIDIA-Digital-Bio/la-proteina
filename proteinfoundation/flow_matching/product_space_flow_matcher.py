import os
from typing import Callable, Dict, Optional, Tuple, Union

import lightning as L
import torch
from jaxtyping import Bool, Float
from loguru import logger
from torch import Tensor

from proteinfoundation.flow_matching.rdn_flow_matcher import RDNFlowMatcher

FLOW_MATCHER_FACTORY = {
    "bb_ca": RDNFlowMatcher,
    "local_latents": RDNFlowMatcher,
}  # maps data modality to correspodning class


class ProductSpaceFlowMatcher(L.LightningModule):
    """
    Base class for flow matcher. Most of the methods in this class are abstract methods
    that should be implemented by classes that inheret from this one.
    """

    def __init__(self, cfg_exp: Dict):
        super().__init__()
        self.cfg_exp = cfg_exp
        self.data_modes = [m for m in self.cfg_exp.product_flowmatcher]
        self.base_flow_matchers = self.get_base_flow_matchers()

    def get_base_flow_matchers(self):
        """Constructs all necessary flow matchers."""
        return {
            m: FLOW_MATCHER_FACTORY[m](**self.cfg_exp.product_flowmatcher[m])
            for m in self.data_modes
        }

    def _apply_mask(
        self, x: Dict[str, Tensor], mask: Optional[Bool[Tensor, "* n"]] = None
    ):
        """
        Masks x.

        Args:
            x: sample to mask.
            mask (optional): binary mask, shape [*, n].

        Returns:
            sample x masked.
        """
        x = {
            data_mode: self.base_flow_matchers[data_mode]._apply_mask(
                x=x[data_mode],
                mask=mask,
            )
            for data_mode in self.data_modes
        }
        return x

    def sample_noise(
        self,
        n: int,
        shape: Tuple = tuple(),
        device: Optional[torch.device] = None,
        mask: Optional[Bool[Tensor, "* n"]] = None,
    ) -> Dict[str, Tensor]:
        """
        Samples reference distribution (possibly centered).

        Args:
            n: number of residues in a single sample (i.e. protein length), int
            shape: tuple (if empty then single sample)
            device (optional): torch device used
            mask (optional): Binary mask of shape [*, n]

        Returns:
            Samples from refenrece with shape [*shaqpe, n, ...]
        """
        x = {
            data_mode: self.base_flow_matchers[data_mode].sample_noise(
                n=n,
                shape=shape,
                device=device,
                mask=mask,
            )
            for data_mode in self.data_modes
        }
        return x

    def interpolate(
        self,
        x_0: Dict[str, torch.Tensor],
        x_1: Dict[str, torch.Tensor],
        t: Dict[str, Float[Tensor, "*"]],
        mask: Optional[Bool[Tensor, "* n"]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Interpolates between x_0 (base) and x_1 (data) using t.

        Args:
            x_0: Samples from reference with batch shape * (each value in the dict)
            x_1: Samples from target with batch shape *
            t: Interpolation times
            mask (optional): Binary mask, shape [*, n]

        Returns:
            x_t: Interpolated samples, same shape as x_0 and x_1
        """
        x_t = {
            data_mode: self.base_flow_matchers[data_mode].interpolate(
                x_0=x_0[data_mode],
                x_1=x_1[data_mode],
                t=t[data_mode],
                mask=mask,
            )
            for data_mode in self.data_modes
        }
        return x_t

    def process_batch(
        self, batch: Dict
    ) -> Tuple[Tensor, Tensor, Tuple, int, torch.dtype]:
        """
        Extracts clean sample, mask, batch size, protein length n, dtype and device
        from the batch coming from the dataloader. (Do we want dtype? Should default
        work?)

        Args:
            batch: batch from dataloader.

        Returns:
            Tuple (x_1, mask, batch_shape, n, dtype, device)

        WARNING: For dtype, it might become a dictionary as well, depending on
        type for sequence (discrete)
        """
        coors_tensor = batch["coords"]  # [b, n, 37, 3]
        device = coors_tensor.device
        dtype = coors_tensor.dtype
        batch_shape = coors_tensor.shape[:-3]
        n = coors_tensor.shape[-3]
        mask = batch["mask_dict"]["coords"][..., 0, 0]  # [b, n] boolean
        x_1 = self._apply_mask(x=batch["x_1"], mask=mask)
        return (x_1, mask, batch_shape, n, dtype, device)

    def corrupt_batch(
        self,
        batch: Dict,
    ) -> Dict:
        """
        Runs forward process on x_1. Essentially samples reference and
        interpolates. If there's any coupling it should be included here.

        Augmentations in principle happen in the dataloader.

        Args:
            batch: data batch

        Returns:
            The same data batch with additional entries t, x_t, x_1, x_0, all of which are
            Dict[str, Tensor].
        """
        x_1, mask, batch_shape, n, dtype, device = self.process_batch(batch)
        t = self.sample_t(shape=batch_shape, device=device)
        x_0 = self.sample_noise(n=n, shape=batch_shape, mask=mask, device=device)
        x_t = self.interpolate(x_0=x_0, x_1=x_1, t=t, mask=mask)
        batch["x_0"] = x_0
        batch["x_1"] = x_1
        batch["x_t"] = x_t
        batch["t"] = t
        batch["mask"] = mask
        return batch

    def sample_t(
        self, shape: tuple, device: torch.device
    ) -> Dict[str, Float[Tensor, "*shape"]]:
        """
        Samples t for each data mode. Can use different distributions for each mode,
        and different modes can share the same t. This is all controlled in the config.

        Args:
            shape: shape of the sample
            device: device of the sample

        Returns:
            Sampled t for each modality (as a dictionary), each tensor has shape [*shape].
        """
        t = {
            data_mode: _sample_t(
                cfg_t_dist=self.cfg_exp.loss.t_distribution[data_mode],
                shape=shape,
                device=device,
            )
            for data_mode in self.data_modes
        }
        if self.cfg_exp.loss.t_distribution.shared_groups is None:
            return t

        # Apply grouping for data modalities we want the same t
        for t_share_modes in self.cfg_exp.loss.t_distribution.shared_groups:
            # each t_shared_modes is a list
            base = t_share_modes[0]  # some modality, the first one in the list
            shared_t = t[base]  # [*]
            for data_mode in t_share_modes[1:]:
                t[data_mode] = shared_t
        return t

    def compute_loss(
        self, batch: Dict, nn_out: Dict[str, Dict[str, Tensor]]
    ) -> Dict[str, Float[Tensor, "*"]]:
        """
        Computes training loss, flow matching and some auxiliary loss.

        Args:
            batch: training batch with clean x_1, x_0, x_t, t
            nn_out: output of nn

        Returns:
            Flow matching loss and auxiliary loss, all in one Dict[str, Tensor w shape [*]]. Each loss
            is identified by its name (str) and is the loss per element in the batch (i.e. before mean
            reduction).
        """
        fm_loss = self.compute_fm_loss(batch, nn_out)
        aux_loss = self.compute_aux_loss(batch, nn_out)
        loss = {**fm_loss, **aux_loss}
        return loss

    def compute_fm_loss(
        self,
        batch: Dict,
        nn_out: Dict[str, Dict[str, Tensor]],
    ) -> Dict[str, Float[Tensor, "*"]]:
        """
        Computes flow matching loss for all modalities.

        Args:
            batch: Input batch containing x_1, x_0, t, mask, etc
            nn_out: Output of nn

        Returns:
            Losses for each data modality, per element in the batch.
        """
        loss = {
            data_mode: self.base_flow_matchers[data_mode].compute_fm_loss(
                x_0=batch["x_0"][data_mode],
                x_1=batch["x_1"][data_mode],
                x_t=batch["x_t"][data_mode],
                mask=batch["mask"],
                t=batch["t"][data_mode],
                nn_out=nn_out[data_mode],
            )
            for data_mode in self.data_modes
        }
        return loss

    def compute_aux_loss(
        self,
        batch: Dict,
        nn_out: Dict[str, Dict[str, Tensor]],
    ) -> Dict[str, Float[Tensor, "*"]]:
        """
        Computes auxiliary loss (if any). This is done here, and not in each specific flow matcher,
        since it may be a full atom loss, etc, meaning it may interact between modalities.

        Args:
            batch: Input batch containing x_1, x_0, t, mask, etc
            nn_out: Output of nn

        Returns:
            Losses for each data modality, per element in the batch.
        """
        losses = {}

        # Optional motif loss
        if "x_motif" in batch:
            motif_mask = batch["motif_mask"]  # [b, n, 37]
            mask_losses = motif_mask.sum(-1).bool()  # [b, n]
            motif_loss = {
                data_mode: self.base_flow_matchers[data_mode].compute_fm_loss(
                    x_0=batch["x_0"][data_mode],
                    x_1=batch["x_1"][data_mode],
                    x_t=batch["x_t"][data_mode],
                    mask=mask_losses,
                    t=batch["t"][data_mode],
                    nn_out=nn_out[data_mode],
                )
                for data_mode in self.data_modes
            }
            for data_mode in motif_loss:
                losses[data_mode + f"motif_loss_now_justlog"] = motif_loss[data_mode]

        return losses

    def simulation_step(
        self,
        x_t: Dict[str, torch.Tensor],
        nn_out: Dict[str, Dict[str, torch.Tensor]],
        t: Dict[str, Float[Tensor, "*"]],
        dt: Dict[str, float],
        gt: Dict[str, float],
        simulation_step_params: Dict[str, Dict],
        mask: Optional[Bool[Tensor, "* n"]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Single integration step of ODE \dot{x_t} = v(x_t, t) using Euler integration scheme.

        Args:
            x_t: Current values with batch shape *
            nn_out: Output of nn
            t: Current time, shape * for each value in the dict
            dt: Step-size, float
            gt: Noise injection schedule, float
            simulation_step_params contain extra simulation steps parameters
            mask (optional): Binary mask of shape [*, n]

        Returns:
            Updated x_t after integration step, same shape as input
        """
        x_updated = {
            data_mode: self.base_flow_matchers[data_mode].simulation_step(
                x_t=x_t[data_mode],
                nn_out=nn_out[data_mode],
                t=t[data_mode],
                dt=dt[data_mode],
                gt=gt[data_mode],
                simulation_step_params=simulation_step_params[data_mode],
                mask=mask,
            )
            for data_mode in self.data_modes
        }
        return x_updated

    def nn_out_to_clean_sample_prediction(
        self,
        batch: Dict,
        nn_out: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Generates clean sample prediction from nn output.

        Args:
            batch: input to nn (needed because it contains t, x_t, and mask)
            nn_out: output of nn

        Returns:
            Clean sample prediction {data_mode: torch.Tensor} from nn output
        """
        nn_out = self.nn_out_add_clean_sample_prediction(batch=batch, nn_out=nn_out)
        return {data_mode: nn_out[data_mode]["x_1"] for data_mode in self.data_modes}

    def nn_out_add_clean_sample_prediction(
        self,
        batch: Dict,
        nn_out: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Adds clean sample prediction to nn output.

        Args:
            batch: input to nn (needed because it contains t, x_t, and mask)
            nn_out: output of nn

        Returns:
            nn_out with clean sample prediction ("x_1" key) added to each data modality.
        """
        for data_mode in self.data_modes:
            nn_out[data_mode] = self.base_flow_matchers[
                data_mode
            ].nn_out_add_clean_sample_prediction(
                x_t=batch["x_t"][data_mode],
                t=batch["t"][data_mode],
                mask=batch["mask"],
                nn_out=nn_out[data_mode],
            )
        return nn_out

    def nn_out_add_simulation_tensor(
        self,
        batch: Dict,
        nn_out: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Adds tensor used for simulation to nn output. This is the velocity "v" for flow matching,
        or "score" for diffusion models. One could work out their equivalence and leverage that,
        but keeping each one seems like an easier implementation.

        Args:
            batch: input to nn (needed because it contains t, x_t, and mask)
            nn_out: output of nn

        Returns:
            nn_out with simulation tensor prediction ("v", "score", ... keys) added to data modalities.
        """
        for data_mode in self.data_modes:
            nn_out[data_mode] = self.base_flow_matchers[
                data_mode
            ].nn_out_add_simulation_tensor(
                x_t=batch["x_t"][data_mode],
                t=batch["t"][data_mode],
                mask=batch["mask"],
                nn_out=nn_out[data_mode],
            )
        return nn_out

    def nn_out_add_guided_simulation_tensor(
        self,
        nn_out: Dict[str, Dict[str, torch.Tensor]],
        nn_out_ag: Union[Dict[str, Dict[str, torch.Tensor]], None],
        nn_out_ucond: Union[Dict[str, Dict[str, torch.Tensor]], None],
        guidance_w: float,
        ag_ratio: float,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Adds guided version of tensor used for simulation to nn output 'nn_out'. Supports classifier free
        guidance and autoguidance.

        Args:
            nn_out: nn output of full model
            nn_out_ag: nn output of model used for autoguidance
            nn_out_ucond: nn output of full but unconditional model
            guidance_w: Guidance weights, float
            ag_ratio: Autoguidance ratio, float

        Returns:
            nn_out with guided simulation tensor prediction ("v_guided", "score_guided", ... keys) added to data modalities
            with guidance enabled.
        """
        for data_mode in self.data_modes:
            nn_out[data_mode] = self.base_flow_matchers[
                data_mode
            ].nn_out_add_guided_simulation_tensor(
                nn_out=nn_out[data_mode],
                nn_out_ag=nn_out_ag[data_mode] if nn_out_ag else None,
                nn_out_ucond=nn_out_ucond[data_mode] if nn_out_ucond else None,
                guidance_w=guidance_w,
                ag_ratio=ag_ratio,
            )
        return nn_out

    def get_clean_pred_n_guided_vector(
        self,
        batch: Dict,
        predict_for_sampling: Callable,
        guidance_w: float,
        ag_ratio: float,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        This makes a prediction with the corresponding nn and conditioning, computes the
        corresponding clean sample prediction, and v/score, and also the guided v/score.

        When calling `predict_for_sampling` (which internally calls some nn), the output
        nn_out is like
        {
            "bb_ca": {"v": tensor}
            "frames": {"x1": tensor}
        }
        depending on the parameterization, which is indicated in the nn config file. Then,
        we call `nn_out_add_clean_sample_prediction` and `nn_out_add_simulation_tensor` to
        extend each data mode in nn_out with "v"/"score" (the simulation tensor) and "x_1".

        This function does this for the original nn, the unconditional input, and the
        autoguidance nn, and combines predictions accordingly, adding to nn_out an entry
        `v_guided`. Overall, it returns nn_out like
        {
            "bb_ca": {"v": tensor, "x_1": tensor, "v_guided": tensor}
            "frames": {"x1": tensor, "v": tensor, "v_guided": tensor}
        }
        where v could be replaced by the score in case of a diffusion.

        Args:
            batch: input to nn
            predict_for_sampling: function that takes as input batch and mode, where the latter indicated
            what model to evaluate (supports "full", "ag", "ucond").
            guidance_w: guidance weights
            ag_ratio: autoguidance ratio

        Returns:
            an nn_out dictionary with keys "x_1", "v" (or "score") and "v_guided" (or "score_guided").
        """

        def _add_clean_n_sim_tensor(batch, mode):
            """
            We will use this for each model (full, ag, ucond).
            """
            nn_out_dict = predict_for_sampling(batch, mode=mode)
            nn_out_dict = self.nn_out_add_clean_sample_prediction(batch, nn_out_dict)
            nn_out_dict = self.nn_out_add_simulation_tensor(batch, nn_out_dict)
            return nn_out_dict

        nn_out = _add_clean_n_sim_tensor(batch, mode="full")
        nn_out_ag = None
        nn_out_ucond = None
        if guidance_w != 1.0:
            if ag_ratio > 0.0:  # Use auto-guidance
                nn_out_ag = _add_clean_n_sim_tensor(batch, mode="ag")
            if ag_ratio < 1.0:  # Use CFG
                nn_out_ucond = _add_clean_n_sim_tensor(batch, mode="ucond")

        nn_out = self.nn_out_add_guided_simulation_tensor(
            nn_out=nn_out,
            nn_out_ag=nn_out_ag,
            nn_out_ucond=nn_out_ucond,
            guidance_w=guidance_w,
            ag_ratio=ag_ratio,
        )  # Adds guided tensor used for simulation ("guided_v", "guided_score", or whatever the base flow matcher uses)
        return nn_out

    def adaptive_heun_simulation(
        self,
        batch: Dict,
        predict_for_sampling: Callable,
        max_nfe: int,
        nsamples: int,
        n: int,
        self_cond: bool,
        sampling_model_args: Dict[str, Dict],
        device: torch.device,
        save_trajectory_every: int = 0,
        guidance_w: float = 1.0,
        ag_ratio: float = 0.0,
        atol: float = 1e-3,
        rtol: float = 1e-2,
        dt_init: float = 0.01,
        dt_min: float = 1e-4,
        dt_max: float = 0.1,
        safety_factor: float = 0.9,
    ) -> Tuple[Dict[str, Tensor], Dict]:
        """
        Adaptive Heun-Euler embedded method for ODE integration from t=0 to t=1.

        Uses the Heun method (trapezoidal rule) for stepping and the difference
        between the Euler trial and Heun-corrected steps as a free local
        truncation error estimate. Step size is adjusted based on the error
        ratio, with steps rejected when the error exceeds tolerance.

        All samples in a batch share the same adaptive schedule (max error
        across samples drives acceptance/rejection).

        Args:
            batch: Input batch dict.
            predict_for_sampling: Callable that runs the neural network.
            max_nfe: Maximum number of function evaluations (NFE budget).
            nsamples: Number of samples to generate.
            n: Protein length.
            self_cond: Whether to use self-conditioning.
            sampling_model_args: Model-specific sampling arguments.
            device: Torch device.
            save_trajectory_every: Save trajectory at NFE milestones (0 to disable).
            guidance_w: Guidance weight.
            ag_ratio: Autoguidance ratio.
            atol: Absolute error tolerance.
            rtol: Relative error tolerance.
            dt_init: Initial step size.
            dt_min: Minimum step size.
            dt_max: Maximum step size.
            safety_factor: Safety factor for step size adjustment.

        Returns:
            x: Generated samples dict.
            additional_info: Dict with mask, trajectory, solver info, NFE counts.
        """
        # Squeeze batch-dim-1 tensors (same as full_simulation)
        for key, value in batch.items():
            if (
                isinstance(value, torch.Tensor)
                and value.dim() > 0
                and value.size(0) == 1
            ):
                batch[key] = value.squeeze(0)

        # Use mask from batch if present, otherwise generate it
        if "mask" in batch and batch["mask"] is not None:
            mask = batch["mask"]
        else:
            mask = torch.ones(nsamples, n).long().bool().to(device)
        assert mask.shape == (nsamples, n)

        trajectory = []

        with torch.no_grad():
            use_bf16 = os.environ.get("LP_USE_BF16", "1") == "1" and device.type == "cuda"
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                x = self.sample_noise(
                    n,
                    shape=(nsamples,),
                    device=device,
                    mask=mask,
                )

                t = 0.0
                dt = dt_init
                nfe_used = 0
                steps_accepted = 0
                steps_rejected = 0
                x_1_pred = None  # for self-conditioning

                while t < 1.0 and nfe_used + 2 <= max_nfe:
                    dt = min(dt, 1.0 - t)

                    # --- First NN evaluation at (x, t) ---
                    t_tensor = {
                        dm: t * torch.ones(nsamples, device=device)
                        for dm in self.data_modes
                    }
                    batch["x_t"] = x
                    batch["t"] = t_tensor
                    batch["mask"] = mask

                    if steps_accepted > 0 and self_cond and x_1_pred is not None:
                        batch["x_sc"] = x_1_pred

                    nn_out_1 = self.get_clean_pred_n_guided_vector(
                        batch=batch,
                        predict_for_sampling=predict_for_sampling,
                        guidance_w=guidance_w,
                        ag_ratio=ag_ratio,
                    )
                    nfe_used += 1

                    v1 = {}
                    for dm in self.data_modes:
                        v1[dm] = nn_out_1[dm].get("v_guided", nn_out_1[dm]["v"])

                    x_1_pred_1 = self.nn_out_to_clean_sample_prediction(
                        batch=batch, nn_out=nn_out_1
                    )

                    # --- Trial Euler step ---
                    x_trial = {dm: x[dm] + v1[dm] * dt for dm in self.data_modes}
                    x_trial = self._apply_mask(x_trial, mask)
                    for dm in self.data_modes:
                        if sampling_model_args[dm]["simulation_step_params"].get("center_every_step", False):
                            x_trial[dm] = self.base_flow_matchers[dm]._force_zero_com(x_trial[dm], mask)

                    # --- Second NN evaluation at (x_trial, t + dt) ---
                    t_trial_tensor = {
                        dm: (t + dt) * torch.ones(nsamples, device=device)
                        for dm in self.data_modes
                    }
                    batch["x_t"] = x_trial
                    batch["t"] = t_trial_tensor
                    if self_cond:
                        batch["x_sc"] = x_1_pred_1

                    nn_out_2 = self.get_clean_pred_n_guided_vector(
                        batch=batch,
                        predict_for_sampling=predict_for_sampling,
                        guidance_w=guidance_w,
                        ag_ratio=ag_ratio,
                    )
                    nfe_used += 1

                    v2 = {}
                    for dm in self.data_modes:
                        v2[dm] = nn_out_2[dm].get("v_guided", nn_out_2[dm]["v"])

                    # --- Error estimate ---
                    # err = 0.5 * (v2 - v1) * dt  (local truncation error of Euler)
                    err_ratio = 0.0
                    for dm in self.data_modes:
                        err_dm = 0.5 * (v2[dm] - v1[dm]) * dt
                        x_scale = atol + rtol * torch.max(
                            torch.abs(x[dm]), torch.abs(x_trial[dm])
                        )
                        scaled = err_dm / x_scale  # [b, n, d]
                        nres = mask.sum(dim=-1)  # [b]
                        d = scaled.shape[-1]
                        per_sample = torch.sqrt(
                            (scaled**2 * mask[..., None]).sum(dim=(-1, -2)) / (nres * d)
                        )  # [b]
                        err_ratio = max(err_ratio, per_sample.max().item())

                    if err_ratio <= 1.0:
                        # Accept step: Heun update
                        x = {
                            dm: x[dm] + 0.5 * (v1[dm] + v2[dm]) * dt
                            for dm in self.data_modes
                        }
                        x = self._apply_mask(x, mask)
                        for dm in self.data_modes:
                            if sampling_model_args[dm]["simulation_step_params"].get("center_every_step", False):
                                x[dm] = self.base_flow_matchers[dm]._force_zero_com(x[dm], mask)

                        t += dt
                        steps_accepted += 1

                        # Update x_1_pred from second (better) eval for self-conditioning
                        x_1_pred = self.nn_out_to_clean_sample_prediction(
                            batch=batch, nn_out=nn_out_2
                        )

                        # Save trajectory at NFE milestones
                        if save_trajectory_every > 0 and nfe_used % save_trajectory_every == 0:
                            trajectory.append({
                                "step": nfe_used,
                                "x": {k: v.detach().clone() for k, v in x.items()},
                            })
                    else:
                        steps_rejected += 1
                        # On rejection: x_1_pred stays from last accepted step

                    # --- Adjust dt ---
                    if err_ratio > 0:
                        dt *= safety_factor * (1.0 / err_ratio) ** 0.5
                    else:
                        dt *= 2.0  # error is zero; double the step
                    dt = max(dt_min, min(dt, dt_max))

                # If t < 1.0 and budget exhausted, do one final forced Heun step
                if t < 1.0 and nfe_used + 2 <= max_nfe + 2:
                    dt_final = 1.0 - t
                    t_tensor = {
                        dm: t * torch.ones(nsamples, device=device)
                        for dm in self.data_modes
                    }
                    batch["x_t"] = x
                    batch["t"] = t_tensor
                    batch["mask"] = mask
                    if self_cond and x_1_pred is not None:
                        batch["x_sc"] = x_1_pred

                    nn_out_1 = self.get_clean_pred_n_guided_vector(
                        batch=batch,
                        predict_for_sampling=predict_for_sampling,
                        guidance_w=guidance_w,
                        ag_ratio=ag_ratio,
                    )
                    nfe_used += 1
                    v1 = {dm: nn_out_1[dm].get("v_guided", nn_out_1[dm]["v"]) for dm in self.data_modes}
                    x_1_pred_1 = self.nn_out_to_clean_sample_prediction(batch=batch, nn_out=nn_out_1)

                    x_trial = {dm: x[dm] + v1[dm] * dt_final for dm in self.data_modes}
                    x_trial = self._apply_mask(x_trial, mask)
                    for dm in self.data_modes:
                        if sampling_model_args[dm]["simulation_step_params"].get("center_every_step", False):
                            x_trial[dm] = self.base_flow_matchers[dm]._force_zero_com(x_trial[dm], mask)

                    t_trial_tensor = {
                        dm: 1.0 * torch.ones(nsamples, device=device)
                        for dm in self.data_modes
                    }
                    batch["x_t"] = x_trial
                    batch["t"] = t_trial_tensor
                    if self_cond:
                        batch["x_sc"] = x_1_pred_1

                    nn_out_2 = self.get_clean_pred_n_guided_vector(
                        batch=batch,
                        predict_for_sampling=predict_for_sampling,
                        guidance_w=guidance_w,
                        ag_ratio=ag_ratio,
                    )
                    nfe_used += 1
                    v2 = {dm: nn_out_2[dm].get("v_guided", nn_out_2[dm]["v"]) for dm in self.data_modes}

                    x = {dm: x[dm] + 0.5 * (v1[dm] + v2[dm]) * dt_final for dm in self.data_modes}
                    x = self._apply_mask(x, mask)
                    for dm in self.data_modes:
                        if sampling_model_args[dm]["simulation_step_params"].get("center_every_step", False):
                            x[dm] = self.base_flow_matchers[dm]._force_zero_com(x[dm], mask)

                    t += dt_final
                    steps_accepted += 1
                    x_1_pred = self.nn_out_to_clean_sample_prediction(batch=batch, nn_out=nn_out_2)

                logger.info(
                    f"Adaptive Heun: t_final={t:.6f}, NFEs={nfe_used}/{max_nfe}, "
                    f"steps_accepted={steps_accepted}, steps_rejected={steps_rejected}"
                )

        additional_info = {
            "mask": mask,
            "trajectory": trajectory,
            "solver": "adaptive_heun",
            "nfe_total": nfe_used,
            "nsteps_taken": steps_accepted,
            "nsteps_rejected": steps_rejected,
        }
        return x, additional_info

    def full_simulation(
        self,
        batch: Dict,
        predict_for_sampling: Callable,
        nsteps: int,
        nsamples: int,
        n: int,
        self_cond: bool,
        sampling_model_args: Dict[str, Dict],
        device: torch.device,
        save_trajectory_every: int = 0,
        guidance_w: float = 1.0,
        ag_ratio: float = 0.0,
        solver: str = "euler",
    ) -> Dict[str, Tensor]:
        """
        Generates samples by simulating the

        dx_t = v(x_t, t) dt

        or

        dx_t = [v(x_t, t) + g(t) s(x_t, t)] dt + \sqrt{2g(t)} dw_t

        from t=0 up to t=1.

        Which equation is simulated depends on data modality and parameters chosen.

        Args:
            batch (Dict): Input batch. May contain:
                - 'nsamples': int, number of samples to generate (optional, will use nsamples param if not present)
                - 'nres' or 'n': int, protein length (optional, will use n param if not present)  
                - 'mask': torch.BoolTensor, shape [nsamples, n], mask for valid residues (optional, will be generated if not present)
                - Any conditioning fields required by the prediction function (e.g., cath_codes, motif info, etc.)
                
                The following fields will be added/updated during simulation:
                - 'x_t': torch.Tensor, current state (updated each step)
                - 't': dict of time values for each data mode (updated each step)
                - 'mask': torch.BoolTensor, shape [nsamples, n], mask for valid residues (added if not present)
                - 'x_sc': torch.Tensor, self-conditioning input (added if self_cond=True)
            predict_for_sampling (Callable): Function that calls the nn, receives the full batch dict
            nsteps (int): Number of simulation steps
            nsamples (int): Number of samples to generate
            n (int): Protein length
            self_cond (bool): Whether to use self-conditioning
            sampling_model_args (Dict): Model-specific sampling arguments
            device (torch.device): Device to run simulation on
            save_trajectory_every (int): How often to save full generation trajectory, default 0 (no saving)
            guidance_w (float): Guidance weight
            ag_ratio (float): Autoguidance ratio

        Returns:
            x (dict): Generated samples, with keys for each data mode and values with batch shape nsamples
            mask (torch.Tensor): Boolean mask of shape [nsamples, n]
        """
        # overwrite n with the correct shape of coors_nm in batch
        for key, value in batch.items():
            if (
                isinstance(value, torch.Tensor)
                and value.dim() > 0
                and value.size(0) == 1
            ):
                batch[key] = value.squeeze(0)

        # Use mask from batch if present, otherwise generate it
        if "mask" in batch and batch["mask"] is not None:
            mask = batch["mask"]
        else:
            mask = torch.ones(nsamples, n).long().bool().to(device)
        assert mask.shape == (nsamples, n)
        
        assert solver in ("euler", "heun", "adaptive_heun", "stochastic_heun"), f"Unknown solver: {solver}"
        if solver in ("heun", "adaptive_heun"):
            for dm in self.data_modes:
                mode = sampling_model_args[dm]["simulation_step_params"]["sampling_mode"]
                assert mode == "vf", (
                    f"{solver} requires sampling_mode='vf' (ODE), but {dm} has '{mode}'"
                )

        if solver == "adaptive_heun":
            adaptive_cfg = sampling_model_args.get("adaptive", {})
            return self.adaptive_heun_simulation(
                batch=batch,
                predict_for_sampling=predict_for_sampling,
                max_nfe=nsteps * 2,
                nsamples=nsamples,
                n=n,
                self_cond=self_cond,
                sampling_model_args=sampling_model_args,
                device=device,
                save_trajectory_every=save_trajectory_every,
                guidance_w=guidance_w,
                ag_ratio=ag_ratio,
                **adaptive_cfg,
            )

        trajectory = []

        ts = {
            data_mode: get_schedule(
                mode=sampling_model_args[data_mode]["schedule"]["mode"],
                nsteps=int(nsteps),
                p1=sampling_model_args[data_mode]["schedule"]["p"],
            )
            for data_mode in self.data_modes
        }  # each [nsteps + 1], first element is 0, last is 1

        gt = {
            data_mode: get_gt(
                t=ts[data_mode][:-1],  # [nsteps], ts[-1]=1 not used to eval
                mode=sampling_model_args[data_mode]["gt"]["mode"],
                param=sampling_model_args[data_mode]["gt"]["p"],
                clamp_val=sampling_model_args[data_mode]["gt"]["clamp_val"],
            )
            for data_mode in self.data_modes
        }

        # Stochastic Heun needs gt at the trial points (t+dt) for drift computation
        if solver == "stochastic_heun":
            gt_trial = {
                data_mode: get_gt(
                    t=ts[data_mode][1:],  # [nsteps], gt at trial endpoints
                    mode=sampling_model_args[data_mode]["gt"]["mode"],
                    param=sampling_model_args[data_mode]["gt"]["p"],
                    clamp_val=sampling_model_args[data_mode]["gt"]["clamp_val"],
                )
                for data_mode in self.data_modes
            }

        with torch.no_grad():
            use_bf16 = os.environ.get("LP_USE_BF16", "1") == "1" and device.type == "cuda"
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                x = self.sample_noise(
                    n,
                    shape=(nsamples,),
                    device=device,
                    mask=mask,
                )

                for step in range(nsteps):
                    t = {
                        data_mode: ts[data_mode][step] * torch.ones(nsamples, device=device)
                        for data_mode in self.data_modes
                    }
                    dt = {
                        data_mode: ts[data_mode][step + 1] - ts[data_mode][step]
                        for data_mode in self.data_modes
                    }
                    gt_step = {
                        data_mode: gt[data_mode][step] for data_mode in self.data_modes
                    }

                    # Update the batch with current x, t, mask, etc.
                    batch["x_t"] = x
                    batch["t"] = t
                    batch["mask"] = mask

                    # self conditioning
                    if step > 0 and self_cond:
                        batch["x_sc"] = x_1_pred

                    # get clean prediction and guided vector
                    nn_out = self.get_clean_pred_n_guided_vector(
                        batch=batch,
                        predict_for_sampling=predict_for_sampling,
                        guidance_w=guidance_w,
                        ag_ratio=ag_ratio,
                    )
                    # Dict[data_mode,
                    #   Dict[str, torch.Tensor]
                    # ] where str is some prediction ("x_1", "v", "score", "guided_v", "guided_score", ...)
                    # We just track all predictions this way

                    x_1_pred = self.nn_out_to_clean_sample_prediction(
                        batch=batch, nn_out=nn_out
                    )
                    # Dict[data_mode, torch.Tensor]

                    if solver == "heun":
                        # Extract v1 from first evaluation
                        v1 = {}
                        for dm in self.data_modes:
                            v1[dm] = nn_out[dm].get("v_guided", nn_out[dm]["v"])

                        # Trial Euler step
                        x_trial = {dm: x[dm] + v1[dm] * dt[dm] for dm in self.data_modes}
                        x_trial = self._apply_mask(x_trial, mask)
                        for dm in self.data_modes:
                            if sampling_model_args[dm]["simulation_step_params"].get("center_every_step", False):
                                x_trial[dm] = self.base_flow_matchers[dm]._force_zero_com(x_trial[dm], mask)

                        # Second NN eval at trial point (x_trial, t+dt)
                        t_trial = {
                            dm: ts[dm][step + 1] * torch.ones(nsamples, device=device)
                            for dm in self.data_modes
                        }
                        batch["x_t"] = x_trial
                        batch["t"] = t_trial
                        if self_cond:
                            batch["x_sc"] = x_1_pred  # use first eval's prediction

                        nn_out_2 = self.get_clean_pred_n_guided_vector(
                            batch=batch,
                            predict_for_sampling=predict_for_sampling,
                            guidance_w=guidance_w,
                            ag_ratio=ag_ratio,
                        )

                        # Extract v2 and apply Heun correction
                        v2 = {}
                        for dm in self.data_modes:
                            v2[dm] = nn_out_2[dm].get("v_guided", nn_out_2[dm]["v"])

                        x = {dm: x[dm] + 0.5 * (v1[dm] + v2[dm]) * dt[dm] for dm in self.data_modes}
                        x = self._apply_mask(x, mask)
                        for dm in self.data_modes:
                            if sampling_model_args[dm]["simulation_step_params"].get("center_every_step", False):
                                x[dm] = self.base_flow_matchers[dm]._force_zero_com(x[dm], mask)

                        # Update x_1_pred from second (better) eval for next step's self-conditioning
                        x_1_pred = self.nn_out_to_clean_sample_prediction(
                            batch=batch, nn_out=nn_out_2
                        )

                    elif solver == "stochastic_heun":
                        # Stochastic Heun: trapezoidal correction on drift,
                        # same Brownian increment for trial and final steps.
                        v1 = {dm: nn_out[dm].get("v_guided", nn_out[dm]["v"]) for dm in self.data_modes}

                        # Compute drift and diffusion at (x, t)
                        drift1 = {}
                        noise_term = {}
                        for dm in self.data_modes:
                            params = sampling_model_args[dm]["simulation_step_params"]
                            d1, dc1 = self.base_flow_matchers[dm].compute_drift_and_diffusion_coeff(
                                x[dm], v1[dm], t[dm], gt_step[dm], params
                            )
                            drift1[dm] = d1
                            eps_dm = torch.randn_like(x[dm])
                            noise_term[dm] = dc1 * torch.sqrt(dt[dm]) * eps_dm

                        # Trial Euler-Maruyama step (drift + same noise)
                        x_trial = {dm: x[dm] + drift1[dm] * dt[dm] + noise_term[dm] for dm in self.data_modes}
                        x_trial = self._apply_mask(x_trial, mask)
                        for dm in self.data_modes:
                            if sampling_model_args[dm]["simulation_step_params"].get("center_every_step", False):
                                x_trial[dm] = self.base_flow_matchers[dm]._force_zero_com(x_trial[dm], mask)

                        # Second NN eval at trial point (x_trial, t+dt)
                        t_trial = {
                            dm: ts[dm][step + 1] * torch.ones(nsamples, device=device)
                            for dm in self.data_modes
                        }
                        batch["x_t"] = x_trial
                        batch["t"] = t_trial
                        if self_cond:
                            batch["x_sc"] = x_1_pred

                        nn_out_2 = self.get_clean_pred_n_guided_vector(
                            batch=batch,
                            predict_for_sampling=predict_for_sampling,
                            guidance_w=guidance_w,
                            ag_ratio=ag_ratio,
                        )

                        # Compute drift at trial point
                        v2 = {dm: nn_out_2[dm].get("v_guided", nn_out_2[dm]["v"]) for dm in self.data_modes}
                        drift2 = {}
                        for dm in self.data_modes:
                            params = sampling_model_args[dm]["simulation_step_params"]
                            d2, _ = self.base_flow_matchers[dm].compute_drift_and_diffusion_coeff(
                                x_trial[dm], v2[dm], t_trial[dm], gt_trial[dm][step], params
                            )
                            drift2[dm] = d2

                        # Stochastic Heun update: trapezoidal on drift, same noise realization
                        x = {
                            dm: x[dm] + 0.5 * (drift1[dm] + drift2[dm]) * dt[dm] + noise_term[dm]
                            for dm in self.data_modes
                        }
                        x = self._apply_mask(x, mask)
                        for dm in self.data_modes:
                            if sampling_model_args[dm]["simulation_step_params"].get("center_every_step", False):
                                x[dm] = self.base_flow_matchers[dm]._force_zero_com(x[dm], mask)

                        x_1_pred = self.nn_out_to_clean_sample_prediction(
                            batch=batch, nn_out=nn_out_2
                        )

                    else:
                        # Existing Euler path
                        simulation_step_params = {
                            data_mode: sampling_model_args[data_mode]["simulation_step_params"]
                            for data_mode in self.data_modes
                        }
                        x = self.simulation_step(
                            x_t=x,
                            nn_out=nn_out,
                            t=t,
                            dt=dt,
                            gt=gt_step,
                            mask=mask,
                            simulation_step_params=simulation_step_params,
                        )

                    if save_trajectory_every > 0 and (step + 1) % save_trajectory_every == 0:
                        trajectory.append({
                            "step": step + 1,
                            "x": {k: v.detach().clone() for k, v in x.items()},
                        })

            additional_info = {
                "mask": mask,
                "trajectory": trajectory,
                "solver": solver,
                "nfe_total": nsteps * (2 if solver in ("heun", "stochastic_heun") else 1),
            }
            return x, additional_info


def get_gt(
    t: Float[Tensor, "nsteps"],
    mode: str,
    param: float,
    clamp_val: Optional[float] = None,
    eps: float = 1e-2,
) -> Float[Tensor, "nsteps"]:
    """
    Computes gt (noise injection schedule) for different modes. This is in the SDE as

    dx_t = [v(x_t, t) + g(t) s(x_t, t)] dt + \sqrt{2g(t)} dw_t

    Args:
        t: times where we'll evaluate, covers [0, 1), shape [nsteps]
        mode: "1-t/t", "tan", "1/t", ...
        param: parameterized transformation
        clamp_val: value to clamp gt, no clamping if None
        eps: small value leave as it is

    Returns:
        Noise injection schedule, shape [nsteps]
    """

    def transform_gt(
        gt: Float[Tensor, "nsteps"], f_pow: float = 1.0
    ) -> Float[Tensor, "nsteps"]:
        """Applies a transformation to the gt. f_pow=1.0 means no transformation."""
        if f_pow == 1.0:
            return gt

        # First we somewhat normalize between 0 and 1
        log_gt = torch.log(gt)
        mean_log_gt = torch.mean(log_gt)
        log_gt_centered = log_gt - mean_log_gt
        normalized = torch.nn.functional.sigmoid(log_gt_centered)
        # Transformation here
        normalized = normalized**f_pow
        # Undo normalization with the transformed variable
        log_gt_centered_rec = torch.logit(normalized, eps=1e-6)
        log_gt_rec = log_gt_centered_rec + mean_log_gt
        gt_rec = torch.exp(log_gt_rec)
        return gt_rec

    t = torch.clamp(t, 0, 1 - 1e-5)  # For numerical reasons

    if mode == "1-t/t":
        num = 1.0 - t
        den = t
        gt = num / (den + eps)
    elif mode == "tan":
        num = torch.sin((1.0 - t) * torch.pi / 2.0)
        den = torch.cos((1.0 - t) * torch.pi / 2.0)
        gt = (torch.pi / 2.0) * num / (den + eps)
    elif mode == "1/t":
        num = 1.0
        den = t
        gt = num / (den + eps)
    else:
        raise NotImplementedError(f"gt not implemented {mode}")
    gt = transform_gt(gt, f_pow=param)
    gt = torch.clamp(gt, 0, clamp_val)  # If None no clamping
    return gt  # [s]


def get_schedule(
    mode: str, nsteps: int, *, p1: float = None, eps: float = 1e-5
) -> Float[Tensor, "nsteps_p_one"]:
    """
    Gets the partition of the unit interval with points where we'll evaluate the vector field / score.
    """
    if mode == "uniform":
        t = torch.linspace(0, 1, nsteps + 1)
        return t
    elif mode == "power":
        assert p1 is not None, "p1 cannot be none for the power schedule"
        t = torch.linspace(0, 1, nsteps + 1)
        t = t**p1
        return t
    elif mode == "log":
        assert p1 is not None, "p1 cannot be none for the log schedule"
        assert p1 > 0, f"p1 must be >0 for the log schedule, got {p1}"
        t = 1.0 - torch.logspace(-p1, 0, nsteps + 1).flip(0)
        t = t - torch.min(t)
        t = t / torch.max(t)
        return t
    else:
        raise IOError(f"Schedule mode not recognized {mode}")


def _sample_t(
    cfg_t_dist: Dict, shape: Tuple, device=torch.device
) -> Float[Tensor, "*shape"]:
    """
    Samples time from different time distributions.

    Args:
        cfg_t_dist: dict specifying distribution to sample
        shape: shape of the returned tensor t
        device: torch.device

    Returns:
        Tensor with random samples t from the requested distribution and shape
    """
    if cfg_t_dist.name == "uniform":
        t_max = cfg_t_dist.p2
        return torch.rand(shape, device=device) * t_max
    elif cfg_t_dist.name == "logit-normal":
        mean = cfg_t_dist.p1
        std = cfg_t_dist.p2
        noise = torch.randn(shape, device=device) * std + mean
        return torch.nn.functional.sigmoid(noise)
    elif cfg_t_dist.name == "beta":
        p1 = cfg_t_dist.p1
        p2 = cfg_t_dist.p2
        dist = torch.distributions.beta.Beta(p1, p2)
        return dist.sample(shape).to(device)
    elif cfg_t_dist.name == "mix_unif_beta":
        p1 = cfg_t_dist.p1
        p2 = cfg_t_dist.p2
        p3 = cfg_t_dist.p3  # For mix weight of uniform
        assert 0.0 < p3 < 1.0, f"p3 value {p3} not in (0, 1)"
        dist = torch.distributions.beta.Beta(p1, p2)
        samples_beta = dist.sample(shape).to(device)
        samples_uniform = torch.rand(shape, device=device)
        u = torch.rand(shape, device=device)
        return torch.where(u < p3, samples_uniform, samples_beta)
    else:
        raise NotImplementedError(
            f"Sampling mode for t {cfg_t_dist.name} not implemented"
        )
