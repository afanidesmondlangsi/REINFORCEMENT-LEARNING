import math
import torch

from .ucdgt_bc_sac import UCDGTBCSAC


class UCDGTAdaptiveTau(UCDGTBCSAC):
    """
    UCDGT with online ESS-controlled gating temperature.

    Experimental extension of UCDGT.
    The original UCDGT implementation remains unchanged.
    """

    def __init__(self, config, device, target_entropy=None):
        super().__init__(config, device, target_entropy)

        self.tau_min = float(config.get("ucdgt_tau_min", 5.0))
        self.tau_max = float(config.get("ucdgt_tau_max", 20.0))
        self.tau_target_ess = float(
            config.get("ucdgt_tau_target_ess", 0.85)
        )
        self.tau_eta = float(config.get("ucdgt_tau_eta", 0.1))
        self.tau_update_interval = int(
            config.get("ucdgt_tau_update_interval", 100)
        )

        if not 0 < self.tau_min <= self.tau_max:
            raise ValueError("Invalid temperature bounds")

        if not 0 < self.tau_target_ess <= 1:
            raise ValueError("ESS target must be in (0, 1]")

        if self.tau_eta <= 0:
            raise ValueError("Temperature update rate must be positive")

        if self.tau_update_interval < 1:
            raise ValueError("Temperature update interval must be positive")

        if not self.tau_min <= self.ucdgt.tau <= self.tau_max:
            raise ValueError("Initial temperature outside bounds")

        self.tau_last_update_insertions = 0
        self.tau_update_count = 0
        self.tau_last_ess = None
        self.tau_last_ess_iteration = -1

    def _source_weights(self, state, action, next_state):
        weights = super()._source_weights(
            state, action, next_state
        )

        # Capture diagnostics without updating temperature here.
        with torch.no_grad():
            w = weights.detach().reshape(-1).float()

            if w.numel() > 0:
                ess = (
                    w.sum().square()
                    / (w.square().sum() + 1e-12)
                )

                self.tau_last_ess = float(
                    (ess / w.numel()).item()
                )
                self.tau_last_ess_iteration = self.total_it

        return weights

    def _update_temperature(self, writer=None):
        if not self.ucdgt_ready:
            return False

        insertions = int(self.ucdgt_target_insertions)

        if (
            insertions - self.tau_last_update_insertions
            < self.tau_update_interval
        ):
            return False

        if self.tau_last_ess is None:
            return False

        if self.tau_last_ess_iteration != self.total_it:
            return False

        old_tau = float(self.ucdgt.tau)

        new_tau = old_tau * math.exp(
            self.tau_eta
            * (self.tau_target_ess - self.tau_last_ess)
        )

        new_tau = min(
            self.tau_max,
            max(self.tau_min, new_tau)
        )

        self.ucdgt.tau = float(new_tau)

        self.tau_last_update_insertions = insertions
        self.tau_update_count += 1

        if writer is not None:
            writer.add_scalar(
                "ucdgt_adaptive_tau/temperature",
                self.ucdgt.tau,
                self.total_it
            )

            writer.add_scalar(
                "ucdgt_adaptive_tau/source_ess_fraction",
                self.tau_last_ess,
                self.total_it
            )

            writer.add_scalar(
                "ucdgt_adaptive_tau/update_count",
                self.tau_update_count,
                self.total_it
            )

        return True

    def train(
        self,
        src_replay_buffer,
        tar_replay_buffer,
        batch_size=128,
        writer=None,
    ):
        result = super().train(
            src_replay_buffer,
            tar_replay_buffer,
            batch_size,
            writer,
        )

        self._update_temperature(writer=writer)

        return result

