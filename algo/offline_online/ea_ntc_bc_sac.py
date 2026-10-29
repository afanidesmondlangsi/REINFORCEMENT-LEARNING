
import numpy as np
import torch
import torch.nn.functional as F

from offline_online.bc_sac import BCSAC
from offline_online.ea_ntc.compatibility import weighted_bc_loss
from offline_online.ea_ntc.dynamics import TargetDynamicsEnsemble
from offline_online.ea_ntc.calibration import TargetErrorCalibrator


class EANTCBCSAC(BCSAC):
    """
    EA-NTC gate-only implementation.

    - Original BC-SAC critic update
    - Original SAC actor objective
    - Calibrated source behavioral-cloning weights
    - Target-only dynamics ensemble
    """

    def __init__(self, config, device):
        super().__init__(config, device)

        self.ea_enabled = bool(
            config.get("ea_gate_bc", True)
        )

        self.ea_warmup = int(
            config.get("ea_warmup_target_steps", 256)
        )

        self.ea_update_interval = int(
            config.get("ea_dynamics_update_interval", 30)
        )

        self.ea_calibration_interval = int(
            config.get("ea_calibration_update_interval", 30)
        )

        self.ea_batch_size = int(
            config.get("ea_dynamics_batch_size", 128)
        )

        if self.ea_update_interval <= 0:
            raise ValueError("Invalid dynamics update interval")

        if self.ea_calibration_interval <= 0:
            raise ValueError("Invalid calibration interval")

        if self.ea_batch_size <= 0:
            raise ValueError("Invalid dynamics batch size")

        self.dynamics = TargetDynamicsEnsemble(
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            device=device,
            ensemble_size=int(
                config.get("ea_ensemble_size", 3)
            ),
            hidden_size=int(
                config.get("ea_hidden_size", 256)
            ),
        )

        self.calibrator = TargetErrorCalibrator(
            gate_temperature=float(
                config.get("ea_gate_temperature", 1.0)
            )
        )

        self.ea_gate_ready = False
        self.ea_dynamics_updates = 0

        # Reliability gate: require enough dynamics training and
        # sufficiently low held-out target prediction error.
        self.ea_min_dynamics_updates = int(
            config.get("ea_min_dynamics_updates", 20)
        )
        self.ea_max_calibration_error = float(
            config.get("ea_max_calibration_error", 0.05)
        )
        if self.ea_min_dynamics_updates < 1:
            raise ValueError("ea_min_dynamics_updates must be positive")
        if not np.isfinite(self.ea_max_calibration_error) or self.ea_max_calibration_error <= 0:
            raise ValueError("ea_max_calibration_error must be finite and positive")
        self.ea_last_calibration_error = None

        self.ea_last_weights = None
        self.ea_last_uncertainty = None

    def _target_batch(self, replay_buffer, indices):
        """
        Fetch selected target transitions without changing
        the original ReplayBuffer implementation.
        """
        state = torch.as_tensor(
            replay_buffer.state[indices],
            dtype=torch.float32,
            device=self.device,
        )

        action = torch.as_tensor(
            replay_buffer.action[indices],
            dtype=torch.float32,
            device=self.device,
        )

        next_state = torch.as_tensor(
            replay_buffer.next_state[indices],
            dtype=torch.float32,
            device=self.device,
        )

        return state, action, next_state

    def update_dynamics_and_calibration(
        self, tar_replay_buffer
    ):
        """
        Update dynamics using even-indexed target data.
        Calibrate using odd-indexed target data.

        The gate is inactive until sufficient target
        transitions have been collected and calibration
        has succeeded.
        """
        if not self.ea_enabled:
            return

        size = int(tar_replay_buffer.size)

        if size < self.ea_warmup:
            return

        train_indices = np.arange(0, size, 2)
        calibration_indices = np.arange(1, size, 2)

        if (
            len(train_indices) < 2
            or len(calibration_indices) < 2
        ):
            return

        if (
            self.total_it % self.ea_update_interval == 0
            or self.ea_dynamics_updates == 0
        ):
            selected = np.random.choice(
                train_indices,
                size=self.ea_batch_size,
                replace=len(train_indices) < self.ea_batch_size,
            )

            state, action, next_state = (
                self._target_batch(
                    tar_replay_buffer, selected
                )
            )

            self.dynamics.train_step(
                state, action, next_state
            )

            self.ea_dynamics_updates += 1

        if (
            self.ea_dynamics_updates > 0
            and (
                self.total_it
                % self.ea_calibration_interval == 0
                or not self.ea_gate_ready
            )
        ):
            state, action, next_state = (
                self._target_batch(
                    tar_replay_buffer,
                    calibration_indices,
                )
            )

            errors, uncertainty = (
                self.dynamics.mismatch(
                    state, action, next_state
                )
            )

            self.calibrator.fit(errors)
            heldout_error = errors.mean().item()
            self.ea_last_calibration_error = heldout_error
            self.ea_gate_ready = (
                self.ea_dynamics_updates >= self.ea_min_dynamics_updates
                and np.isfinite(heldout_error)
                and heldout_error <= self.ea_max_calibration_error
            )

    @torch.no_grad()
    def source_weights(
        self,
        src_state,
        src_action,
        src_next_state,
    ):
        if not self.ea_enabled or not self.ea_gate_ready:
            return torch.ones(
                src_state.shape[0],
                device=src_state.device,
            )

        mismatch, uncertainty = self.dynamics.mismatch(
            src_state,
            src_action,
            src_next_state,
        )

        weights = self.calibrator.weights(mismatch)

        # Store diagnostics without affecting gradients.
        self.ea_last_uncertainty = (
            uncertainty.detach()
        )

        return weights

    def update_policy_and_temp(
        self,
        state_batch,
        src_state_batch,
        src_action_batch,
        source_weights=None,
    ):
        action_batch, logprobs_batch, _ = (
            self.policy(
                state_batch,
                get_logprob=True,
            )
        )

        q_b1, q_b2 = self.q_funcs(
            state_batch,
            action_batch,
        )

        qval_batch = torch.min(q_b1, q_b2)

        p_w = (
            self.config["weight"]
            / qval_batch.abs().mean().detach()
        )

        pred_src_act, _, _ = self.policy(
            src_state_batch,
            get_logprob=True,
        )

        if source_weights is None:
            source_weights = torch.ones(
                src_state_batch.shape[0],
                device=src_state_batch.device,
            )

        # Mean weighted per-transition BC error.
        # This allows overall BC influence to decrease
        # when source compatibility is low.
        per_transition_bc = (
            pred_src_act - src_action_batch
        ).pow(2).mean(dim=-1)

        bc_loss = (
            source_weights * per_transition_bc
        ).mean()

        policy_loss = (
            p_w
            * (
                self.alpha * logprobs_batch
                - qval_batch
            ).mean()
            + bc_loss
        )

        temp_loss = (
            -self.alpha
            * (
                logprobs_batch.detach()
                + self.target_entropy
            ).mean()
        )

        return policy_loss, temp_loss

    def train(
        self,
        src_replay_buffer,
        tar_replay_buffer,
        batch_size=128,
        writer=None,
    ):
        self.total_it += 1

        # Dynamics can start updating before the
        # original BC-SAC replay-size requirement.
        self.update_dynamics_and_calibration(
            tar_replay_buffer
        )

        if (
            src_replay_buffer.size < batch_size
            or tar_replay_buffer.size < batch_size
        ):
            return

        (
            src_state,
            src_action,
            src_next_state,
            src_reward,
            src_not_done,
        ) = src_replay_buffer.sample(batch_size)

        (
            tar_state,
            tar_action,
            tar_next_state,
            tar_reward,
            tar_not_done,
        ) = tar_replay_buffer.sample(batch_size)

        state = torch.cat(
            [src_state, tar_state], dim=0
        )
        action = torch.cat(
            [src_action, tar_action], dim=0
        )
        next_state = torch.cat(
            [src_next_state, tar_next_state], dim=0
        )
        reward = torch.cat(
            [src_reward, tar_reward], dim=0
        )
        not_done = torch.cat(
            [src_not_done, tar_not_done], dim=0
        )

        # Identical critic update to original BC-SAC.
        q_loss_step = self.update_q_functions(
            state,
            action,
            reward,
            next_state,
            not_done,
            writer,
        )

        self.q_optimizer.zero_grad()
        q_loss_step.backward()
        self.q_optimizer.step()

        self.update_target()

        weights = self.source_weights(
            src_state,
            src_action,
            src_next_state,
        )

        self.ea_last_weights = weights.detach()

        for p in self.q_funcs.parameters():
            p.requires_grad = False

        pi_loss_step, a_loss_step = (
            self.update_policy_and_temp(
                state,
                src_state,
                src_action,
                source_weights=weights,
            )
        )

        self.policy_optimizer.zero_grad()
        pi_loss_step.backward()
        self.policy_optimizer.step()

        if self.config["temperature_opt"]:
            self.temp_optimizer.zero_grad()
            a_loss_step.backward()
            self.temp_optimizer.step()

        for p in self.q_funcs.parameters():
            p.requires_grad = True

        if writer is not None and self.total_it % 100 == 0:
            writer.add_scalar(
                "ea_ntc/mean_source_weight",
                weights.mean().item(),
                self.total_it,
            )

            writer.add_scalar(
                "ea_ntc/gate_ready",
                float(self.ea_gate_ready),
                self.total_it,
            )

            writer.add_scalar(
                "ea_ntc/dynamics_updates",
                self.ea_dynamics_updates,
                self.total_it,
            )

            if self.ea_last_uncertainty is not None:
                writer.add_scalar(
                    "ea_ntc/mean_uncertainty",
                    self.ea_last_uncertainty.mean().item(),
                    self.total_it,
                )
