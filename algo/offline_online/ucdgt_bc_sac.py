
import numpy as np
import torch
import torch.nn.functional as F

from .bc_sac import BCSAC
from .ucdgt_core import UCDGTCore


class UCDGTBCSAC(BCSAC):
    """
    BC-SAC with target-dynamics compatibility gating.

    The original BCSAC implementation remains unchanged.
    """

    def __init__(self, config, device, target_entropy=None):
        super().__init__(config, device, target_entropy)

        self.ucdgt_warmup = int(
            config.get("ucdgt_warmup", 256)
        )

        self.ucdgt_train_steps = int(
            config.get("ucdgt_train_steps", 300)
        )

        self.ucdgt_cal_fraction = float(
            config.get("ucdgt_cal_fraction", 0.2)
        )

        self.ucdgt_source_scale = float(
            config.get("ucdgt_source_scale", 1.0)
        )

        self.ucdgt_gate_actor = bool(
            config.get("ucdgt_gate_actor", True)
        )

        # Ablation controls; defaults preserve the full UCDGT algorithm.
        self.ucdgt_enable_refresh = config.get(
            "ucdgt_enable_refresh", True
        )
        self.ucdgt_enable_uncertainty = config.get(
            "ucdgt_enable_uncertainty", True
        )
        for name in ("ucdgt_enable_refresh", "ucdgt_enable_uncertainty"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean (true/false)")

        if self.ucdgt_warmup < 2:
            raise ValueError(
                "ucdgt_warmup must be at least 2"
            )

        if not 0 < self.ucdgt_cal_fraction < 1:
            raise ValueError(
                "ucdgt_cal_fraction must be between 0 and 1"
            )

        self.ucdgt = UCDGTCore(
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            device=device,
            ensemble_size=int(
                config.get("ucdgt_ensemble_size", 3)
            ),
            learning_rate=float(
                config.get("ucdgt_dynamics_lr", 1e-3)
            ),
            tau=float(
                config.get("ucdgt_gate_tau", 10.0)
            ),
        )

        self.ucdgt_ready = False
        self.ucdgt_dynamics_updates = 0

        # Periodic target-dynamics adaptation.
        self.ucdgt_update_interval = int(
            config.get("ucdgt_update_interval", 1000)
        )
        self.ucdgt_update_steps = int(
            config.get("ucdgt_update_steps", 50)
        )
        self.ucdgt_update_window = int(
            config.get("ucdgt_update_window", 2000)
        )

        if self.ucdgt_update_interval < 1:
            raise ValueError("ucdgt_update_interval must be positive")
        if self.ucdgt_update_steps < 1:
            raise ValueError("ucdgt_update_steps must be positive")
        if self.ucdgt_update_window < 1:
            raise ValueError("ucdgt_update_window must be positive")

        self.ucdgt_last_update_size = 0
        self.ucdgt_refresh_count = 0

        self.ucdgt_refresh_strategy = config.get(
            "ucdgt_refresh_strategy", "periodic"
        )
        if self.ucdgt_refresh_strategy not in ("periodic", "adaptive"):
            raise ValueError("Invalid UCDGT refresh strategy")

        self.ucdgt_adaptive_min_interval = int(
            config.get("ucdgt_adaptive_min_interval", 100)
        )
        self.ucdgt_adaptive_max_interval = int(
            config.get("ucdgt_adaptive_max_interval", 300)
        )
        self.ucdgt_adaptive_check_interval = int(
            config.get("ucdgt_adaptive_check_interval", 30)
        )
        self.ucdgt_adaptive_validation_size = int(
            config.get("ucdgt_adaptive_validation_size", 64)
        )
        self.ucdgt_adaptive_error_threshold = float(
            config.get("ucdgt_adaptive_error_threshold", 0.10)
        )

        if not (
            1 <= self.ucdgt_adaptive_check_interval
            <= self.ucdgt_adaptive_min_interval
            <= self.ucdgt_adaptive_max_interval
        ):
            raise ValueError("Invalid adaptive refresh intervals")

        if self.ucdgt_adaptive_validation_size < 1:
            raise ValueError("Validation size must be positive")

        if self.ucdgt_adaptive_error_threshold < 0:
            raise ValueError("Error threshold must be nonnegative")

        self.ucdgt_match_validation_exclusion = bool(
            config.get("ucdgt_match_validation_exclusion", False)
        )

        self.ucdgt_refresh_subset_mode = config.get(
            "ucdgt_refresh_subset_mode", "all"
        )
        self.ucdgt_refresh_subset_size = int(
            config.get("ucdgt_refresh_subset_size", 64)
        )

        if self.ucdgt_refresh_subset_mode not in ("all", "random"):
            raise ValueError("Invalid UCDGT refresh subset mode")

        if self.ucdgt_refresh_subset_size < 1:
            raise ValueError("UCDGT refresh subset size must be positive")

        self.ucdgt_subset_rng = np.random.RandomState(
            int(config.get("seed", 0)) + 73021
        )

        self.ucdgt_adaptive_last_check = 0
        self.ucdgt_adaptive_excluded_indices = None

        # Reserve the initial calibration set for future refreshes.
        self.ucdgt_calibration_indices = None
        self.ucdgt_calibration_data = None

        # Monotonic insertion count, independent of replay-buffer size.
        self.ucdgt_target_insertions = 0
        self.ucdgt_last_observed_ptr = None
        self.ucdgt_last_observed_size = None

    def _initialize_ucdgt(self, tar_replay_buffer):
        """
        Initialize dynamics from a fixed snapshot of
        the initial target replay-buffer transitions.
        """

        if self.ucdgt_ready:
            return

        if tar_replay_buffer.size < self.ucdgt_warmup:
            return

        n = self.ucdgt_warmup

        indices = np.random.permutation(n)

        n_cal = max(
            1,
            int(n * self.ucdgt_cal_fraction),
        )

        train_indices = indices[n_cal:]
        cal_indices = indices[:n_cal]

        # Never include these initial indices in dynamics training.
        self.ucdgt_calibration_indices = cal_indices.copy()

        def tensor(data, idx):
            return torch.as_tensor(
                data[idx],
                dtype=torch.float32,
                device=self.device,
            )

        train_state = tensor(
            tar_replay_buffer.state,
            train_indices,
        )
        train_action = tensor(
            tar_replay_buffer.action,
            train_indices,
        )
        train_next_state = tensor(
            tar_replay_buffer.next_state,
            train_indices,
        )

        cal_state = tensor(
            tar_replay_buffer.state,
            cal_indices,
        )
        cal_action = tensor(
            tar_replay_buffer.action,
            cal_indices,
        )
        cal_next_state = tensor(
            tar_replay_buffer.next_state,
            cal_indices,
        )

        self.ucdgt.fit_normalization(
            train_state,
            train_action,
            train_next_state,
        )

        n_train = train_state.shape[0]
        batch_size = min(128, n_train)

        for _ in range(self.ucdgt_train_steps):
            sampled = torch.randint(
                0,
                n_train,
                (batch_size,),
                device=self.device,
            )

            self.ucdgt.train_dynamics(
                train_state[sampled],
                train_action[sampled],
                train_next_state[sampled],
            )

            self.ucdgt_dynamics_updates += 1

        # Snapshot held-out transitions before replay-buffer wraparound.
        self.ucdgt_calibration_data = (
            cal_state.detach().clone(),
            cal_action.detach().clone(),
            cal_next_state.detach().clone(),
        )

        self.ucdgt.calibrate(
            cal_state,
            cal_action,
            cal_next_state,
        )

        self.ucdgt_ready = True
        self.ucdgt_last_update_size = tar_replay_buffer.size
        self.ucdgt_target_insertions = tar_replay_buffer.size
        self.ucdgt_last_observed_ptr = tar_replay_buffer.ptr
        self.ucdgt_last_observed_size = tar_replay_buffer.size

        print(
            "UCDGT initialized | "
            f"training transitions={n_train} | "
            f"calibration transitions={n_cal} | "
            f"dynamics updates="
            f"{self.ucdgt_dynamics_updates}"
        )

    def _ucdgt_refresh_indices(self, tar_replay_buffer):
        """Eligible recent transitions, excluding still-present calibration data.

        Original calibration transitions are identified by their insertion slots.
        A reserved slot becomes eligible once its original transition is overwritten.
        """
        size = int(tar_replay_buffer.size)
        capacity = int(tar_replay_buffer.max_size)
        ptr = int(tar_replay_buffer.ptr)
        if capacity < 1 or not 0 <= size <= capacity or not 0 <= ptr < capacity:
            raise ValueError("Invalid target replay-buffer size/capacity/pointer")

        window = min(size, self.ucdgt_update_window)
        indices = (ptr - 1 - np.arange(window, dtype=np.int64)) % capacity
        reserved = self.ucdgt_calibration_indices
        if reserved is not None and len(reserved):
            reserved = np.asarray(reserved, dtype=np.int64)
            # For an initially empty circular buffer, slot j is overwritten
            # on insertion number capacity + j + 1.
            still_present = self.ucdgt_target_insertions <= capacity + reserved
            indices = indices[~np.isin(indices, reserved[still_present])]
        if (
            (
                getattr(self, "ucdgt_refresh_strategy", "periodic") == "adaptive"
                or getattr(
                    self, "ucdgt_match_validation_exclusion", False
                )
            )
            and self.ucdgt_adaptive_excluded_indices is not None
        ):
            indices = indices[
                ~np.isin(
                    indices,
                    self.ucdgt_adaptive_excluded_indices
                )
            ]

        return indices

    def _ucdgt_select_refresh_subset(self, indices):
        """Select refresh-training transitions without replacement."""
        indices = np.asarray(indices, dtype=np.int64)

        if getattr(self, "ucdgt_refresh_subset_mode", "all") != "random":
            return indices

        n = min(self.ucdgt_refresh_subset_size, len(indices))

        if n == len(indices):
            return indices

        positions = self.ucdgt_subset_rng.choice(
            len(indices), size=n, replace=False
        )

        return indices[positions]

    def _ucdgt_should_adapt(self, tar_replay_buffer, writer=None):
        since_refresh = (
            self.ucdgt_target_insertions
            - self.ucdgt_last_update_size
        )

        if since_refresh < self.ucdgt_adaptive_min_interval:
            return False

        if (
            self.ucdgt_target_insertions
            - self.ucdgt_adaptive_last_check
            < self.ucdgt_adaptive_check_interval
            and since_refresh < self.ucdgt_adaptive_max_interval
        ):
            return False

        self.ucdgt_adaptive_last_check = self.ucdgt_target_insertions

        size = int(tar_replay_buffer.size)
        capacity = int(tar_replay_buffer.max_size)
        ptr = int(tar_replay_buffer.ptr)

        n = min(
            self.ucdgt_adaptive_validation_size,
            size,
            since_refresh
        )

        if n < 1:
            return False

        recent = (
            ptr - 1 - np.arange(n, dtype=np.int64)
        ) % capacity

        self.ucdgt_adaptive_excluded_indices = recent.copy()

        def tensor(array):
            return torch.as_tensor(
                array[recent],
                dtype=torch.float32,
                device=self.device
            )

        with torch.no_grad():
            recent_mismatch, _ = self.ucdgt.calculate_statistics(
                tensor(tar_replay_buffer.state),
                tensor(tar_replay_buffer.action),
                tensor(tar_replay_buffer.next_state)
            )

            calibration_mismatch, _ = (
                self.ucdgt.calculate_statistics(
                    *self.ucdgt_calibration_data
                )
            )

            recent_error = float(
                recent_mismatch.mean().item()
            )
            calibration_error = float(
                calibration_mismatch.mean().item()
            )

        ratio = recent_error / max(calibration_error, 1e-8)

        forced = (
            since_refresh >= self.ucdgt_adaptive_max_interval
        )

        triggered = (
            ratio > 1.0 + self.ucdgt_adaptive_error_threshold
        )

        refresh = triggered or forced

        if writer is not None:
            writer.add_scalar(
                "ucdgt/adaptive_recent_mismatch",
                recent_error,
                self.total_it
            )
            writer.add_scalar(
                "ucdgt/adaptive_calibration_mismatch",
                calibration_error,
                self.total_it
            )
            writer.add_scalar(
                "ucdgt/adaptive_error_ratio",
                ratio,
                self.total_it
            )
            writer.add_scalar(
                "ucdgt/adaptive_trigger",
                int(refresh),
                self.total_it
            )

        print(
            "UCDGT adaptive check | "
            f"target transitions={self.ucdgt_target_insertions} | "
            f"ratio={ratio:.4f} | "
            f"forced={forced} | "
            f"refresh={refresh}"
        )

        return refresh

    def _refresh_ucdgt(self, tar_replay_buffer, writer=None):
        """Update the existing ensemble after enough NEW target insertions.

        Assumes fewer than max_size insertions between calls (the current
        offline-online train loop inserts at most one per call).
        """
        if not self.ucdgt_ready:
            return False

        size = int(tar_replay_buffer.size)
        capacity = int(tar_replay_buffer.max_size)
        ptr = int(tar_replay_buffer.ptr)
        if capacity < 1 or not 0 <= ptr < capacity:
            raise ValueError("Invalid target replay-buffer capacity/pointer")

        if self.ucdgt_last_observed_ptr is None:
            raise RuntimeError("UCDGT insertion tracking was not initialized")

        previous_size = self.ucdgt_last_observed_size
        previous_ptr = self.ucdgt_last_observed_ptr
        if size < previous_size and previous_size < capacity:
            raise RuntimeError("Target replay-buffer size unexpectedly decreased")

        if previous_size < capacity and size < capacity:
            inserted = size - previous_size
        else:
            inserted = (ptr - previous_ptr) % capacity
            if previous_size < capacity:
                # Transition from partially filled to full buffer.
                inserted = size - previous_size

        if inserted < 0:
            raise RuntimeError("Target replay-buffer insertion count decreased")

        self.ucdgt_target_insertions += inserted
        self.ucdgt_last_observed_ptr = ptr
        self.ucdgt_last_observed_size = size

        if getattr(self, "ucdgt_refresh_strategy", "periodic") == "adaptive":
            if not self._ucdgt_should_adapt(
                tar_replay_buffer, writer
            ):
                return False
        elif (
            self.ucdgt_target_insertions
            - self.ucdgt_last_update_size
            < self.ucdgt_update_interval
        ):
            return False

        # Matched periodic ablation: exclude the same number of
        # recent validation transitions used by adaptive refresh.
        if (
            getattr(self, "ucdgt_refresh_strategy", "periodic") == "periodic"
            and getattr(
                self, "ucdgt_match_validation_exclusion", False
            )
        ):
            n_validation = min(
                self.ucdgt_adaptive_validation_size,
                size,
                self.ucdgt_target_insertions
                - self.ucdgt_last_update_size,
            )

            self.ucdgt_adaptive_excluded_indices = (
                ptr - 1 - np.arange(
                    n_validation, dtype=np.int64
                )
            ) % capacity

        indices = self._ucdgt_refresh_indices(tar_replay_buffer)
        indices = self._ucdgt_select_refresh_subset(indices)

        if len(indices) == 0:
            return False

        def tensor(data):
            return torch.as_tensor(
                data[indices], dtype=torch.float32, device=self.device
            )

        train_state = tensor(tar_replay_buffer.state)
        train_action = tensor(tar_replay_buffer.action)
        train_next_state = tensor(tar_replay_buffer.next_state)
        n_train = len(indices)
        batch_size = min(128, n_train)

        # Retain fixed normalization and existing ensemble/optimizer state.
        for _ in range(self.ucdgt_update_steps):
            sampled = torch.randint(
                0, n_train, (batch_size,), device=self.device
            )
            self.ucdgt.train_dynamics(
                train_state[sampled],
                train_action[sampled],
                train_next_state[sampled],
            )
            self.ucdgt_dynamics_updates += 1

        # train_dynamics invalidates calibration; restore it before gating.
        self.ucdgt.calibrate(*self.ucdgt_calibration_data)
        self.ucdgt_last_update_size = self.ucdgt_target_insertions

        if getattr(self, "ucdgt_refresh_strategy", "periodic") == "adaptive":
            self.ucdgt_adaptive_last_check = (
                self.ucdgt_target_insertions
            )
            self.ucdgt_adaptive_excluded_indices = None

        if getattr(
            self, "ucdgt_match_validation_exclusion", False
        ):
            self.ucdgt_adaptive_excluded_indices = None

        self.ucdgt_refresh_count += 1

        if writer is not None:
            writer.add_scalar(
                "ucdgt/refresh_count", self.ucdgt_refresh_count, self.total_it
            )
            writer.add_scalar(
                "ucdgt/dynamics_updates", self.ucdgt_dynamics_updates,
                self.total_it,
            )
            writer.add_scalar(
                "ucdgt/target_insertions", self.ucdgt_target_insertions,
                self.total_it,
            )

        print(
            "UCDGT refreshed | "
            f"target transitions={self.ucdgt_target_insertions} | "
            f"training transitions={n_train} | "
            f"refresh count={self.ucdgt_refresh_count} | "
            f"dynamics updates={self.ucdgt_dynamics_updates}"
        )
        return True

    def _source_weights(
        self,
        state,
        action,
        next_state,
    ):
        """
        Calculate conservative compatibility weights.

        The mismatch gate is combined with an
        uncertainty penalty, rather than allowing
        high uncertainty to increase source trust.
        """

        with torch.no_grad():
            mismatch, disagreement = (
                self.ucdgt.calculate_statistics(
                    state,
                    action,
                    next_state,
                )
            )

            relative_mismatch = (
                mismatch
                / self.ucdgt.mismatch_reference
            )

            relative_uncertainty = (
                disagreement
                / self.ucdgt.uncertainty_reference
            )

            mismatch_penalty = torch.relu(
                relative_mismatch - 1.0
            )

            if self.ucdgt_enable_uncertainty:
                uncertainty_penalty = torch.relu(
                    relative_uncertainty - 1.0
                )
            else:
                uncertainty_penalty = torch.zeros_like(mismatch_penalty)

            penalty = (
                mismatch_penalty
                + uncertainty_penalty
            ) / self.ucdgt.tau

            weights = torch.exp(-penalty)

            return weights.unsqueeze(-1)

    def _update_weighted_q(
        self,
        src_state,
        src_action,
        src_next_state,
        src_reward,
        src_not_done,
        tar_state,
        tar_action,
        tar_next_state,
        tar_reward,
        tar_not_done,
        src_weights,
    ):
        state = torch.cat(
            [src_state, tar_state], dim=0
        )
        action = torch.cat(
            [src_action, tar_action], dim=0
        )
        next_state = torch.cat(
            [src_next_state, tar_next_state],
            dim=0,
        )
        reward = torch.cat(
            [src_reward, tar_reward], dim=0
        )
        not_done = torch.cat(
            [src_not_done, tar_not_done], dim=0
        )

        with torch.no_grad():
            next_action, next_logprob, _ = (
                self.policy(
                    next_state,
                    get_logprob=True,
                )
            )

            target_q1, target_q2 = (
                self.target_q_funcs(
                    next_state,
                    next_action,
                )
            )

            target_q = torch.minimum(
                target_q1,
                target_q2,
            )

            value_target = (
                reward
                + not_done
                * self.discount
                * (
                    target_q
                    - self.alpha * next_logprob
                )
            )

        q1, q2 = self.q_funcs(
            state,
            action,
        )

        per_transition_loss = (
            (q1 - value_target).pow(2)
            + (q2 - value_target).pow(2)
        )

        target_weights = torch.ones(
            (tar_state.shape[0], 1),
            device=self.device,
        )

        combined_weights = torch.cat(
            [
                self.ucdgt_source_scale
                * src_weights,
                target_weights,
            ],
            dim=0,
        )

        # Fixed denominator preserves the original
        # BC-SAC critic-loss scale when all weights=1.
        return (
            combined_weights
            * per_transition_loss
        ).mean()

    def train(
        self,
        src_replay_buffer,
        tar_replay_buffer,
        batch_size=128,
        writer=None,
    ):
        self.total_it += 1

        if tar_replay_buffer.size < batch_size:
            return

        self._initialize_ucdgt(tar_replay_buffer)
        if self.ucdgt_ready and self.ucdgt_enable_refresh:
            self._refresh_ucdgt(tar_replay_buffer, writer)

        tar_state, tar_action, tar_next_state, \
            tar_reward, tar_not_done = (
                tar_replay_buffer.sample(batch_size)
            )

        if not self.ucdgt_ready:
            # Target-only warm-up.
            q_loss = self.update_q_functions(
                tar_state,
                tar_action,
                tar_reward,
                tar_next_state,
                tar_not_done,
                writer,
            )

            state = tar_state
            src_state = None
            src_action = None
            src_weights = None

        else:
            if src_replay_buffer.size < batch_size:
                return

            src_state, src_action, src_next_state, \
                src_reward, src_not_done = (
                    src_replay_buffer.sample(batch_size)
                )

            src_weights = self._source_weights(
                src_state,
                src_action,
                src_next_state,
            )

            q_loss = self._update_weighted_q(
                src_state,
                src_action,
                src_next_state,
                src_reward,
                src_not_done,
                tar_state,
                tar_action,
                tar_next_state,
                tar_reward,
                tar_not_done,
                src_weights,
            )

            state = torch.cat(
                [src_state, tar_state],
                dim=0,
            )

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        self.update_target()

        for parameter in self.q_funcs.parameters():
            parameter.requires_grad = False

        if not self.ucdgt_ready:
            action, logprob, _ = self.policy(
                state,
                get_logprob=True,
            )

            q1, q2 = self.q_funcs(
                state,
                action,
            )

            q_value = torch.minimum(q1, q2)

            policy_loss = (
                self.alpha * logprob - q_value
            ).mean()

            temp_loss = (
                -self.alpha
                * (
                    logprob.detach()
                    + self.target_entropy
                ).mean()
            )

        else:
            action, logprob, _ = self.policy(
                state,
                get_logprob=True,
            )

            q1, q2 = self.q_funcs(
                state,
                action,
            )

            q_value = torch.minimum(q1, q2)

            p_w = (
                self.config["weight"]
                / q_value.abs().mean().detach().clamp_min(1e-8)
            )

            pred_src_action, _, _ = self.policy(
                src_state,
                get_logprob=True,
            )

            bc_error = (
                pred_src_action - src_action
            ).pow(2).mean(dim=-1, keepdim=True)

            if self.ucdgt_gate_actor:
                bc_loss = (
                    src_weights * bc_error
                ).mean()
            else:
                bc_loss = bc_error.mean()

            policy_loss = (
                p_w
                * (
                    self.alpha * logprob - q_value
                ).mean()
                + bc_loss
            )

            temp_loss = (
                -self.alpha
                * (
                    logprob.detach()
                    + self.target_entropy
                ).mean()
            )

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.config["temperature_opt"]:
            self.temp_optimizer.zero_grad()
            temp_loss.backward()
            self.temp_optimizer.step()

        for parameter in self.q_funcs.parameters():
            parameter.requires_grad = True

        if (
            writer is not None
            and self.ucdgt_ready
            and self.total_it % 100 == 0
        ):
            writer.add_scalar(
                "ucdgt/source_weight_mean",
                src_weights.mean().item(),
                self.total_it,
            )
            writer.add_scalar(
                "ucdgt/source_weight_min",
                src_weights.min().item(),
                self.total_it,
            )
            writer.add_scalar(
                "ucdgt/source_weight_max",
                src_weights.max().item(),
                self.total_it,
            )
            with torch.no_grad():
                w = src_weights.detach().float().reshape(-1)
                ess = w.sum().square() / w.square().sum().clamp_min(1e-12)
                writer.add_scalar("ucdgt/source_weight_std", w.std(unbiased=False).item(), self.total_it)
                writer.add_scalar("ucdgt/source_weight_ess", ess.item(), self.total_it)
                writer.add_scalar("ucdgt/source_weight_ess_fraction", (ess / w.numel()).item(), self.total_it)
                writer.add_scalar("ucdgt/source_weight_p10", torch.quantile(w, 0.10).item(), self.total_it)
                writer.add_scalar("ucdgt/source_weight_p50", torch.quantile(w, 0.50).item(), self.total_it)
                writer.add_scalar("ucdgt/source_weight_p90", torch.quantile(w, 0.90).item(), self.total_it)
                mismatch, disagreement = self.ucdgt.calculate_statistics(
                    src_state, src_action, src_next_state
                )
                writer.add_scalar("ucdgt/source_mismatch_mean", mismatch.float().mean().item(), self.total_it)
                writer.add_scalar("ucdgt/source_uncertainty_mean", disagreement.float().mean().item(), self.total_it)
