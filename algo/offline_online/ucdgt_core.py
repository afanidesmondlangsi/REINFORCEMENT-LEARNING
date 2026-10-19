
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsModel(nn.Module):
    """Predict the target state change, next_state - state."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class UCDGTCore:
    """
    Uncertainty-Calibrated Dynamics-Gated Transfer core.

    Training procedure:
      1. Fit normalization using target training transitions.
      2. Train ensemble using target training transitions.
      3. Calibrate using held-out target transitions.
      4. Evaluate compatibility of source transitions.

    No source transitions are used to fit target dynamics.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        ensemble_size=3,
        learning_rate=1e-3,
        tau=10.0,
        hidden_dim=256,
        eps=1e-8,
    ):
        if ensemble_size < 2:
            raise ValueError("ensemble_size must be at least 2")

        if tau <= 0:
            raise ValueError("tau must be positive")

        self.device = torch.device(device)
        self.tau = float(tau)
        self.eps = eps

        self.models = [
            DynamicsModel(
                state_dim,
                action_dim,
                hidden_dim=hidden_dim,
            ).to(self.device)
            for _ in range(ensemble_size)
        ]

        self.optimizers = [
            torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
            )
            for model in self.models
        ]

        self.delta_scale = torch.ones(
            state_dim,
            device=self.device,
        )

        self.mismatch_reference = torch.tensor(
            1.0,
            device=self.device,
        )

        self.uncertainty_reference = torch.tensor(
            1.0,
            device=self.device,
        )

        self.normalization_ready = False
        self.calibrated = False

    def _prepare(self, state, action, next_state):
        state = state.to(self.device, dtype=torch.float32)
        action = action.to(self.device, dtype=torch.float32)
        next_state = next_state.to(
            self.device,
            dtype=torch.float32,
        )

        if state.ndim != 2 or action.ndim != 2:
            raise ValueError(
                "State and action must be 2D batches"
            )

        if next_state.shape != state.shape:
            raise ValueError(
                "next_state must match state shape"
            )

        if action.shape[0] != state.shape[0]:
            raise ValueError(
                "State and action batch sizes must match"
            )

        return state, action, next_state

    @torch.no_grad()
    def fit_normalization(
        self,
        state,
        action,
        next_state,
    ):
        """
        Fit target delta-state scaling BEFORE training.

        Call once on target training data only.
        """
        if self.normalization_ready:
            raise RuntimeError(
                "Normalization is already fitted"
            )

        state, action, next_state = self._prepare(
            state, action, next_state
        )

        if state.shape[0] < 2:
            raise ValueError(
                "At least two transitions are required"
            )

        delta = next_state - state

        self.delta_scale = (
            delta.std(dim=0, unbiased=False)
            .clamp_min(1e-3)
        )

        self.normalization_ready = True
        self.calibrated = False

    def train_dynamics(
        self,
        state,
        action,
        next_state,
    ):
        """
        Train each ensemble member on an independently
        resampled minibatch of target transitions.
        """

        if not self.normalization_ready:
            raise RuntimeError(
                "Call fit_normalization before training"
            )

        state, action, next_state = self._prepare(
            state, action, next_state
        )

        batch_size = state.shape[0]

        if batch_size == 0:
            raise ValueError("Empty training batch")

        delta = next_state - state
        losses = []

        for model, optimizer in zip(
            self.models,
            self.optimizers,
        ):
            # Independent bootstrap indices per member.
            indices = torch.randint(
                low=0,
                high=batch_size,
                size=(batch_size,),
                device=self.device,
            )

            prediction = model(
                state[indices],
                action[indices],
            )

            normalized_prediction = (
                prediction / self.delta_scale
            )

            normalized_target = (
                delta[indices] / self.delta_scale
            )

            loss = F.mse_loss(
                normalized_prediction,
                normalized_target,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().item())

        # Training changes the model; recalibration
        # is required before evaluating weights.
        self.calibrated = False

        return sum(losses) / len(losses)

    @torch.no_grad()
    def calculate_statistics(
        self,
        state,
        action,
        next_state,
    ):
        """
        Return:
          mismatch: per-transition prediction error
          disagreement: per-transition ensemble variance
        """

        if not self.normalization_ready:
            raise RuntimeError(
                "Normalization has not been fitted"
            )

        state, action, next_state = self._prepare(
            state, action, next_state
        )

        delta = next_state - state

        predictions = torch.stack(
            [
                model(state, action)
                for model in self.models
            ],
            dim=0,
        )

        normalized_errors = (
            predictions - delta.unsqueeze(0)
        ) / self.delta_scale

        mismatch = (
            normalized_errors.pow(2)
            .mean(dim=-1)
            .mean(dim=0)
        )

        mean_prediction = predictions.mean(dim=0)

        normalized_deviation = (
            predictions - mean_prediction.unsqueeze(0)
        ) / self.delta_scale

        disagreement = (
            normalized_deviation.pow(2)
            .mean(dim=-1)
            .mean(dim=0)
        )

        return mismatch, disagreement

    @torch.no_grad()
    def calibrate(
        self,
        state,
        action,
        next_state,
    ):
        """
        Calibrate using held-out target transitions.

        The caller must ensure these transitions were
        excluded from ensemble training.
        """

        mismatch, disagreement = (
            self.calculate_statistics(
                state, action, next_state
            )
        )

        if mismatch.numel() == 0:
            raise ValueError(
                "Calibration batch cannot be empty"
            )

        self.mismatch_reference = (
            mismatch.median().clamp_min(self.eps)
        )

        self.uncertainty_reference = (
            disagreement.median().clamp_min(self.eps)
        )

        self.calibrated = True

    @torch.no_grad()
    def calculate_weights(
        self,
        state,
        action,
        next_state,
        use_uncertainty=True,
    ):
        """
        Compute source compatibility weights in [0, 1].
        """

        if not self.calibrated:
            raise RuntimeError(
                "Calibrate before calculating weights"
            )

        mismatch, disagreement = (
            self.calculate_statistics(
                state, action, next_state
            )
        )

        relative_mismatch = (
            mismatch / self.mismatch_reference
        )

        if use_uncertainty:
            confidence = 1.0 / (
                1.0
                + disagreement
                / self.uncertainty_reference
            )
        else:
            confidence = torch.ones_like(
                relative_mismatch
            )

        penalty = (
            confidence
            * torch.relu(relative_mismatch - 1.0)
            / self.tau
        )

        return torch.exp(-penalty)
