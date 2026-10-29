
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsNetwork(nn.Module):
    """
    Predicts target-environment state change:

        delta_s = next_state - state
    """

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim),
        )

    def forward(self, state, action):
        inputs = torch.cat([state, action], dim=-1)
        return self.network(inputs)


class TargetDynamicsEnsemble:
    """
    Ensemble of independently initialized target-dynamics
    models.

    Uses target transitions only for training.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        device="cpu",
        ensemble_size=3,
        hidden_size=256,
        learning_rate=3e-4,
    ):
        self.device = torch.device(device)
        self.ensemble_size = ensemble_size

        self.models = [
            DynamicsNetwork(
                state_dim,
                action_dim,
                hidden_size,
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

        self.update_count = 0

    def train_step(self, state, action, next_state):
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)

        target_delta = next_state - state

        losses = []

        for model, optimizer in zip(
            self.models, self.optimizers
        ):
            predicted_delta = model(state, action)

            loss = F.mse_loss(
                predicted_delta,
                target_delta,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        self.update_count += 1

        return sum(losses) / len(losses)

    @torch.no_grad()
    def predict(self, state, action):
        state = state.to(self.device)
        action = action.to(self.device)

        predictions = torch.stack(
            [
                model(state, action)
                for model in self.models
            ],
            dim=0,
        )

        mean_prediction = predictions.mean(dim=0)

        uncertainty = predictions.var(
            dim=0,
            unbiased=False,
        ).mean(dim=-1)

        return mean_prediction, uncertainty

    @torch.no_grad()
    def mismatch(self, state, action, next_state):
        state = state.to(self.device)
        next_state = next_state.to(self.device)

        observed_delta = next_state - state

        predicted_delta, uncertainty = self.predict(
            state, action
        )

        mismatch = (
            predicted_delta - observed_delta
        ).pow(2).mean(dim=-1)

        return mismatch, uncertainty
