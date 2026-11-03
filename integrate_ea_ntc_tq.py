from pathlib import Path

path = Path("algo/offline_online/ea_ntc_bc_sac.py")
src = path.read_text(encoding="utf-8")

if "def _transfer_quality_probe(" in src:
    raise SystemExit("Transfer-quality probe already exists. No changes made.")

def replace_once(old, new):
    global src
    count = src.count(old)
    if count != 1:
        raise RuntimeError(
            "Expected one matching insertion point, found "
            + str(count) + ": " + old[:70]
        )
    src = src.replace(old, new, 1)

replace_once(
    "import numpy as np",
    "import copy\nimport math\nimport numpy as np"
)

replace_once(
    "from offline_online.ea_ntc.adaptive_gate import AdaptiveGateController",
    """from offline_online.ea_ntc.adaptive_gate import AdaptiveGateController
from offline_online.ea_ntc.transfer_quality_controller import (
    TransferQualityController,
    TransferQualityConfig,
)"""
)

replace_once(
    '        self.ea_bc_reduction = config.get(',
    '''        self.ea_transfer_quality_enabled = bool(
            config.get("ea_transfer_quality_enabled", False)
        )
        self.ea_tq_interval = int(config.get("ea_tq_interval", 100))
        self.ea_tq_batch_size = int(config.get("ea_tq_batch_size", 64))
        self.ea_tq_probe_lr = float(
            config.get("ea_tq_probe_lr", config["actor_lr"])
        )
        self.ea_tq_controller = TransferQualityController(
            TransferQualityConfig(
                max_strength=float(
                    config.get("ea_tq_max_strength", 0.5)
                ),
                error_low=float(
                    config.get("ea_adaptive_error_low", 0.35)
                ),
                error_high=float(
                    config.get("ea_adaptive_error_high", 0.75)
                ),
                uncertainty_penalty=float(
                    config.get("ea_tq_uncertainty_penalty", 1.0)
                ),
                margin=float(config.get("ea_tq_margin", 0.0)),
                temperature=float(
                    config.get("ea_tq_temperature", 0.1)
                ),
                ema_decay=float(
                    config.get("ea_tq_ema_decay", 0.9)
                ),
            )
        )
        self.ea_tq_strength = 0.0
        self.ea_tq_quality = None
        self.ea_tq_uncertainty = None

        if self.ea_transfer_quality_enabled:
            if not self.ea_enabled:
                raise ValueError("Transfer quality requires ea_gate_bc=true")
            if self.ea_tq_interval <= 0 or self.ea_tq_batch_size < 2:
                raise ValueError("Invalid transfer-quality probe schedule")
            if self.ea_tq_probe_lr <= 0:
                raise ValueError("Invalid transfer-quality probe learning rate")

        self.ea_bc_reduction = config.get('''
)

probe = '''
    def _transfer_quality_probe(
        self, tar_replay_buffer, src_state, src_action,
        raw_source_weights,
    ):
        """Training-only paired actor-update diagnostic.

        This is a one-step critic-based proxy, not measured policy return.
        Probe actors are discarded; the live actor and optimizer are unchanged.
        """
        if not self.ea_gate_ready:
            return

        size = int(tar_replay_buffer.size)
        count = min(self.ea_tq_batch_size, size)

        if count < 2:
            return

        # Use a fixed early target-state block, excluded from
        # dynamics-ensemble fitting by the existing calibration split.
        indices = np.arange(count)
        target_states, _, _ = self._target_batch(
            tar_replay_buffer, indices
        )

        # Independent temporary actors.
        source_actor = copy.deepcopy(self.policy)
        target_actor = copy.deepcopy(self.policy)

        source_actor.train()
        target_actor.train()

        # The same RNG state is used for corresponding stochastic
        # action samples. fork_rng restores the global RNG afterward.
        devices = []
        if torch.device(self.device).type == "cuda":
            idx = torch.device(self.device).index
            if idx is None:
                idx = torch.cuda.current_device()
            devices = [idx]

        def actor_objective(actor, include_source):
            sampled_action, logprob, _ = actor(
                target_states, get_logprob=True
            )
            q1, q2 = self.q_funcs(target_states, sampled_action)
            qvalue = torch.min(q1, q2)

            scale = (
                self.config["weight"]
                / qvalue.abs().mean().detach().clamp(min=1e-8)
            )
            loss = scale * (
                self.alpha.detach() * logprob - qvalue
            ).mean()

            if include_source:
                predicted, _, _ = actor(
                    src_state, get_logprob=True
                )
                bc = (predicted - src_action).pow(2).mean(dim=-1)

                if self.ea_bc_reduction == "normalized":
                    bc_loss = (
                        (raw_source_weights * bc).sum()
                        / raw_source_weights.sum().clamp(min=1e-8)
                    )
                else:
                    bc_loss = (raw_source_weights * bc).mean()

                loss = loss + bc_loss

            return loss

        def temporary_step(actor, include_source):
            loss = actor_objective(actor, include_source)
            parameters = tuple(actor.parameters())
            gradients = torch.autograd.grad(loss, parameters)
            with torch.no_grad():
                for parameter, gradient in zip(parameters, gradients):
                    parameter.add_(
                        gradient, alpha=-self.ea_tq_probe_lr
                    )

        # Critic parameters remain frozen, but Q derivatives with
        # respect to actor actions remain available.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(77123 + int(self.total_it))
            temporary_step(source_actor, True)

            torch.manual_seed(77123 + int(self.total_it))
            temporary_step(target_actor, False)

        with torch.no_grad():
            _, _, source_mean = source_actor(target_states)
            _, _, target_mean = target_actor(target_states)

            source_q1, source_q2 = self.q_funcs(
                target_states, source_mean
            )
            target_q1, target_q2 = self.q_funcs(
                target_states, target_mean
            )

            difference = (
                torch.min(source_q1, source_q2)
                - torch.min(target_q1, target_q2)
            ).reshape(-1)

            quality = difference.mean().item()
            uncertainty = (
                difference.std(unbiased=True).item()
                / math.sqrt(difference.numel())
            )

        if not (
            math.isfinite(quality)
            and math.isfinite(uncertainty)
        ):
            return

        self.ea_tq_quality = quality
        self.ea_tq_uncertainty = uncertainty
        self.ea_tq_strength = self.ea_tq_controller.update(
            relative_error=self.ea_last_relative_error,
            quality=quality,
            uncertainty=uncertainty,
        )

'''

replace_once(
    "    def update_policy_and_temp(",
    probe + "    def update_policy_and_temp("
)

replace_once(
    "        if self.ea_adaptive_gate:\n            strength =",
    "        if self.ea_transfer_quality_enabled:\n"
    "            strength = self.ea_tq_strength\n"
    "        elif self.ea_adaptive_gate:\n"
    "            strength ="
)

replace_once(
    "        weights = self.source_weights(\n",
    '''        if (
            self.ea_transfer_quality_enabled
            and self.ea_gate_ready
            and self.total_it % self.ea_tq_interval == 0
        ):
            # Probe the source weights under the previous gate setting.
            # This avoids a circular dependency on the new strength.
            probe_weights = self.source_weights(
                src_state, src_action, src_next_state
            )
            self._transfer_quality_probe(
                tar_replay_buffer,
                src_state,
                src_action,
                probe_weights.detach(),
            )

        weights = self.source_weights(
'''
)

replace_once(
    '            writer.add_scalar(\n                "ea_ntc/gate_ready",',
    '''            if self.ea_transfer_quality_enabled:
                writer.add_scalar(
                    "ea_ntc/transfer_gate_strength",
                    self.ea_tq_strength,
                    self.total_it,
                )
                if self.ea_tq_quality is not None:
                    writer.add_scalar(
                        "ea_ntc/transfer_quality",
                        self.ea_tq_quality,
                        self.total_it,
                    )
                if self.ea_tq_uncertainty is not None:
                    writer.add_scalar(
                        "ea_ntc/transfer_uncertainty",
                        self.ea_tq_uncertainty,
                        self.total_it,
                    )

            writer.add_scalar(
                "ea_ntc/gate_ready",'''
)

import ast
ast.parse(src)
path.write_text(src, encoding="utf-8")
print("EA-NTC transfer-quality integration written successfully.")
