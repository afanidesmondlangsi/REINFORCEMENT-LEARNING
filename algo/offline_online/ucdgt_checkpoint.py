
import copy
import random

import numpy as np
import torch


def capture_rng():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def capture_policy(policy):
    core = policy.ucdgt

    return {
        "actor": copy.deepcopy(policy.policy.state_dict()),
        "critic": copy.deepcopy(policy.q_funcs.state_dict()),
        "target_critic": copy.deepcopy(
            policy.target_q_funcs.state_dict()
        ),
        "actor_optimizer": copy.deepcopy(
            policy.policy_optimizer.state_dict()
        ),
        "critic_optimizer": copy.deepcopy(
            policy.q_optimizer.state_dict()
        ),
        "log_alpha": policy.log_alpha.detach().clone(),
        "temp_optimizer": copy.deepcopy(
            policy.temp_optimizer.state_dict()
        ),
        "total_it": policy.total_it,
        "dynamics_models": [
            copy.deepcopy(model.state_dict())
            for model in core.models
        ],
        "dynamics_optimizers": [
            copy.deepcopy(optimizer.state_dict())
            for optimizer in core.optimizers
        ],
        "dynamics_state": {
            "delta_scale": core.delta_scale.detach().clone(),
            "mismatch_reference":
                core.mismatch_reference.detach().clone(),
            "uncertainty_reference":
                core.uncertainty_reference.detach().clone(),
            "normalization_ready": core.normalization_ready,
            "calibrated": core.calibrated,
        },
        "ucdgt_state": {
            name: copy.deepcopy(getattr(policy, name))
            for name in (
                "ucdgt_ready",
                "ucdgt_dynamics_updates",
                "ucdgt_last_update_size",
                "ucdgt_refresh_count",
                "ucdgt_calibration_indices",
                "ucdgt_calibration_data",
                "ucdgt_target_insertions",
                "ucdgt_last_observed_ptr",
                "ucdgt_last_observed_size",
            )
        },
    }


def restore_policy(policy, state):
    policy.policy.load_state_dict(state["actor"])
    policy.q_funcs.load_state_dict(state["critic"])
    policy.target_q_funcs.load_state_dict(
        state["target_critic"]
    )

    policy.policy_optimizer.load_state_dict(
        state["actor_optimizer"]
    )
    policy.q_optimizer.load_state_dict(
        state["critic_optimizer"]
    )

    with torch.no_grad():
        policy.log_alpha.copy_(
            state["log_alpha"].to(policy.device)
        )

    policy.temp_optimizer.load_state_dict(
        state["temp_optimizer"]
    )

    policy.total_it = state["total_it"]

    core = policy.ucdgt

    assert len(core.models) == len(
        state["dynamics_models"]
    )

    for model, saved in zip(
        core.models, state["dynamics_models"]
    ):
        model.load_state_dict(saved)

    for optimizer, saved in zip(
        core.optimizers, state["dynamics_optimizers"]
    ):
        optimizer.load_state_dict(saved)

    for name, value in state["dynamics_state"].items():
        if torch.is_tensor(value):
            value = value.to(core.device)
        setattr(core, name, copy.deepcopy(value))

    for name, value in state["ucdgt_state"].items():
        if name == "ucdgt_calibration_data" and value is not None:
            value = tuple(
                x.to(policy.device) if torch.is_tensor(x)
                else x
                for x in value
            )
        setattr(policy, name, copy.deepcopy(value))


def capture_buffer(buffer):
    size = int(buffer.size)

    return {
        "size": size,
        "ptr": int(buffer.ptr),
        "max_size": int(buffer.max_size),
        "state": buffer.state[:size].copy(),
        "action": buffer.action[:size].copy(),
        "next_state": buffer.next_state[:size].copy(),
        "reward": buffer.reward[:size].copy(),
        "not_done": buffer.not_done[:size].copy(),
    }


def restore_buffer(buffer, saved):
    size = saved["size"]

    if buffer.max_size != saved["max_size"]:
        raise ValueError("Replay-buffer capacity mismatch")

    for name in (
        "state",
        "action",
        "next_state",
        "reward",
        "not_done",
    ):
        getattr(buffer, name)[:size] = saved[name]

    buffer.size = size
    buffer.ptr = saved["ptr"]


def capture_environment(env):
    base = env.unwrapped

    result = {
        "sim_state": copy.deepcopy(base.sim.get_state()),
        "elapsed_steps": getattr(
            env, "_elapsed_steps", None
        ),
    }

    if hasattr(base, "np_random"):
        result["environment_rng"] = copy.deepcopy(
            base.np_random.get_state()
            if hasattr(base.np_random, "get_state")
            else base.np_random.bit_generator.state
        )

    return result


def restore_environment(env, saved):
    base = env.unwrapped

    base.sim.set_state(saved["sim_state"])
    base.sim.forward()

    if saved["elapsed_steps"] is not None:
        env._elapsed_steps = saved["elapsed_steps"]

    if "environment_rng" in saved:
        rng = base.np_random

        if hasattr(rng, "set_state"):
            rng.set_state(saved["environment_rng"])
        else:
            rng.bit_generator.state = saved["environment_rng"]
