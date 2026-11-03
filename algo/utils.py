
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn

from typing import Callable, Optional
from torch.nn.modules.dropout import Dropout
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence


# ============================================================
# MEMORY-MAPPED REPLAY BUFFER
# ============================================================

class ReplayBuffer(object):

    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        max_size=int(1e6),
        storage_dir=None
    ):

        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.device = device

        # Create unique disk-backed storage.
        # Separate buffers must never share the same files.
        if storage_dir is None:
            self.storage_dir = tempfile.mkdtemp(
                prefix="offdynamics_replay_"
            )
        else:
            self.storage_dir = os.path.abspath(storage_dir)
            os.makedirs(self.storage_dir, exist_ok=True)

        self.state = self._create_array(
            "state.dat",
            (self.max_size, state_dim)
        )

        self.action = self._create_array(
            "action.dat",
            (self.max_size, action_dim)
        )

        self.next_state = self._create_array(
            "next_state.dat",
            (self.max_size, state_dim)
        )

        self.reward = self._create_array(
            "reward.dat",
            (self.max_size, 1)
        )

        self.not_done = self._create_array(
            "not_done.dat",
            (self.max_size, 1)
        )

    def _create_array(self, filename, shape):

        path = os.path.join(
            self.storage_dir,
            filename
        )

        return np.memmap(
            path,
            dtype=np.float32,
            mode="w+",
            shape=shape
        )

    # --------------------------------------------------------
    # ADD ONLINE TRANSITION
    # --------------------------------------------------------

    def add(
        self,
        state,
        action,
        next_state,
        reward,
        done
    ):

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state

        self.reward[self.ptr] = reward

        self.not_done[self.ptr] = 1.0 - float(done)

        self.ptr = (self.ptr + 1) % self.max_size

        self.size = min(
            self.size + 1,
            self.max_size
        )

    # --------------------------------------------------------
    # SAMPLE MINIBATCH
    # --------------------------------------------------------

    def sample(self, batch_size):

        if self.size < batch_size:
            raise ValueError(
                f"Replay buffer contains {self.size} transitions. "
                f"Requested batch size: {batch_size}."
            )

        ind = np.random.randint(
            0,
            self.size,
            size=batch_size
        )

        return (
            torch.as_tensor(
                np.asarray(self.state[ind]),
                dtype=torch.float32
            ).to(self.device),

            torch.as_tensor(
                np.asarray(self.action[ind]),
                dtype=torch.float32
            ).to(self.device),

            torch.as_tensor(
                np.asarray(self.next_state[ind]),
                dtype=torch.float32
            ).to(self.device),

            torch.as_tensor(
                np.asarray(self.reward[ind]),
                dtype=torch.float32
            ).to(self.device),

            torch.as_tensor(
                np.asarray(self.not_done[ind]),
                dtype=torch.float32
            ).to(self.device)
        )

    # --------------------------------------------------------
    # CONVERT AN ALREADY-LOADED D4RL DATASET
    # --------------------------------------------------------

    def convert_D4RL(
        self,
        dataset,
        chunk_size=8192
    ):

        n = len(dataset["rewards"])

        if n > self.max_size:
            raise ValueError(
                f"Dataset size {n} exceeds buffer capacity "
                f"{self.max_size}."
            )

        self.size = n
        self.ptr = n % self.max_size

        for start in range(0, n, chunk_size):

            end = min(
                start + chunk_size,
                n
            )

            self.state[start:end] = (
                dataset["observations"][start:end]
            )

            self.action[start:end] = (
                dataset["actions"][start:end]
            )

            self.next_state[start:end] = (
                dataset["next_observations"][start:end]
            )

            self.reward[start:end, 0] = (
                dataset["rewards"][start:end]
            )

            self.not_done[start:end, 0] = (
                1.0 - dataset["terminals"][start:end]
            )

        self.flush()

    # --------------------------------------------------------
    # STREAM D4RL HDF5 DATASET DIRECTLY TO DISK
    # --------------------------------------------------------

    def convert_D4RL_hdf5(
        self,
        env,
        chunk_size=8192,
        terminate_on_end=False
    ):

        import h5py
        import d4rl

        base_env = env.unwrapped

        dataset_url = getattr(
            base_env,
            "dataset_url",
            None
        )

        if dataset_url is None:

            dataset_url = getattr(
                base_env,
                "_dataset_url",
                None
            )

        if dataset_url is None:

            dataset_url = getattr(
                env,
                "dataset_url",
                None
            )

        if dataset_url is None:

            raise ValueError(
                "Could not locate the D4RL dataset URL. "
                "Check the environment's dataset configuration."
            )

        h5path = d4rl.offline_env.download_dataset_from_url(
            dataset_url
        )

        print("\n========================================")
        print("D4RL MEMORY-MAPPED DATASET LOADING")
        print("========================================")
        print("Environment:", getattr(env, "spec", None))
        print("Dataset:", h5path)

        with h5py.File(h5path, "r") as dataset:

            required_keys = [
                "observations",
                "actions",
                "rewards",
                "terminals"
            ]

            for key in required_keys:

                if key not in dataset:

                    raise ValueError(
                        f"D4RL dataset missing key: {key}"
                    )

            observations = dataset["observations"]
            actions = dataset["actions"]
            rewards = dataset["rewards"]
            terminals = dataset["terminals"]

            n = len(rewards)

            if n == 0:
                raise ValueError("Empty D4RL dataset.")

            if "timeouts" in dataset:

                timeouts = dataset["timeouts"]

            else:

                timeouts = None

            max_episode_steps = getattr(
                base_env,
                "_max_episode_steps",
                None
            )

            if max_episode_steps is None:

                max_episode_steps = getattr(
                    env,
                    "_max_episode_steps",
                    None
                )

            # =================================================
            # PASS 1: COUNT VALID TRANSITIONS
            # =================================================

            valid_count = 0
            episode_step = 0

            for start in range(
                0,
                n - 1,
                chunk_size
            ):

                end = min(
                    start + chunk_size,
                    n - 1
                )

                if timeouts is not None:

                    timeout_chunk = np.asarray(
                        timeouts[start:end],
                        dtype=bool
                    ).reshape(-1)

                    if not terminate_on_end:

                        valid_count += int(
                            np.count_nonzero(
                                ~timeout_chunk
                            )
                        )

                    else:

                        valid_count += len(timeout_chunk)

                else:

                    if max_episode_steps is None:

                        raise ValueError(
                            "Cannot reconstruct D4RL episode "
                            "timeouts because max_episode_steps "
                            "is unavailable."
                        )

                    terminal_chunk = np.asarray(
                        terminals[start:end],
                        dtype=bool
                    ).reshape(-1)

                    for done_bool in terminal_chunk:

                        final_timestep = (
                            episode_step ==
                            max_episode_steps - 1
                        )

                        if (
                            not terminate_on_end
                            and final_timestep
                        ):

                            episode_step = 0
                            continue

                        if done_bool or final_timestep:

                            episode_step = 0

                        valid_count += 1
                        episode_step += 1

            if valid_count > self.max_size:

                raise ValueError(
                    f"Dataset requires {valid_count} slots, "
                    f"but capacity is {self.max_size}."
                )

            # =================================================
            # PASS 2: STREAM VALID TRANSITIONS
            # =================================================

            write_position = 0
            episode_step = 0

            for start in range(
                0,
                n - 1,
                chunk_size
            ):

                end = min(
                    start + chunk_size,
                    n - 1
                )

                indices = np.arange(
                    start,
                    end
                )

                if timeouts is not None:

                    timeout_chunk = np.asarray(
                        timeouts[start:end],
                        dtype=bool
                    ).reshape(-1)

                    if terminate_on_end:

                        valid_indices = indices

                    else:

                        valid_indices = indices[
                            ~timeout_chunk
                        ]

                    count = len(valid_indices)

                    if count == 0:
                        continue

                    destination = slice(
                        write_position,
                        write_position + count
                    )

                    # Original D4RL behavior:
                    # next observation = observations[i+1]

                    self.state[destination] = (
                        observations[valid_indices]
                    )

                    self.action[destination] = (
                        actions[valid_indices]
                    )

                    self.next_state[destination] = (
                        observations[valid_indices + 1]
                    )

                    self.reward[destination, 0] = (
                        rewards[valid_indices]
                    )

                    self.not_done[destination, 0] = (
                        1.0 - terminals[valid_indices]
                    )

                    write_position += count

                else:

                    terminal_chunk = np.asarray(
                        terminals[start:end],
                        dtype=bool
                    ).reshape(-1)

                    for offset, i in enumerate(indices):

                        done_bool = bool(
                            terminal_chunk[offset]
                        )

                        final_timestep = (
                            episode_step ==
                            max_episode_steps - 1
                        )

                        if (
                            not terminate_on_end
                            and final_timestep
                        ):

                            episode_step = 0
                            continue

                        if done_bool or final_timestep:

                            episode_step = 0

                        self.state[write_position] = (
                            observations[i]
                        )

                        self.action[write_position] = (
                            actions[i]
                        )

                        self.next_state[write_position] = (
                            observations[i + 1]
                        )

                        self.reward[write_position, 0] = (
                            rewards[i]
                        )

                        self.not_done[write_position, 0] = (
                            1.0 - float(done_bool)
                        )

                        write_position += 1
                        episode_step += 1

            # =================================================
            # VERIFY DATASET LOADING
            # =================================================

            if write_position != valid_count:

                raise RuntimeError(
                    f"Dataset loading mismatch. "
                    f"Expected {valid_count}, "
                    f"wrote {write_position}."
                )

            self.size = write_position
            self.ptr = write_position % self.max_size

        self.flush()

        print("\nD4RL LOADING COMPLETE")
        print("Transitions:", f"{self.size:,}")
        print("State shape:", self.state.shape)
        print("Action shape:", self.action.shape)
        print("Next state shape:", self.next_state.shape)
        print("Reward shape:", self.reward.shape)
        print("Storage:", self.storage_dir)
        print("========================================\n")

    # --------------------------------------------------------
    # FLUSH MEMORY-MAPPED ARRAYS TO DISK
    # --------------------------------------------------------

    def flush(self):

        for array in (
            self.state,
            self.action,
            self.next_state,
            self.reward,
            self.not_done
        ):

            if isinstance(array, np.memmap):
                array.flush()

    # --------------------------------------------------------
    # RELEASE MEMORY-MAPPED FILES
    # --------------------------------------------------------

    def close(self):

        self.flush()

        for name in (
            "state",
            "action",
            "next_state",
            "reward",
            "not_done"
        ):

            array = getattr(self, name, None)

            if isinstance(array, np.memmap):

                array.flush()
                mmap_object = getattr(array, "_mmap", None)

                if mmap_object is not None:
                    mmap_object.close()

                setattr(self, name, None)


# ============================================================
# STANDARD MULTILAYER PERCEPTRON
# ============================================================

class MLP(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        n_layers,
        activations: Callable = nn.ReLU,
        activate_final: int = False,
        dropout_rate: Optional[float] = None
    ):

        super().__init__()

        self.affines = []

        self.affines.append(
            nn.Linear(in_dim, hidden_dim)
        )

        for i in range(n_layers - 2):

            self.affines.append(
                nn.Linear(hidden_dim, hidden_dim)
            )

        self.affines.append(
            nn.Linear(hidden_dim, out_dim)
        )

        self.affines = nn.ModuleList(
            self.affines
        )

        self.activations = activations()

        self.activate_final = activate_final

        self.dropout_rate = dropout_rate

        if dropout_rate is not None:

            self.dropout = Dropout(
                self.dropout_rate
            )

            self.norm_layer = nn.LayerNorm(
                hidden_dim
            )

    def forward(self, x):

        for i in range(len(self.affines)):

            x = self.affines[i](x)

            if (
                i != len(self.affines) - 1
                or self.activate_final
            ):

                x = self.activations(x)

                if self.dropout_rate is not None:

                    x = self.dropout(x)

                    # Preserve original behavior:
                    # x = self.norm_layer(x)

        return x


# ============================================================
# INITIALIZATION UTILITIES
# ============================================================

def identity(x):

    return x


def fanin_init(tensor, scale=1):

    size = tensor.size()

    if len(size) == 2:

        fan_in = size[0]

    elif len(size) > 2:

        fan_in = np.prod(size[1:])

    else:

        raise Exception(
            "Shape must have dimension at least 2."
        )

    bound = scale / np.sqrt(fan_in)

    return tensor.data.uniform_(
        -bound,
        bound
    )


def orthogonal_init(tensor, gain=0.01):

    torch.nn.init.orthogonal_(
        tensor,
        gain=gain
    )


# ============================================================
# PARALLELIZED ENSEMBLE LAYER
# ============================================================

class ParallelizedLayerMLP(nn.Module):

    def __init__(
        self,
        ensemble_size,
        input_dim,
        output_dim,
        w_std_value=1.0,
        b_init_value=0.0
    ):

        super().__init__()

        # Approximation to truncated normal of 2 stds.

        w_init = torch.randn(
            (
                ensemble_size,
                input_dim,
                output_dim
            )
        )

        w_init = (
            torch.fmod(w_init, 2)
            * w_std_value
        )

        self.W = nn.Parameter(
            w_init,
            requires_grad=True
        )

        # Constant initialization.

        b_init = torch.zeros(
            (
                ensemble_size,
                1,
                output_dim
            )
        ).float()

        b_init += b_init_value

        self.b = nn.Parameter(
            b_init,
            requires_grad=True
        )

    def forward(self, x):

        # Input:
        # (ensemble_size, batch_size, dimension)

        return x @ self.W + self.b


# ============================================================
# PARALLELIZED ENSEMBLE FLATTEN MLP
# ============================================================

class ParallelizedEnsembleFlattenMLP(nn.Module):

    def __init__(
        self,
        ensemble_size,
        hidden_sizes,
        input_size,
        output_size,
        init_w=3e-3,
        hidden_init=fanin_init,
        w_scale=1,
        b_init_value=0.1,
        layer_norm=None,
        final_init_scale=None,
        dropout_rate=None
    ):

        super().__init__()

        self.ensemble_size = ensemble_size

        self.input_size = input_size

        self.output_size = output_size

        self.elites = [
            i for i in range(self.ensemble_size)
        ]

        self.sampler = np.random.default_rng()

        self.hidden_activation = F.relu

        self.output_activation = identity

        self.layer_norm = layer_norm

        self.fcs = []

        self.dropout_rate = dropout_rate

        if self.dropout_rate is not None:

            self.dropout = Dropout(
                self.dropout_rate
            )

        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):

            fc = ParallelizedLayerMLP(
                ensemble_size=ensemble_size,
                input_dim=in_size,
                output_dim=next_size
            )

            for j in self.elites:

                hidden_init(
                    fc.W[j],
                    w_scale
                )

                fc.b[j].data.fill_(
                    b_init_value
                )

            self.__setattr__(
                'fc%d' % i,
                fc
            )

            self.fcs.append(fc)

            in_size = next_size

        self.last_fc = ParallelizedLayerMLP(
            ensemble_size=ensemble_size,
            input_dim=in_size,
            output_dim=output_size
        )

        if final_init_scale is None:

            self.last_fc.W.data.uniform_(
                -init_w,
                init_w
            )

            self.last_fc.b.data.uniform_(
                -init_w,
                init_w
            )

        else:

            for j in self.elites:

                orthogonal_init(
                    self.last_fc.W[j],
                    final_init_scale
                )

                self.last_fc.b[j].data.fill_(0)

    def forward(self, *inputs, **kwargs):

        flat_inputs = torch.cat(
            inputs,
            dim=-1
        )

        dim = len(flat_inputs.shape)

        # Repeat input for ensemble parallelization.

        if dim < 3:

            flat_inputs = flat_inputs.unsqueeze(0)

            if dim == 1:

                flat_inputs = flat_inputs.unsqueeze(0)

            flat_inputs = flat_inputs.repeat(
                self.ensemble_size,
                1,
                1
            )

        h = flat_inputs

        # Standard feedforward network.

        for _, fc in enumerate(self.fcs):

            h = fc(h)

            h = self.hidden_activation(h)

            if self.dropout_rate:

                h = self.dropout(h)

            if (
                hasattr(self, 'layer_norm')
                and self.layer_norm is not None
            ):

                h = self.layer_norm(h)

        preactivation = self.last_fc(h)

        output = self.output_activation(
            preactivation
        )

        # Original single-input behavior.

        if dim == 1:

            output = output.squeeze(1)

        # Output:
        # (ensemble_size, batch_size, output_size)

        return output

    # --------------------------------------------------------
    # ENSEMBLE SAMPLING
    # --------------------------------------------------------

    def sample(self, *inputs):

        preds = self.forward(*inputs)

        sample_idxs = np.random.choice(
            self.ensemble_size,
            2,
            replace=False
        )

        preds_sample = preds[sample_idxs]

        return (
            torch.min(
                preds_sample,
                dim=0
            )[0],
            sample_idxs
        )
