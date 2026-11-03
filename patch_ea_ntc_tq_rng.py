from pathlib import Path

path = Path("algo/offline_online/ea_ntc_bc_sac.py")
src = path.read_text(encoding="utf-8")

old = """                self._transfer_quality_probe(
                    tar_replay_buffer,
                    src_state,
                    src_action,
                    raw_probe_weights.detach(),
                )"""

new = """                # Isolate every stochastic operation in the probe,
                # including stochastic samples made during evaluation.
                probe_devices = []
                if torch.device(self.device).type == "cuda":
                    device_index = torch.device(self.device).index
                    if device_index is None:
                        device_index = torch.cuda.current_device()
                    probe_devices = [device_index]

                with torch.random.fork_rng(devices=probe_devices):
                    self._transfer_quality_probe(
                        tar_replay_buffer,
                        src_state,
                        src_action,
                        raw_probe_weights.detach(),
                    )"""

if src.count(old) != 1:
    raise RuntimeError(
        "Expected transfer-quality probe call not found exactly once. "
        "No changes made."
    )

src = src.replace(old, new, 1)
path.write_text(src, encoding="utf-8")

print("Training-call RNG isolation patch applied.")
