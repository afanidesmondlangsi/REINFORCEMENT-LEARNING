from pathlib import Path

path = Path("algo/offline_online/ea_ntc_bc_sac.py")
src = path.read_text(encoding="utf-8")

old = """            self._transfer_quality_probe(
                tar_replay_buffer,
                src_state,
                src_action,
                probe_weights.detach(),
            )"""

new = """            raw_probe_weights = getattr(
                self, "ea_last_raw_source_weights", None
            )
            if raw_probe_weights is not None:
                self._transfer_quality_probe(
                    tar_replay_buffer,
                    src_state,
                    src_action,
                    raw_probe_weights.detach(),
                )"""

if src.count(old) != 1:
    raise RuntimeError("Expected probe call not found exactly once")

src = src.replace(old, new, 1)
path.write_text(src, encoding="utf-8")

print("Raw compatibility weighting patch applied.")
