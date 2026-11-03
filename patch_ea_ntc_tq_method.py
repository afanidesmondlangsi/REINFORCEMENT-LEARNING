from pathlib import Path

path = Path("algo/offline_online/ea_ntc_bc_sac.py")
src = path.read_text(encoding="utf-8")

old = """    def _transfer_quality_probe(
        self, tar_replay_buffer, src_state, src_action,
        raw_source_weights,
    ):"""

new = """    def _transfer_quality_probe(
        self, tar_replay_buffer, src_state, src_action,
        raw_source_weights,
    ):
        # Preserve the training RNG across the entire diagnostic.
        devices = []
        if torch.device(self.device).type == "cuda":
            index = torch.device(self.device).index
            if index is None:
                index = torch.cuda.current_device()
            devices = [index]

        with torch.random.fork_rng(devices=devices):
            return self._transfer_quality_probe_impl(
                tar_replay_buffer,
                src_state,
                src_action,
                raw_source_weights,
            )

    def _transfer_quality_probe_impl(
        self, tar_replay_buffer, src_state, src_action,
        raw_source_weights,
    ):"""

if src.count(old) != 1:
    raise RuntimeError(
        "Expected original probe definition not found exactly once. "
        "No changes made."
    )

src = src.replace(old, new, 1)
path.write_text(src, encoding="utf-8")

print("Probe-level RNG isolation patch applied.")
