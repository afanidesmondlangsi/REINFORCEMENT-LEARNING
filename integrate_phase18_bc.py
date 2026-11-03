
from pathlib import Path
import ast
import textwrap

path = Path("algo/offline_online/ea_ntc_bc_sac.py")
src = path.read_text(encoding="utf-8")

if "self.ea_bc_controller =" in src:
    raise SystemExit(
        "Phase 18 controller already present. No changes made."
    )


def replace_once(old, new):
    global src

    count = src.count(old)

    if count != 1:
        raise RuntimeError(
            "Expected one occurrence, found "
            + str(count) + ": " + repr(old[:100])
        )

    src = src.replace(old, new, 1)


# --------------------------------------------------
# 1. Import adaptive BC controller
# --------------------------------------------------

anchor = (
    "from offline_online.ea_ntc.adaptive_gate "
    "import AdaptiveGateController"
)

replace_once(
    anchor,
    anchor + "\n"
    "from offline_online.ea_ntc.adaptive_bc "
    "import AdaptiveBCController"
)


# --------------------------------------------------
# 2. Initialize adaptive BC controller
# --------------------------------------------------

anchor = "        self.ea_tq_strength = 0.0"

replace_once(
    anchor,
    """        self.ea_adaptive_bc_enabled = bool(
            config.get("ea_adaptive_bc_enabled", False)
        )

        self.ea_bc_controller = AdaptiveBCController(
            confidence_multiplier=float(
                config.get("ea_bc_confidence_multiplier", 2.0)
            ),
            margin=float(
                config.get("ea_bc_quality_margin", 0.0)
            ),
            uncertainty_scale=float(
                config.get("ea_bc_uncertainty_scale", 2.0)
            ),
            max_coefficient=float(
                config.get("ea_bc_max_coefficient", 1.0)
            ),
        )

        self.ea_bc_coefficient = (
            0.0 if self.ea_adaptive_bc_enabled else 1.0
        )

        if (
            self.ea_adaptive_bc_enabled
            and not self.ea_transfer_quality_enabled
        ):
            raise ValueError(
                "Adaptive BC requires transfer quality to be enabled"
            )

""" + anchor
)


# --------------------------------------------------
# 3. Update beta after transfer-quality probe
# --------------------------------------------------

anchor = """        self.ea_tq_quality = quality
        self.ea_tq_uncertainty = uncertainty"""

replace_once(
    anchor,
    anchor + """

        if self.ea_adaptive_bc_enabled:
            self.ea_bc_coefficient = self.ea_bc_controller.update(
                quality, uncertainty
            )"""
)


# --------------------------------------------------
# 4. Modify weighted BC objective using AST
# --------------------------------------------------

start = src.index("    def update_policy_and_temp(")

end = src.index("\n    def ", start + 5)

section = src[start:end]

# Remove class indentation only for AST parsing.
dedented = textwrap.dedent(section)

tree = ast.parse(dedented)

policy_assignments = [
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.Assign)
    and any(
        isinstance(target, ast.Name)
        and target.id == "policy_loss"
        for target in node.targets
    )
]

if len(policy_assignments) != 1:
    raise RuntimeError(
        "Expected exactly one policy_loss assignment; found "
        + str(len(policy_assignments))
    )

assignment = policy_assignments[0]

bc_references = [
    node
    for node in ast.walk(assignment.value)
    if isinstance(node, ast.Name)
    and node.id == "bc_loss"
]

if len(bc_references) != 1:
    raise RuntimeError(
        "Expected exactly one bc_loss reference; found "
        + str(len(bc_references))
    )

node = bc_references[0]

original_lines = section.splitlines(keepends=True)
dedented_lines = dedented.splitlines(keepends=True)

line_index = node.lineno - 1

original_line = original_lines[line_index]
dedented_line = dedented_lines[line_index]

# Calculate the indentation removed from this line.
removed_indent = len(original_line) - len(
    original_line.lstrip(" ")
)
dedented_indent = len(dedented_line) - len(
    dedented_line.lstrip(" ")
)

offset_adjustment = removed_indent - dedented_indent

start_column = node.col_offset + offset_adjustment
end_column = node.end_col_offset + offset_adjustment

old_fragment = original_line[start_column:end_column]

if old_fragment != "bc_loss":
    raise RuntimeError(
        "Unexpected BC loss expression: "
        + repr(old_fragment)
    )

original_lines[line_index] = (
    original_line[:start_column]
    + "(self.ea_bc_coefficient * bc_loss)"
    + original_line[end_column:]
)

modified_section = "".join(original_lines)

src = (
    src[:start]
    + modified_section
    + src[end:]
)


# --------------------------------------------------
# 5. Add TensorBoard diagnostics
# --------------------------------------------------

anchor = """                writer.add_scalar(
                    "ea_ntc/transfer_gate_strength","""

replace_once(
    anchor,
    """                writer.add_scalar(
                    "ea_ntc/source_bc_coefficient",
                    self.ea_bc_coefficient,
                    self.total_it,
                )

                if self.ea_bc_controller.last_lower_bound is not None:
                    writer.add_scalar(
                        "ea_ntc/transfer_quality_lower_bound",
                        self.ea_bc_controller.last_lower_bound,
                        self.total_it,
                    )

""" + anchor
)


# --------------------------------------------------
# 6. Validate complete Python syntax before saving
# --------------------------------------------------

ast.parse(src)

path.write_text(src, encoding="utf-8")

print("Phase 18 adaptive BC integration completed.")
