from pathlib import Path
import xml.etree.ElementTree as ET

source = Path("envs/mujoco/assets")
destination = Path("envs/mujoco/assets_corrected")
destination.mkdir(parents=True, exist_ok=True)

for shift in ["0.1", "0.5", "2.0", "5.0"]:
    filename = f"hopper_friction_{shift}.xml"

    tree = ET.parse(source / filename)
    root = tree.getroot()

    floor = root.find(".//geom[@name='floor']")

    if floor is None:
        raise RuntimeError(f"Floor geometry missing: {filename}")

    floor.set("friction", "0.01 0.005 0.0001")

    output = destination / filename
    tree.write(output, encoding="unicode", xml_declaration=True)

    print(f"Created: {output}")
