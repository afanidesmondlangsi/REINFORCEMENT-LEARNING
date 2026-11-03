import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--params", required=True)
args = parser.parse_args()

config = json.loads(args.params)
print("Received reduction:", config["ea_bc_reduction"])
print("Received gate BC:", config["ea_gate_bc"])
