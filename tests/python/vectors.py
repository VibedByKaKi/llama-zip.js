import json
from pathlib import Path

VECTORS = Path(__file__).resolve().parents[1] / "vectors"


def load_vectors(name):
    with open(VECTORS / name, encoding="utf-8") as f:
        return json.load(f)


def case_ids(cases):
    return [c["name"] for c in cases]
