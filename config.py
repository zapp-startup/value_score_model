import yaml
from pathlib import Path


def load_config(path=None) -> dict:

    if path is None:
        path = Path(__file__).parent / "configs" / "default.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)