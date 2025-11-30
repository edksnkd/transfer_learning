import yaml
from pathlib import Path

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def merge_dicts(parent, child):
    """Рекурсивное обновление полей."""
    for k, v in child.items():
        if isinstance(v, dict) and k in parent:
            merge_dicts(parent[k], v)
        else:
            parent[k] = v
    return parent

def load_config(path: str):
    cfg = load_yaml(path)

    if "inherits" in cfg:
        base_path = Path(path).parent / cfg["inherits"]
        base_cfg = load_yaml(base_path)
        cfg = merge_dicts(base_cfg, cfg)

    return cfg
