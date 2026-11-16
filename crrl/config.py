from pathlib import Path
import copy
import yaml


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    parent = cfg.pop("base", None)
    if parent:
        merged = load_config(path.parent / parent)
        deep_update(merged, cfg)
        cfg = merged
    return cfg


def deep_update(destination, source):
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(destination.get(key), dict):
            deep_update(destination[key], value)
        else:
            destination[key] = copy.deepcopy(value)


def save_config(cfg, path):
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
