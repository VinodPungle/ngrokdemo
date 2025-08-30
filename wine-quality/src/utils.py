import os
import yaml
from typing import Any, Dict

def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)