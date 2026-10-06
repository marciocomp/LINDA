# @author: Marcio Lopes

import json
import os

def load_json(path):
    """
    Generic helper to load a JSON file from a given path.
    Raises FileNotFoundError if the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found at: {path}")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON from {path}: {e}")