import json
import pickle
from typing import Any
import yaml


def read_file(path: str) -> str:
    with open(path) as file:
        return file.read()


def save_file(content: str, path: str) -> None:
    with open(path, "w") as file:
        file.write(content)


def read_yaml(path: str) -> str:
    with open(path) as file:
        return yaml.full_load(file)


def save_pickle(obj: Any, path: str) -> None:
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def save_json(obj: Any, path: str) -> None:
    with open(path, "w") as file:
        json.dump(obj, file)
