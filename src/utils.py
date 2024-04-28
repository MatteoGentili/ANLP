import json
import os
import pickle
from typing import Any

import pandas as pd


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        # If the folder doesn't exist, create it
        try:
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created successfully.")
        except OSError as e:
            print(f"Error: Failed to create folder '{folder_path}'. Reason: {e}")


class ObjectManager:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def save_object(self, obj) -> None:
        with open(self.file_path, "wb") as outp:  # Overwrites any existing file
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

    def load_object(self) -> Any:
        with open(self.file_path, "rb") as inp:
            obj = pickle.load(inp)
            return obj


def load_data(file_path: str, transform_labels: bool = False) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Could not find "{file_path}".')

    if file_path.endswith(".json"):
        # training file
        with open(file_path) as f:
            json_data: dict = json.load(f)

        data = []
        for label, sentences in json_data.items():
            for sentence in sentences:
                data.append({"text": sentence, "label": label})

    if file_path.endswith(".txt"):
        with open(file_path, "r") as file:
            txt_data = file.readlines()
        data = [{"text": sentence.strip(), "label": None} for sentence in txt_data]

    data = pd.DataFrame(data)

    if transform_labels:
        # Transform labels in numbers
        unique_labels = data["label"].unique()
        label_dict = {label: id for id, label in enumerate(unique_labels)}
        data["label"] = data["label"].apply(lambda x: label_dict[x])
    return data
