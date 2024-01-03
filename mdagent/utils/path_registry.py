import json
import os
from datetime import datetime
from enum import Enum


class FileType(Enum):
    PROTEIN = 1
    SIMULATION = 2
    RECORD = 3


class PathRegistry:
    instance = None

    @classmethod
    def get_instance(cls):
        if not cls.instance:
            cls.instance = cls()
        return cls.instance

    def __init__(self):
        self.json_file_path = "paths_registry.json"

    def _get_full_path(self, file_path):
        return os.path.abspath(file_path)

    def _check_for_json(self):
        # check short path first
        short_path = self.json_file_path
        if os.path.exists(short_path):
            return True
        full_path = self._get_full_path(self.json_file_path)
        if os.path.exists(full_path):
            self.json_file_path = full_path
            return True
        return False

    def _save_mapping_to_json(self, path_dict):
        existing_data = {}
        if self._check_for_json():
            with open(self.json_file_path, "r") as json_file:
                existing_data = json.load(json_file)
                existing_data.update(path_dict)
        with open(self.json_file_path, "w") as json_file:
            json.dump(existing_data, json_file, indent=4)

    def _check_json_content(self, name):
        if not self._check_for_json():
            return False
        with open(self.json_file_path, "r") as json_file:
            data = json.load(json_file)
            return name in data

    # we use this fxn to "save" files (paths) to the json file
    def map_path(self, name, path, description=None):
        description = description or "No description provided"
        full_path = self._get_full_path(path)
        path_dict = {name: {"path": full_path, "description": description}}
        self._save_mapping_to_json(path_dict)
        saved = self._check_json_content(name)
        return f"Path {'successfully' if saved else 'not'} mapped to name: {name}"

    # this if we want to get the path. not use as often
    def get_mapped_path(self, name):
        if not self._check_for_json():
            return "The JSON file does not exist."
        with open(self.json_file_path, "r") as json_file:
            data = json.load(json_file)
            return data.get(name, {}).get("path", "Name not found in path registry.")

    def _clear_json(self):
        if self._check_for_json():
            with open(self.json_file_path, "w") as json_file:
                json.dump({}, json_file)  # Writing an empty JSON object to the file
            return "JSON file cleared"
        return "JSON file does not exist"

    def _remove_path_from_json(self, name):
        if not self._check_for_json():
            return "JSON file does not exist"
        with open(self.json_file_path, "r") as json_file:
            data = json.load(json_file)
        if name in data:
            del data[name]
            with open(self.json_file_path, "w") as json_file:
                json.dump(data, json_file, indent=4)
            return f"Path {name} removed from registry"
        return f"Path {name} not found in registry"

    def list_path_names(self):
        if not self._check_for_json():
            return "JSON file does not exist"
        with open(self.json_file_path, "r") as json_file:
            data = json.load(json_file)
        names = [key for key in data.keys()]
        return (
            "Names found in registry: " + ", ".join(names)
            if names
            else "No names found. The JSON file is empty or does not"
            "contain name mappings."
        )

    def get_timestamp(self):
        # Get the current date and time
        now = datetime.now()
        # Format the date and time as "YYYYMMDD_HHMMSS"
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        return timestamp

    # File Name/ID in Path Registry JSON
    def get_fileid(self, file_name: str, type: FileType):
        # Split the filename on underscores
        parts, ending = file_name.split(".")
        parts_list = parts.split("_")

        # Extract the timestamp (assuming it's always in the second to last part)
        timestamp_part = parts_list[-1]
        # Get the last 6 digits of the timestamp
        timestamp_digits = timestamp_part[-6:]

        if type == FileType.PROTEIN:
            # Extract the PDB ID (assuming it's always the first part)
            pdb_id = parts_list[0]
            print(pdb_id, "pdb abbreviation")
            return pdb_id + "_" + timestamp_digits
        if type == FileType.SIMULATION:
            return "sim" + "_" + timestamp_digits
        if type == FileType.RECORD:
            return "rec" + "_" + timestamp_digits

    def write_file_name(self, type: FileType, **kwargs):
        time_stamp = self.get_timestamp()
        protein_name = kwargs.get("protein_name", None)
        description = kwargs.get("description", "No description provided")
        file_format = kwargs.get("file_format", "No file format provided")
        protein_file_id = kwargs.get("protein_file_id", None)
        type_of_sim = kwargs.get("type_of_sim", None)
        conditions = kwargs.get("conditions", None)
        Sim_id = kwargs.get("Sim_id", None)
        if type == FileType.PROTEIN:
            file_name = f"{protein_name}_{description}_{time_stamp}.{file_format}"
        if type == FileType.SIMULATION:
            file_name = f"{type_of_sim}_{protein_file_id}_{conditions}_{time_stamp}"
        if type == FileType.RECORD:
            file_name = f"{protein_file_id}_{Sim_id}_{time_stamp}"

        return file_name
