import json
import os


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
            json.dump(existing_data, json_file)

    def _check_json_content(self, name):
        if not self._check_for_json():
            return False
        with open(self.json_file_path, "r") as json_file:
            data = json.load(json_file)
            return name in data

    def map_path(self, name, path, description=None):
        description = description or "No description provided"
        full_path = self._get_full_path(path)
        path_dict = {name: {"path": full_path, "description": description}}
        self._save_mapping_to_json(path_dict)
        saved = self._check_json_content(name)
        return f"Path {'successfully' if saved else 'not'} mapped to name: {name}"

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
                json.dump(data, json_file)
            return f"Path {name} removed from registry"
        return f"Path {name} not found in registry"

    def list_path_names(self):
        if not self._check_for_json():
            return "JSON file does not exist"
        with open(self.json_file_path, "r") as json_file:
            data = json.load(json_file)
        names = [key for key in data.keys()]
        return (
            "Names in path registry: " + ", ".join(names)
            if names
            else """No names found.
            The JSON file is empty or doesn't
            contain name mappings."""
        )
