# defining tools
import json
import os
from typing import Optional

from langchain.tools import BaseTool


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
                json_file.truncate(0)
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


class MapPath2Name(BaseTool):
    name = "MapPath2Name"
    description = """Input the desired filename
    followed by the file's path, separated by a comma.
    Make sure the name is first, then the path.
    Your path should look something like: name.pdb
    Stores the path in the registry with the
    name provided in the filename.
    If the output says Path mapped to name,
    then it was successful.
    You do not need to check that file was created."""
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, file_and_path: str) -> str:
        """Use the tool"""
        try:
            if self.path_registry is None:
                return "Path registry not initialized"
            if "," not in file_and_path:
                return "Please separate filename and path with a comma"
            file, path = file_and_path.split(",")
            map_name = self.path_registry.map_path(file, path)
            return map_name
        except Exception:
            return "Error writing paths to file"

    async def _arun(self, file_name: str) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError


class ListRegistryPaths(BaseTool):
    name = "ListRegistryPaths"
    description = """Use this tool to list all paths saved in memory.
    Input the word 'paths' and the tool will return a list of all names
    in the registry that are mapped to paths."""
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, paths: str) -> str:
        """Use the tool"""
        try:
            if self.path_registry is None:
                return "Path registry not initialized"
            return self.path_registry.list_path_names()
        except Exception:
            return "Error listing paths"

    async def _arun(self, paths: str) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError
