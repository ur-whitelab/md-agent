# defining tools
from typing import Optional

from langchain.tools import BaseTool

from .openmm_registry import OpenMMObjectRegistry
from .path_registry import PathRegistry


class Paths2File(BaseTool):
    name = "Paths2File"
    description = """Input the desired output file name,
    if applicable. If no file name is provided,
    use path_registry.txt.
    creates a file listing all paths saved in registry.
    Only use this tool if user requests this file explicitly."""
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, file_name: str) -> str:
        """Use the tool"""
        try:
            if self.path_registry is None:
                return "Path registry not initialized"
            file_written = self.path_registry.write_paths_to_file(file_name)
            return file_written
        except Exception:
            return "Error writing paths to file"

    async def _arun(self, file_name: str) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError


class Objects2File(BaseTool):
    name = "Objects2File"
    description = """Input the desired output file name,
    if applicable. If no file name is provided,
    use object_registry.txt.
    creates a file listing all object names and
    descriptions saved in registry.
    Only use this tool if user requests this file explicitly."""
    object_registry: Optional[OpenMMObjectRegistry]

    def __init__(self, object_registry: Optional[OpenMMObjectRegistry]):
        super().__init__()
        self.object_registry = object_registry

    def _run(self, file_name: str) -> str:
        """Use the tool"""
        try:
            if self.object_registry is None:
                return "Object registry not initialized"
            file_written = self.object_registry.write_objects_to_file(file_name)
            return file_written
        except Exception:
            return "Error writing object registry to file"

    async def _arun(self, file_name: str) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError


class FullRegistry2File(BaseTool):
    name = "FullRegistry2File"
    description = """Input the desired output file name,
    if applicable. If no file name is provided,
    use full_registry.txt.
    creates a file listing all paths and objects saved in registry.
    Only use this tool if user requests this file explicitly."""
    path_registry: Optional[PathRegistry]
    object_registry: Optional[OpenMMObjectRegistry]

    def __init__(
        self,
        path_registry: Optional[PathRegistry],
        object_registry: Optional[OpenMMObjectRegistry],
    ):
        super().__init__()
        self.path_registry = path_registry
        self.object_registry = object_registry

    def _run(self, file_name="full_registry.txt"):
        """Use the tool"""
        try:
            file_path = self.path_registry._get_full_path(file_name)
            with open(file_name, "w") as file:
                # Write object registry title
                file.write("Objects saved in registry:\n")
                file.write("Object Name: description\n")

                # Write object registry
                for name, obj_info in self.object_registry.objects.items():
                    description = obj_info.get("description", "")
                    file.write(f"{name}: {description}\n")

                # Write path registry title
                file.write("\nPath directories registered:\n")
                file.write("Path Name: path (description)\n")

                # Write path registry
                for name, path_info in self.paths.items():
                    path = path_info["path"]
                    description = path_info.get("description", "")
                    file.write(f"{name}: {path} ({description})\n")

            return f"Full registry written to file: {file_path}"

        except Exception:
            return "Error writing registry to file"

    async def _arun(self, file_name: str) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError


class MapPath2Name(BaseTool):
    name = "MapPath2Name"
    description = """Input the desired filename
    followed by the file's path, separated by a comma.
    Make sure the name is first, then the path.
    Stores the path in the registry with the
    name provided in the filename.
    If the output says Path mapped to name,
    then it was successful.
    You do not need to check that file was created.
    Once you use this tool, you may not
    utilize the full path again"""
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
        except Exception as e:
            print(e)
            return "Error writing paths to file"

    async def _arun(self, file_name: str) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError


class ListRegistryObjects(BaseTool):
    name = "ListRegistryObjects"
    description = """Use this tool to list all objects saved in memory.
    Input the word 'objects' and the tool will return a list of all names
    in the registry that are mapped to objects."""
    object_registry: Optional[OpenMMObjectRegistry]

    def __init__(self, object_registry: Optional[OpenMMObjectRegistry]):
        super().__init__()
        self.object_registry = object_registry

    def _run(self, objects: str) -> str:
        """Use the tool"""
        try:
            if self.object_registry is None:
                return "Object registry not initialized"
            return self.object_registry.list_object_names(objects)
        except Exception:
            return "Error listing objects"

    async def _arun(self, objects: str) -> str:
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
            return self.path_registry.list_path_names(paths)
        except Exception:
            return "Error listing paths"

    async def _arun(self, paths: str) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError
