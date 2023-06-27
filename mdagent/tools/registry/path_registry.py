import os


class PathRegistry:
    instance = None

    @classmethod
    def get_instance(cls):
        if not cls.instance:
            cls.instance = cls()
        return cls.instance

    def __init__(self):
        if PathRegistry.instance:
            raise Exception("PathRegistry class has already been initialized")
        self.paths = {}

    def _get_full_path(self, file_path):
        return os.path.abspath(file_path)

    def map_path(self, name, path, description=None, question=None):
        """Map a name to a path in registry,
        with an optional description."""
        if description is None:
            description = "No description provided"
        full_path = self._get_full_path(path)
        self.paths[name] = {"path": full_path, "description": description}

        return f"Path mapped to name: {name}"

    def get_path(self, name):
        """use this to get the path of a file, given name input
        this will be used inside of functions
        when we need to access a file"""
        path_info = self.paths.get(name.strip().lower())
        if path_info:
            return f"Path: {path_info['path']}"
        else:
            return f"{name} not found in registry"

    def clear_path_registry(self):
        """ "Clear all paths from registry."""
        self.paths.clear()
        return "Path registry cleared"

    def remove_path(self, name):
        """Remove a single path from registry."""
        if name in self.paths:
            del self.paths[name]
            return f"{name} removed from registry"
        else:
            return f"{name} not found in registry"

    def list_path_names(self, paths_str=None):
        """lists names that are mapped to paths in registry"""
        names = ", ".join(self.paths.keys())
        return "Names in path registry: " + names

    def write_paths_to_file(self, file_name="path_registry.txt"):
        """Write path directories to a text file."""
        file_name = "path_registry.txt"  # forcing this for now
        file_path = self._get_full_path(file_name)
        with open(file_name, "w") as file:
            for name, path_info in self.paths.items():
                path = path_info["path"]
                description = path_info.get("description", "")
                file.write(f"{name}: {path} ({description})\n")

        return f"Path directories written to file: {file_path}"
