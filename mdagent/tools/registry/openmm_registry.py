class OpenMMObjectRegistry:
    instance = None

    @classmethod
    def get_instance(cls):
        if not cls.instance:
            cls.instance = cls()
        return cls.instance

    def __init__(self):
        if OpenMMObjectRegistry.instance:
            raise Exception("OpenMMObjectRegistry class has already been initialized")
        self.objects = {}

    def map_object(self, name, obj, description=None):
        """use this to add an open-mm object in registry
        and map to a name.
        whenever we create an open-mm object,
        we should add
        it to this registry"""
        if description is None:
            description = "No description provided"
        self.objects[name] = {"object": obj, "description": description}
        return "Object added to registry under name: {name}"

    def get_object(self, name):
        """Get an open-mm object from registry, given name input
        Use this inside of functions where the name is the input"""
        return self.objects.get(name)

    def remove_object(self, name):
        """remove a single open-mm object from registry"""
        if name in self.objects:
            del self.objects[name]
            return f"Object {name} removed from registry"
        else:
            return f"Object {name} not found in registry"

    def clear_object_registry(self):
        """Clear all open-mm objects from registry"""
        self.objects.clear()
        return "Object registry cleared"

    def list_object_names(self, objects_str=None):
        """list names that are mapped to open-mm objects in registry"""
        names = ", ".join(self.objects.keys())
        return "Names in object registry: " + names

    def write_objects_to_file(self, file_name="object_registry.txt"):
        """Write object names and descriptions to a text file."""
        file_name = "object_registry.txt"  # forcing this for now
        with open(file_name, "w") as file:
            for name, obj_info in self.objects.items():
                description = obj_info.get("description", "")
                file.write(f"{name}: {description}\n")

        return f"Object names and descriptions written to file: {file_name}"
