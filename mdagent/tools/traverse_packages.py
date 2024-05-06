import importlib
import inspect
import io
import subprocess
import sys
from contextlib import redirect_stdout


class TraversePackages:
    def __init__(self):
        pass

    def list_attributes(self, package):
        all_attributes = dir(package)
        return all_attributes

    def import_module(self, module_name):
        try:
            return importlib.import_module(module_name)
        # if pacakge not installed, try to install it
        except ModuleNotFoundError:
            print(f"Module {module_name} not found. Trying to install it.")
            self.pip_install_package(module_name)
            try:
                return importlib.import_module(module_name)
            except Exception:
                print(f"Error importing module {module_name} after installation.")
                return None
        except ImportError as e:
            print(f"Error importing module {module_name}: {e}")
            return None

    def pip_install_package(self, package):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Error installing package {package}: {e}")
            return None

    def help_all_attributes(self, package, depth=0, path=""):
        if depth > 5:
            return {}

        all_attributes = dir(package)
        all_attribute_details = {}
        for attribute_name in all_attributes:
            attribute = getattr(package, attribute_name)

            qualified_name = f"{path}.{attribute_name}" if path else attribute_name

            if inspect.isfunction(attribute) or inspect.isclass(attribute):
                f = io.StringIO()
                with redirect_stdout(f):
                    help(attribute)
                all_attribute_details[qualified_name] = f.getvalue()
            elif inspect.ismodule(attribute) and depth < 5:
                submodule_details = self.help_all_attributes(
                    attribute, depth + 1, qualified_name
                )
                all_attribute_details.update(submodule_details)

        return all_attribute_details

    def help_on_attribute(self, module_name, attribute_name):
        module = self.import_module(module_name)
        if module:
            try:
                attribute = getattr(module, attribute_name)
                f = io.StringIO()
                with redirect_stdout(f):
                    help(attribute)
                return f.getvalue()
            except AttributeError:
                return f"Error: attribute {attribute_name} not found in {module_name}."
        else:
            return "Module not found."
