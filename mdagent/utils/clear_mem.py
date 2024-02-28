import glob
import os
import shutil


class ClearMemory:
    def _clear_ckpts(self):
        directories_to_remove = ["files/pdb", "files/simulations", "files/records"]

        for directory in directories_to_remove:
            if os.path.exists(directory):
                shutil.rmtree(directory)

        temp_files = glob.glob("temp_*")
        for temp_file in temp_files:
            if os.path.isfile(temp_file):
                os.remove(temp_file)

        if os.path.exists("paths_registry.json"):
            os.remove("paths_registry.json")

    def clear_ckpts_root(self):
        current_dir = os.getcwd()
        while current_dir != os.path.dirname(current_dir):
            if "setup.py" in os.listdir(current_dir):
                os.chdir(current_dir)
                self._clear_ckpts()
                break
            else:
                current_dir = os.path.dirname(current_dir)
        os.chdir(current_dir)  # Reset the working directory
