import shutil
from pathlib import Path


def find_repo_root(start_path):
    path = Path(start_path).resolve()
    while not (path / "setup.py").exists():
        if path.parent == path:
            raise FileNotFoundError("Could not find the repository root with setup.py.")
        path = path.parent
    print("path: ", path)
    return path


def clear_memory(
    clear_skill=True, clear_files=True, ask_confirmation=False, repo_root=None
):
    print(
        """This script will delete the following:
        1. All files in files/pdb, files/simulation, and files/records directories
        2. All files starting with temp_ in the current directory
        3. The file path_registry.json"""
    )
    if repo_root is None:
        repo_root = find_repo_root(__file__)
    else:
        repo_root = Path(repo_root)
    if ask_confirmation:
        confirmation = input("Are you sure you want to proceed? (y/n): ")
    else:
        confirmation = "y"
    if confirmation.lower() == "y":
        directories_to_clear = []
        if clear_files:
            directories_to_clear += [Path(repo_root) / "files"]
        if clear_skill:
            directories_to_clear += [Path(repo_root) / "ckpt"]
        if not clear_files and not clear_skill:
            return None
        for directory in directories_to_clear:
            shutil.rmtree(directory, ignore_errors=True)
            directory.mkdir(parents=True, exist_ok=True)

        return "Deletion complete."
    else:
        return "Deletion aborted."


if __name__ == "__main__":
    clear_memory()
