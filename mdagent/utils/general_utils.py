import os


def find_file_path(file_name: str, exact_match: bool = True):
    """get the path of a file, if it exists in repo"""
    setup_dir = None
    for dirpath, dirnames, filenames in os.walk("."):
        if "setup.py" in filenames:
            setup_dir = dirpath
            break

    if setup_dir is None:
        raise FileNotFoundError("Unable to find root directory.")

    for dirpath, dirnames, filenames in os.walk(setup_dir):
        for filename in filenames:
            if (exact_match and filename == file_name) or (
                not exact_match and file_name in filename
            ):
                return os.path.join(dirpath, filename)

    return None
