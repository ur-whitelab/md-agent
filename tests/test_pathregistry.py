import os

import pytest


@pytest.mark.parametrize("with_files, raw_or_clean", [("raw", False), ("raw", True)])
def test_registry_init(get_registry, with_files, raw_or_clean):
    # make the test directory the cwd
    # print(os.curdir)
    # if os.curdir.split("/")[-1] != "tests":
    #    os.chdir("tests")
    registry_without_files = get_registry(raw_or_clean, with_files)
    print(with_files, raw_or_clean)
    if not with_files:
        assert registry_without_files._load_existing_registry() == {}
    else:
        if raw_or_clean == "raw":
            absolute_path = os.path.abspath("files/pdb/ALA_raw_123456.pdb")
            expected_json = {
                "ALA_123456": {
                    "path": f"{absolute_path}",
                    "name": "ALA_raw_123456.pdb",
                    "description": (
                        "Protein ALA pdb file. "
                        "downloaded from RCSB Protein Data Bank. "
                    ),
                }
            }
            assert registry_without_files._load_existing_registry() == expected_json
        elif raw_or_clean == "clean":
            absolute_path = os.path.abspath("files/pdb/ALA_clean_654321.pdb")
            expected_json = {
                "ALA_654321": {
                    "path": f"{absolute_path}",
                    "name": "ALA_clean_654321.pdb",
                    "description": (
                        "Protein ALA pdb file. "
                        "downloaded from RCSB Protein Data Bank. "
                    ),
                }
            }
            assert registry_without_files._load_existing_registry() == expected_json
