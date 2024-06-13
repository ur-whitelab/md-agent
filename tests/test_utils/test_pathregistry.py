import os

import pytest


@pytest.mark.parametrize("raw_or_clean, with_files", [("raw", False), ("raw", True)])
def test_registry_init(get_registry, with_files, raw_or_clean):
    registry = get_registry(raw_or_clean, with_files)
    print(with_files, raw_or_clean)
    if not with_files:
        assert registry._load_existing_registry() == {}
    else:
        if raw_or_clean == "raw":
            absolute_path = os.path.abspath(f"{registry.ckpt_pdb}/ALA_raw_123456.pdb")
            expected_json = {
                "ALA_123456": {
                    "path": f"{absolute_path}",
                    "name": "ALA_raw_123456.pdb",
                    "description": "No description provided",
                }
            }
            reg = registry._load_existing_registry()
            assert "ALA_123456" in reg
            assert "ALA_654321" not in reg  # ensure the clean file is not there
            assert (
                reg["ALA_123456"] == expected_json["ALA_123456"]
            )  # ensure dicts match

        elif raw_or_clean == "clean":
            absolute_path = os.path.abspath(f"{registry.ckpt_pdb}/ALA_clean_654321.pdb")
            expected_json = {
                "ALA_654321": {
                    "path": f"{absolute_path}",
                    "name": "ALA_clean_654321.pdb",
                    "description": "No description provided",
                }
            }
            reg = registry._load_existing_registry()
            assert "ALA_654321" in registry
            assert "ALA_123456" not in registry
            assert registry["ALA_654321"] == expected_json["ALA_654321"]
