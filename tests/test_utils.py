import os

import pytest

from mdagent.utils.clear_mem import ClearMemory


@pytest.fixture
def clear_memory():
    return ClearMemory()


@pytest.fixture
def setup_test_environment(tmp_path):
    # temporary test environment
    base_dir = tmp_path / "test_env"
    base_dir.mkdir()
    (base_dir / "files" / "pdb").mkdir(parents=True)
    (base_dir / "files" / "simulations").mkdir(parents=True)
    (base_dir / "files" / "records").mkdir(parents=True)
    (base_dir / "temp_file").write_text("Temporary file content")
    (base_dir / "paths_registry.json").write_text("{}")
    return base_dir


def test_clear_ckpts(setup_test_environment, clear_memory):
    base_dir = setup_test_environment
    os.chdir(base_dir)
    clear_memory._clear_ckpts()

    # Check that the directories and files have been removed
    assert not os.path.exists(base_dir / "files" / "pdb")
    assert not os.path.exists(base_dir / "files" / "simulations")
    assert not os.path.exists(base_dir / "files" / "records")
    assert not os.path.exists(base_dir / "temp_file")
    assert not os.path.exists(base_dir / "paths_registry.json")


def test_clear_ckpts_root(setup_test_environment, clear_memory):
    base_dir = setup_test_environment
    os.chdir(base_dir / "files")
    (base_dir / "setup.py").write_text("# setup.py content")

    clear_memory.clear_ckpts_root()
    assert os.getcwd() == str(base_dir)

    # Check that the directories and files have been removed
    assert not os.path.exists(base_dir / "files" / "pdb")
    assert not os.path.exists(base_dir / "files" / "simulations")
    assert not os.path.exists(base_dir / "files" / "records")
    assert not os.path.exists(base_dir / "temp_file")
    assert not os.path.exists(base_dir / "paths_registry.json")
