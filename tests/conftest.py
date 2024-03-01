import os
import shutil
from pathlib import Path

import pytest

from mdagent.utils import PathRegistry


@pytest.fixture(scope="module")
def raw_alanine_pdb_file(request):
    pdb_content = """
    ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
    ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C
    ATOM      3  C   ALA A   1       2.175   1.395   0.000  1.00 20.00           C
    ATOM      4  O   ALA A   1       1.461   2.400   0.000  1.00 20.00           O
    ATOM      5  CB  ALA A   1       1.958  -0.735  -1.231  1.00 20.00           C
    TER
    END
        """.strip()
    with open("ALA_raw_123456.pdb", "w") as f:
        f.write(pdb_content)
    yield "ALA_raw_123456.pdb"

    request.addfinalizer(lambda: os.remove("ALA_raw_123456.pdb"))


@pytest.fixture(scope="module")
def clean_alanine_pdb_file(request):
    pdb_content = """
REMARK  ACE
CRYST1   32.155   32.155   56.863  90.00  90.00 120.00 P 31 2 1      6
ATOM      1 1HH3 ACE     1       2.000   1.000  -0.000
ATOM      2  CH3 ACE     1       2.000   2.090   0.000
ATOM      3 2HH3 ACE     1       1.486   2.454   0.890
ATOM      4 3HH3 ACE     1       1.486   2.454  -0.890
ATOM      5  C   ACE     1       3.427   2.641  -0.000
ATOM      6  O   ACE     1       4.391   1.877  -0.000
ATOM      7  N   ALA     2       3.555   3.970  -0.000
ATOM      8  H   ALA     2       2.733   4.556  -0.000
ATOM      9  CA  ALA     2       4.853   4.614  -0.000
ATOM     10  HA  ALA     2       5.408   4.316   0.890
ATOM     11  CB  ALA     2       5.661   4.221  -1.232
ATOM     12 1HB  ALA     2       5.123   4.521  -2.131
ATOM     13 2HB  ALA     2       6.630   4.719  -1.206
ATOM     14 3HB  ALA     2       5.809   3.141  -1.241
ATOM     15  C   ALA     2       4.713   6.129   0.000
ATOM     16  O   ALA     2       3.601   6.653   0.000
ATOM     17  N   NME     3       5.846   6.835   0.000
ATOM     18  H   NME     3       6.737   6.359  -0.000
ATOM     19  CH3 NME     3       5.846   8.284   0.000
ATOM     20 1HH3 NME     3       4.819   8.648   0.000
ATOM     21 2HH3 NME     3       6.360   8.648   0.890
ATOM     22 3HH3 NME     3       6.360   8.648  -0.890
TER
END
        """
    with open("ALA_clean_654321.pdb", "w") as f:
        f.write(pdb_content)

    yield "ALA_clean_654321.pdb"

    request.addfinalizer(lambda: os.remove("ALA_clean_654321.pdb"))


@pytest.fixture(scope="function")
def get_registry(raw_alanine_pdb_file, clean_alanine_pdb_file, request):
    created_paths = []  # Keep track of created directories for cleanup

    def create(raw_or_clean, with_files):
        base_path = "files"
        if with_files:
            pdb_path = Path(base_path) / "pdb"
            record_path = Path(base_path) / "records"
            simulation_path = Path(base_path) / "simulation"

            # Create directories
            for path in [pdb_path, record_path, simulation_path]:
                os.makedirs(path, exist_ok=True)
                created_paths.append(path)
            if raw_or_clean == "raw":
                # Copy the alanine pdb file to the pdb/alanine directory
                shutil.copy(raw_alanine_pdb_file, pdb_path)
            elif raw_or_clean == "clean":
                shutil.copy(clean_alanine_pdb_file, pdb_path)

        # Assuming PathRegistry is defined elsewhere and properly implemented
        return PathRegistry()

    # Cleanup: Remove created directories and the copied pdb file
    def cleanup():
        for path in reversed(created_paths):  # Remove directories
            shutil.rmtree(path, ignore_errors=True)
        if os.path.exists("path_registry.json"):
            os.remove("path_registry.json")

    request.addfinalizer(cleanup)

    return create
