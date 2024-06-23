import pytest 
import mdtraj as md
import numpy as np
from mdagent.tools.base_tools.secondary_structure import ComputeDSSP, ComputeGyrationTensor

@pytest.fixture
def traj():
  return md.load('3pqr.cif')

@pytest.fixture
def compute_dssp_simple(get_registry):
   registry = get_registry("raw", True)
   return ComputeDSSP(path_registry=registry, simplified=True)

@pytest.fixture
def compute_dssp(get_registry):
   registry = get_registry("raw", True)
   return ComputeDSSP(path_registry=registry, simplified=False)

def test_compute_dssp(traj,         compute_dssp_simple, compute_dssp):
  dssp_simple = compute_dssp_simple._compute_dssp(traj)
  assert dssp_simple.shape[1] == 374
  assert np.all(dssp_simple[0][:10] == np.array(['C', 'C', 'C', 'E', 'E', 'E', 'C', 'C', 'E', 'E']))

  dssp = compute_dssp._compute_dssp(traj)
  assert dssp.shape[1] == 374
  assert np.all(dssp[0][:10] == [' ', ' ', ' ', 'E', 'E', 'E', 'T', 'T', 'E', 'E'])

def test_dssp_codes(compute_dssp_simple, compute_dssp):
    dssp_codes_simple = compute_dssp_simple._dssp_codes()
    assert dssp_codes_simple == ["H", "E", "C"]

    nl_simple = compute_dssp_simple._dssp_natural_language()
    assert nl_simple == {"H": "helix","E": "strand","C": "coil"}

    dssp_codes = compute_dssp._dssp_codes()
    assert dssp_codes == ["H", "B", "E", "G", "I", "T", "S", " "]

    nl = compute_dssp._dssp_natural_language()
    assert nl == {
            "H": "alpha helix",
            "B": "beta bridge",
            "E": "extended strand",
            "G": "three helix",
            "I": "five helix",
            "T": "hydrogen bonded turn",
            "S": "bend",
            " ": "loop or irregular"
        }
    
def test_convert_dssp_counts(compute_dssp_simple, compute_dssp):
    dssp_counts = {'H': 0, 'E': 5, 'C': 5}

    descriptive_counts = compute_dssp_simple._convert_dssp_counts(dssp_counts)
    assert descriptive_counts == {'helix': 0, 'strand': 5, 'coil': 5}

    dssp_counts = {'H': 0, 'B': 0, 'E': 5, 'G': 0, 'I': 0, 'T': 0, 'S': 0, ' ': 5}

    descriptive_counts = compute_dssp._convert_dssp_counts(dssp_counts)

    assert descriptive_counts == {
        "alpha helix": 0,
        "beta bridge": 0,
        "extended strand": 5,
        "three helix": 0,
        "five helix": 0,
        "hydrogen bonded turn": 0,
        "bend": 0,
        "loop or irregular": 5
    }

def test_summarize_dssp(compute_dssp_simple, compute_dssp):
    dssp_array = np.array([['C', 'C', 'C', 'E', 'E', 'E', 'C', 'C', 'E', 'E']])
    summary = compute_dssp_simple._summarize_dssp(dssp_array)
    assert summary == {'helix': 0, 'strand': 5, 'coil': 5}

    dssp_array = np.array([[' ', ' ', ' ', 'E', 'E', 'E', 'T', 'T', 'E', 'E']])
    summary = compute_dssp._summarize_dssp(dssp_array)
    assert summary == {
        'alpha helix': 0,
        'beta bridge': 0,
        'extended strand': 5,
        'three helix': 0,
        'five helix': 0,
        'hydrogen bonded turn': 2,
        'bend': 0,
        'loop or irregular': 3}
    
def test_compute_gyration_tensor(get_registry):
    gyration_tensor = np.array([[[ 3.45897484,  0.17571401, -0.08759158],
    [ 0.17571401,  0.944077  ,  0.17698189],
        [-0.08759158,  0.17698189,  0.73760228]]])
    registry = get_registry("raw", True)
    gy_tensor = ComputeGyrationTensor(path_registry=registry)._compute_gyration_tensor(traj)
    assert np.allclose(gy_tensor, gyration_tensor)