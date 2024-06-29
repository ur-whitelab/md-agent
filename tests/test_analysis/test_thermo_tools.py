import mdtraj as md
import numpy as np
import pytest

from mdagent.tools.base_tools.analysis_tools.thermo_tools import (
    ComputeDipoleMoments,
    ComputeIsothermalCompressabilityKappaT,
    ComputeMassDensity,
    ComputeStaticDielectric,
    GetTrajCharges,
)


@pytest.fixture
def get_traj_charges():
    return GetTrajCharges()


@pytest.fixture
def dummy_traj_with_box(num_frames=10, num_atoms=10, box_size=(2.0, 2.0, 2.0)):
    np.random.seed(49)
    box_lengths = np.array([box_size] * num_frames)

    # random positions
    positions = np.random.rand(num_frames, num_atoms, 3) * np.array(box_size)

    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("DUM", chain)
    for _ in range(num_atoms):
        topology.add_atom("X", element=md.element.carbon, residue=residue)

    angles = np.tile([90, 90, 90], (num_frames, 1))

    traj = md.Trajectory(
        positions, topology, unitcell_lengths=box_lengths, unitcell_angles=angles
    )
    return traj


@pytest.fixture
def dummy_charges():
    return {"X": 0.1}


def test_match_charges_to_traj(get_traj_charges, dummy_charges, dummy_traj_with_box):
    output = get_traj_charges.match_charges_to_traj(dummy_charges, dummy_traj_with_box)
    assert all(x == 0.1 for x in output)


@pytest.mark.skip(reason="Not implemented yet")
def test_compute_charges_from_traj(raw_alanine_pdb_file, get_traj_charges):
    traj = md.load(raw_alanine_pdb_file)
    output = get_traj_charges.compute_charges_from_traj(traj)
    assert len(output) == traj.n_atoms


def test_compute_dipole_moments(get_registry, dummy_traj_with_box):
    registry = get_registry("raw", True)
    dipole_moments = ComputeDipoleMoments(registry)

    output_expected = np.array(
        [
            [0.9420658, -0.42306627, -3.37293351],
            [-1.95939145, 0.42888127, -1.67964765],
            [5.45737268, -1.75187598, 0.19304526],
            [1.56570501, -0.3875705, 0.01168074],
            [0.05044822, -0.34277508, 2.18897295],
            [-1.54375403, -1.86517372, -0.24699889],
            [1.64506076, 0.50568743, -3.32181685],
            [-0.36744733, 0.96620585, 1.32034504],
            [-0.05364994, 5.52703055, 2.77790702],
            [1.14024026, 2.18740438, -2.0352943],
        ]
    )

    charges = np.random.randn(dummy_traj_with_box.n_atoms)

    output = dipole_moments._compute_dipole_moments(
        charges=charges, traj=dummy_traj_with_box
    )
    assert np.allclose(output, output_expected, rtol=1e-3)


def test_compute_static_dielectric(loaded_cif_traj, get_registry):
    registry = get_registry("raw", True)
    static_dielectric = ComputeStaticDielectric(registry)
    charges = np.random.randn(loaded_cif_traj.n_atoms)
    output = static_dielectric._compute_static_dielectric(
        charges=charges, traj=loaded_cif_traj, temperature=400
    )
    assert np.isclose(output, 1.0, rtol=1e-3)


def test_compute_isothermal_compressability_kappa_T(dummy_traj_with_box, get_registry):
    registry = get_registry("raw", True)

    isothermal_compressability = ComputeIsothermalCompressabilityKappaT(registry)
    output = isothermal_compressability._compute_isothermal_compressability_kappa_T(
        traj=dummy_traj_with_box, temperature=300
    )
    assert np.allclose(output, 0.0, rtol=1e-3)


def test_compute_density(dummy_traj_with_box, get_registry):
    registry = get_registry("raw", True)
    density = ComputeMassDensity(registry)
    density = density._compute_density(dummy_traj_with_box)
    density_should_be = np.array(
        [
            24.930458,
            24.930458,
            24.930458,
            24.930458,
            24.930458,
            24.930458,
            24.930458,
            24.930458,
            24.930458,
            24.930458,
        ]
    )
    assert np.allclose(density, density_should_be)


def test_stack_data(get_registry):
    registry = get_registry("raw", True)
    test_inputs = [1, 2, 3, 4, 5]
    shold_be = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
    data = ComputeMassDensity(registry)._stack_data(test_inputs)
    assert np.allclose(data, shold_be)
