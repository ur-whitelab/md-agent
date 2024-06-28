import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from langchain.tools import BaseTool

from mdagent.utils import PathRegistry, load_single_traj


class ComputeAngles(BaseTool):
    name = "compute_angles"
    description = """Calculate the bond angles for the given sets of three atoms in
    each snapshot, and provide a list of indices specifying which atoms are involved
    in each bond angle calculation.."""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file: str, angle_indices: list, top_file: str | None = None):
        try:
            traj = load_single_traj(self.path_registry, traj_file, top_file)
            if not traj:
                return "Failed. Trajectory could not be loaded."

            if (
                not angle_indices
                or not isinstance(angle_indices, list)
                or not all(len(indices) == 3 for indices in angle_indices)
            ):
                return (
                    "Failed. Invalid angle_indices. It should be a list of tuples, "
                    "each containing three atom indices."
                )

            angles = md.compute_angles(traj, angle_indices, periodic=True, opt=True)

            # Check if path_registry is not None
            if self.path_registry is not None:
                plot_save_path = self.path_registry.get_mapped_path("angles_plot.png")
                plot_angles(angles, title="Bond Angles", save_path=plot_save_path)
                return "Succeeded. Bond angles computed, saved to file and plot saved."
            else:
                return "Failed. Path registry is not initialized."

        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

    async def _arun(
        self,
        traj_file: str,
        angle_indices: list,
        top_file: str | None = None,
    ):
        raise NotImplementedError("Async version not implemented")


class ComputeDihedrals(BaseTool):
    name = "compute_dihedrals"
    description = """Calculate the dihedral angles for the given groups of four atoms
      in each snapshot, and provide a list of dihedral angles along with a list of
      indices specifying which atoms are involved in each dihedral angle calculation."""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file: str, indices: list, top_file: str | None = None):
        try:
            traj = load_single_traj(self.path_registry, traj_file, top_file)
            if not traj:
                return "Failed. Trajectory could not be loaded."

            if (
                not indices
                or not isinstance(indices, list)
                or not all(
                    isinstance(tup, tuple) and all(isinstance(i, int) for i in tup)
                    for tup in indices
                )
            ):
                return "Failed. Invalid indices. It should be a list of tuples."

            dihedrals = md.compute_dihedrals(traj, indices, periodic=True, opt=True)

            # Check if path_registry is not None
            if self.path_registry is not None:
                plot_save_path = self.path_registry.get_mapped_path(
                    "dihedrals_plot.png"
                )
                plot_angles(
                    dihedrals, title="Dihedral Angles", save_path=plot_save_path
                )
                return (
                    "Succeeded. Dihedral angles computed, saved to file and plot saved."
                )
            else:
                return "Failed. Path registry is not initialized."

        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

    async def _arun(self, traj_file: str, indices: list, top_file: str | None = None):
        raise NotImplementedError("Async version not implemented")


class ComputePhi(BaseTool):
    name = "compute_phi"
    description = """This class calculates phi torsion angles and provides a list of phi
      angles and indices specifying which atoms are involved in the calculations"""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file: str, top_file: str | None = None):
        try:
            traj = load_single_traj(self.path_registry, traj_file, top_file)
            if not traj:
                return "Failed. Trajectory could not be loaded."

            indices, angles = md.compute_phi(traj, periodic=True, opt=True)

            # Check if path_registry is initialized
            if self.path_registry is not None:
                # Save results to a file
                save_results_to_file("phi_results.npz", indices, angles)

                # Generate and save a plot
                plot_save_path = self.path_registry.get_mapped_path("phi_plot.png")
                plot_angles(angles, title="Phi Angles", save_path=plot_save_path)

                # Return success message
                return "Succeeded. Phi angles computed, saved to file and plot saved."
            else:
                # Return failure message if path_registry is not initialized
                return "Failed. Path registry is not initialized."

        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

    async def _arun(self, traj_file: str, top_file: str | None = None):
        raise NotImplementedError("Async version not implemented")


class ComputePsi(BaseTool):
    name = "compute_psi"
    description = """Calculate the psi angles for each snapshot, providing a list of
    psi angles for each frame in the trajectory and a list of indices specifying the
    atoms involved in calculating each psi angle"""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file: str, top_file: str | None = None):
        try:
            traj = load_single_traj(self.path_registry, traj_file, top_file)
            if not traj:
                return "Failed. Trajectory could not be loaded."

            indices, angles = md.compute_psi(traj, periodic=True, opt=True)

            # Check if path_registry is initialized
            if self.path_registry is not None:
                # Save results to a file
                save_results_to_file("psi_results.npz", indices, angles)

                # Generate and save a plot
                plot_save_path = self.path_registry.get_mapped_path("psi_plot.png")
                plot_angles(angles, title="Psi Angles", save_path=plot_save_path)

                # Return success message
                return "Succeeded. Psi angles computed, saved to file and plot saved."
            else:
                # Return failure message if path_registry is not initialized
                return "Failed. Path registry is not initialized."

        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

    async def _arun(self, traj_file: str, top_file: str | None = None):
        raise NotImplementedError("Async version not implemented")


class ComputeChi1(BaseTool):
    name = "compute_chi1"
    description = """Calculate the chi1 angles (the first side chain torsion angle
    formed between four atoms around the CA-CB axis) for each snapshot, providing a
    list of chi1 angles and indices specifying the atoms involved in each chi1 angle
    calculation."""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file: str, top_file: str | None = None):
        try:
            traj = load_single_traj(self.path_registry, traj_file, top_file)
            if not traj:
                return "Failed. Trajectory could not be loaded."

            indices, angles = md.compute_chi1(traj, periodic=True, opt=True)

            # Check if path_registry is initialized
            if self.path_registry is not None:
                # Save results to a file
                save_results_to_file("chi1_results.npz", indices, angles)

                # Generate and save a plot
                plot_save_path = self.path_registry.get_mapped_path("chi1_plot.png")
                plot_angles(angles, title="Chi1 Angles", save_path=plot_save_path)

                # Return success message
                return "Succeeded. chi1 angles computed, saved to file and plot saved."
            else:
                # Return failure message if path_registry is not initialized
                return "Failed. Path registry is not initialized."

        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")


class ComputeChi2(BaseTool):
    name = "compute_chi2"
    description = """Calculate the chi2 angles (the second side chain torsion angle
    formed between four atoms around the CB-CG axis) for each snapshot, providing a
    list of chi2 angles and a list of indices specifying the atoms involved in
    calculating each chi2 angle."""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file: str, top_file: str | None = None):
        try:
            traj = load_single_traj(self.path_registry, traj_file, top_file)
            if not traj:
                return "Failed. Trajectory could not be loaded."

            indices, angles = md.compute_chi2(traj, periodic=True, opt=True)

            # Check if path_registry is initialized
            if self.path_registry is not None:
                # Save results to a file
                save_results_to_file("chi2_results.npz", indices, angles)

                # Generate and save a plot
                plot_save_path = self.path_registry.get_mapped_path("chi2_plot.png")
                plot_angles(angles, title="Chi2 Angles", save_path=plot_save_path)

                # Return success message
                return "Succeeded. chi2 angles computed, saved to file and plot saved."
            else:
                # Return failure message if path_registry is not initialized
                return "Failed. Path registry is not initialized."

        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

    async def _arun(self, traj_file: str, top_file: str | None = None):
        raise NotImplementedError("Async version not implemented")


class ComputeChi3(BaseTool):
    name = "compute_chi3"
    description = """Calculate the chi3 angles (the third side chain torsion angle
    formed between four atoms around the CG-CD axis) for each snapshot in the
    trajectory, providing a list of chi3 angles and indices specifying the atoms
    involved in the calculation of each chi3 angle. Note: Only the residues ARG, GLN,
      GLU, LYS, and MET have these atoms."""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file: str, top_file: str | None = None):
        try:
            traj = load_single_traj(self.path_registry, traj_file, top_file)
            if not traj:
                return "Failed. Trajectory could not be loaded."

            indices, angles = md.compute_chi3(traj, periodic=True, opt=True)

            # Check if path_registry is initialized
            if self.path_registry is not None:
                # Save results to a file
                save_results_to_file("chi3_results.npz", indices, angles)

                # Generate and save a plot
                plot_save_path = self.path_registry.get_mapped_path("chi3_plot.png")
                plot_angles(angles, title="Chi3 Angles", save_path=plot_save_path)

                # Return success message
                return "Succeeded. chi3 angles computed, saved to file and plot saved."
            else:
                # Return failure message if path_registry is not initialized
                return "Failed. Path registry is not initialized."

        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

    async def _arun(self, traj_file: str, top_file: str | None = None):
        raise NotImplementedError("Async version not implemented")


class ComputeChi4(BaseTool):
    name = "compute_chi4"
    description = """Calculate the chi4 angles (the fourth side chain torsion angle
    formed between four atoms around the CD-CE or CD-NE axis) for each snapshot in the
    trajectory, providing a list of indices specifying which atoms are involved in the
    chi4 angle calculations. """

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file: str, top_file: str | None = None):
        try:
            traj = load_single_traj(self.path_registry, traj_file, top_file)
            if not traj:
                return "Failed. Trajectory could not be loaded."

            indices, angles = md.compute_chi4(traj, periodic=True, opt=True)

            # Check if path_registry is initialized
            if self.path_registry is not None:
                # Save results to a file
                save_results_to_file("chi4_results.npz", indices, angles)

                # Generate and save a plot
                plot_save_path = self.path_registry.get_mapped_path("chi4_plot.png")
                plot_angles(angles, title="Chi4 Angles", save_path=plot_save_path)

                # Return success message
                return "Succeeded. chi4 angles computed, saved to file and plot saved."
            else:
                # Return failure message if path_registry is not initialized
                return "Failed. Path registry is not initialized."

        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

    async def _arun(self, traj_file: str, top_file: str | None = None):
        raise NotImplementedError("Async version not implemented")


class ComputeOmega(BaseTool):
    name = "compute_omega"
    description = """Calculate the omega angles for each snapshot in the trajectory,
    providing a list of indices specifying which atoms are involved in the omega angle
    calculations.."""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file: str, top_file: str | None = None):
        try:
            traj = load_single_traj(self.path_registry, traj_file, top_file)
            if not traj:
                return "Failed. Trajectory could not be loaded."

            indices, angles = md.compute_omega(traj, periodic=True, opt=True)

            # Check if path_registry is initialized
            if self.path_registry is not None:
                # Save results to a file
                save_results_to_file("omega_results.npz", indices, angles)

                # Generate and save a plot
                plot_save_path = self.path_registry.get_mapped_path("omega_plot.png")
                plot_angles(angles, title="Omega Angles", save_path=plot_save_path)

                # Return success message
                return "Succeeded. omega angles computed, saved to file and plot saved."
            else:
                # Return failure message if path_registry is not initialized
                return "Failed. Path registry is not initialized."

        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

    async def _arun(self, traj_file: str, top_file: str | None = None):
        raise NotImplementedError("Async version not implemented")


class RamachandranPlot(BaseTool):
    name = "ramachandran_plot"
    description = """Generate a Ramachandran plot for the given trajectory, showing
    the distribution of phi and psi angles for each frame."""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file: str, top_file: str | None = None):
        try:
            traj = load_single_traj(self.path_registry, traj_file, top_file)
            if not traj:
                return "Failed. Trajectory could not be loaded."

            phi_indices, phi_angles = md.compute_phi(traj, periodic=True, opt=True)
            psi_indices, psi_angles = md.compute_psi(traj, periodic=True, opt=True)

            # Map indices to residues for further analysis or reporting
            map_indices_to_residues(traj, phi_indices)
            map_indices_to_residues(traj, psi_indices)

            # Check if path_registry is not None
            if self.path_registry is not None:
                plot_save_path = self.path_registry.get_mapped_path(
                    "ramachandran_plot.png"
                )
                plt.figure(figsize=(10, 8))
                plt.scatter(
                    phi_angles.flatten(), psi_angles.flatten(), s=1, color="blue"
                )
                plt.xlabel("Phi Angles (radians)")
                plt.ylabel("Psi Angles (radians)")
                plt.title("Ramachandran Plot")
                plt.grid(True)
                print(f"Saving plot to: {plot_save_path}")
                plt.savefig(plot_save_path)
                print(f"Ramachandran plot saved to: {plot_save_path}")
                return "Succeeded. Ramachandran plot generated and saved to file."
            else:
                return "Failed. Path registry is not initialized."

        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

    async def _arun(self, traj_file: str, top_file: str | None = None):
        raise NotImplementedError("Async version not implemented")


# Helper functions suggested by Jorge
def map_indices_to_residues(traj, indices):
    atom_to_residue = {atom.index: atom.residue for atom in traj.topology.atoms}
    residues_per_angle = [
        [atom_to_residue[idx] for idx in angle_set] for angle_set in indices
    ]
    return residues_per_angle


def save_results_to_file(filename, indices, angles):
    np.savez(filename, indices=indices, angles=angles)


def plot_angles(angles, title="Angles", save_path=None):
    print(f"Save path received: {save_path}")  # Debugging help
    plt.figure(figsize=(10, 8))
    for angle_set in angles.T:
        plt.plot(angle_set, label="Angle")
    plt.xlabel("Frame")
    plt.ylabel("Angle (radians)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        print(f"Calling savefig with path: {save_path}")  # Debugging help
        plt.savefig(save_path)
    else:
        plt.show()
