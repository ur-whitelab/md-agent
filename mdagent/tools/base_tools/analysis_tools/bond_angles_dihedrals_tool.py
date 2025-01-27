from typing import Optional

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from mdagent.utils import FileType, PathRegistry, load_single_traj


class ComputingAnglesSchema(BaseModel):
    trajectory_fileid: str = Field(
        description="Trajectory File ID of the simulation to be analyzed"
    )
    topology_fileid: str = Field(
        description=("Topology File ID of the simulation to be analyzed")
    )
    analysis: str = Field(
        "all",
        description=(
            "Which analysis to be done. Availables are: "
            "phi-psi (saves a Ramachandran plot and histograms for the Phi-Psi angles),"
            "chis (gets the chis 1-4 angles and plots a time evolutiuon plot for all"
            "residues is saved. For the plots it only uses sidechains with enough "
            "carbons), all (makes all of the previous analysis)"
        ),
    )
    # This arg is here, but is not used in the code. As of now it will get the analysis
    # of all the residues in the simulation
    selection: Optional[str] = Field(
        "backbone and sidechain",
        description=(
            "Which selection of atoms from the simulation "
            "to use for the pca analysis"
        ),
    )


class ComputeAngles(BaseTool):
    name = "ComputeAngles"
    description = """Analyze dihedral angles from a trajectory file. The tool allows for
    analysis of the phi-psi angles, chi1-chi2 angles, or both. """

    path_registry: PathRegistry | None = None
    args_schema = ComputingAnglesSchema

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, input):

        try:
            input = self.validate_input(**input)

        except ValueError as e:
            return f"Failed. Error using the ComputeAngle Tool: {str(e)}"

        (
            traj_id,
            top_id,
            analysis,
            selection,
            error,
            system_input_message,
        ) = self.get_values(input)

        if error:
            return f"Failed. Error with the tool inputs: {error} "
        if system_input_message == "Tool Messages:":
            system_input_message = ""

        try:
            traj = load_single_traj(
                self.path_registry,
                top_id,
                traj_fileid=traj_id,
                traj_required=True,
            )
        except ValueError as e:
            if (
                "The topology and the trajectory files might not\
                  contain the same atoms"
                in str(e)
            ):
                return (
                    "Failed. Error loading trajectory. Make sure the topology file"
                    " is from the initial positions of the trajectory. Error: {str(e)}"
                )
            return f"Failed. Error loading trajectory: {str(e)}"
        except OSError as e:
            if (
                "The topology is loaded by filename extension, \
                and the detected"
                in str(e)
            ):
                return (
                    "Failed. Error loading trajectory. Make sure you include the"
                    "correct file for the topology. Supported extensions are:"
                    "'.pdb', '.pdb.gz', '.h5', '.lh5', '.prmtop', '.parm7', '.prm7',"
                    "  '.psf', '.mol2', '.hoomdxml', '.gro', '.arc', '.hdf5' and '.gsd'"
                )
            return f"Failed. Error loading trajectory: {str(e)}"
        except Exception as e:
            return f"Failed. Error loading trajectory: {str(e)}"

        return self.analyze_trajectory(traj, analysis, self.path_registry, traj_id)

    async def _arun(self, input):
        raise NotImplementedError("Async version not implemented")

    # Example helper functions (optional). You can instead just keep them as
    # blocks in the if-statements.
    def compute_and_plot_phi_psi(self, traj, path_registry, sim_id):
        """
        Computes phi-psi angles, saves results to file, and produces Ramachandran plot.
        """
        try:
            # Compute phi and psi angles
            phi_indices, phi_angles = md.compute_phi(traj)
            psi_indices, psi_angles = md.compute_psi(traj)

            # Convert angles to degrees
            phi_angles = phi_angles * (180.0 / np.pi)
            psi_angles = psi_angles * (180.0 / np.pi)
        except Exception as e:
            return None, f"Failed. Error computing phi-psi angles: {str(e)}"

        # If path_registry is available, save files and produce plot
        if path_registry is not None:
            # Save angle results
            save_results_to_file("phi_results.npz", phi_indices, phi_angles)
            save_results_to_file("psi_results.npz", psi_indices, psi_angles)

            # Make Ramachandran plot
            try:
                plt.hist2d(
                    phi_angles.flatten(), psi_angles.flatten(), bins=150, cmap="Blues"
                )
                plt.xlabel(r"$\phi$")
                plt.ylabel(r"$\psi$")
                plt.colorbar()

                file_name = path_registry.write_file_name(
                    FileType.FIGURE,
                    fig_analysis="ramachandran",
                    file_format="png",
                    Sim_id=sim_id,
                )
                desc = f"Ramachandran plot for the simulation {sim_id}"
                plot_id = path_registry.get_fileid(file_name, FileType.FIGURE)
                path = path_registry.ckpt_dir + "/figures/"
                plt.savefig(path + file_name)
                path_registry.map_path(plot_id, path + file_name, description=desc)
                plt.clf()  # Clear the current figure so it does not overlay next plot
                print("Ramachandran plot saved to file")
                return plot_id, "Succeeded. Ramachandran plot saved."
            except Exception as e:
                return None, f"Failed. Error saving Ramachandran plot: {str(e)}"
        else:
            return (
                None,
                "Succeeded. Computed phi-psi angles (no path_registry to save).",
            )

    def classify_chi(self, ang_deg, res_name=""):
        """Return an integer code depending on angle range."""
        # Example classification with made-up intervals:
        if res_name == "PRO" or res_name == "P":
            if ang_deg < 0:
                return 3  # e.g. "p-"
            else:
                return 4  # e.g. "p+"
        # angles for g+
        if 0 <= ang_deg < 120:
            return 0  # e.g. "g+"
        # angles for t
        elif -120 >= ang_deg or ang_deg > 120:
            return 1  # e.g. "t"
        # angles for g-
        elif -120 <= ang_deg < 0:
            return 2  # e.g. "g-"

    # function that takes an array and classifies the angles
    def classify_chi_angles(self, angles, res_name=""):
        return [self.classify_chi(ang, res_name) for ang in angles]

    def _plot_one_chi_angle(self, ax, angle_array, residue_names, title=None):
        """
        Classify angles per residue/frame, then do imshow on a given Axes.
        angle_array: shape (n_frames, n_residues) or (n_residues, n_frames)
        residue_names: length n_residues
        """
        state_sequence = np.array(
            [
                [self.classify_chi_angles(a, str(name)[:3])]
                for i, (a, name) in enumerate(zip(angle_array.T, residue_names))
            ]
        )
        states_per_res = state_sequence.reshape(
            state_sequence.shape[0], state_sequence.shape[2]
        )  # shape = (#res,1, #frames)
        # -> (#res, #frames)

        n_residues = len(residue_names)
        unique_states = np.unique(states_per_res)
        n_states = len(unique_states)
        cmap = plt.get_cmap("tab20", n_states)

        im = ax.imshow(
            states_per_res,
            aspect="auto",
            interpolation="none",
            cmap=cmap,
            origin="upper",
        )

        ax.set_xlabel("Frame index")
        ax.set_ylabel("Residue")
        if title:
            ax.set_title(title)

        ax.set_yticks(np.arange(n_residues))
        ax.set_yticklabels([str(r) for r in residue_names], fontsize=8)

        cbar = plt.colorbar(im, ax=ax, ticks=range(n_states), pad=0.01)

        # Example state -> label mapping
        state_labels_map = {0: "g+", 1: "t", 2: "g-", 3: "Cγ endo", 4: "Cγ exo"}
        tick_labels = [state_labels_map.get(s, f"State {s}") for s in unique_states]
        cbar.ax.set_yticklabels(tick_labels, fontsize=8)

    ###################################################
    # Main function to produce a single figure w/ 4 subplots
    ###################################################
    def compute_plot_all_chi_angles(self, traj, sim_id="sim"):
        """
        Create one figure with 4 subplots (2x2):
        - subplot(0,0): χ1
        - subplot(0,1): χ2
        - subplot(1,0): χ3
        - subplot(1,1): χ4
        """
        chi1_indices, chi_1_angles = md.compute_chi1(traj)
        chi2_indices, chi_2_angles = md.compute_chi2(traj)
        chi3_indices, chi_3_angles = md.compute_chi3(traj)
        chi4_indices, chi_4_angles = md.compute_chi4(traj)

        chi_1_angles_degrees = np.rad2deg(chi_1_angles)
        chi_2_angles_degrees = np.rad2deg(chi_2_angles)
        chi_3_angles_degrees = np.rad2deg(chi_3_angles)
        chi_4_angles_degrees = np.rad2deg(chi_4_angles)
        residue_names_1 = [traj.topology.atom(i).residue for i in chi1_indices[:, 1]]
        residue_names_2 = [traj.topology.atom(i).residue for i in chi2_indices[:, 1]]
        residue_names_3 = [traj.topology.atom(i).residue for i in chi3_indices[:, 1]]
        residue_names_4 = [traj.topology.atom(i).residue for i in chi4_indices[:, 1]]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Top-left: χ1
        self._plot_one_chi_angle(
            axes[0, 0], chi_1_angles_degrees, residue_names_1, title=r"$\chi$1"
        )

        # Top-right: χ2
        self._plot_one_chi_angle(
            axes[0, 1], chi_2_angles_degrees, residue_names_2, title="$\chi$2"
        )

        # Bottom-left: χ3
        self._plot_one_chi_angle(
            axes[1, 0], chi_3_angles_degrees, residue_names_3, title="$\chi$3"
        )

        # Bottom-right: χ4
        self._plot_one_chi_angle(
            axes[1, 1], chi_4_angles_degrees, residue_names_4, title="$\chi$4"
        )
        # add title
        fig.suptitle("Chi angles per residue for simulation {sim}", fontsize=16)
        plt.tight_layout()
        # plt.show()
        # Save the figure
        file_name = self.path_registry.write_file_name(
            FileType.FIGURE,
            fig_analysis="chi_angles",
            file_format="png",
            Sim_id=sim_id,
        )
        desc = f"Chi angles plot for the simulation {sim_id}"
        plot_id = self.path_registry.get_fileid(file_name, FileType.FIGURE)
        path = self.path_registry.ckpt_dir + "/figures/"
        plt.savefig(path + file_name)
        self.path_registry.map_path(plot_id, path + file_name, description=desc)
        plt.clf()  # Clear the current figure so it does not overlay next plot
        return plot_id, "Succeeded. Chi angles plot saved."

    def analyze_trajectory(self, traj, analysis, path_registry=None, sim_id="sim"):
        """
        Main function to decide which analysis to do:
        'phi-psi', 'chis', or 'all'.
        """
        # Store optional references for convenience
        self_path_registry = path_registry
        self_sim_id = sim_id

        # ================ PHI-PSI ONLY =================
        if analysis == "phi-psi":
            ram_plot_id, phi_message = self.compute_and_plot_phi_psi(
                traj, self_path_registry, self_sim_id
            )
            return f"Ramachandran plot with ID {ram_plot_id}, message: {phi_message} "

        # ================ CHI1-CHI2 ONLY ================
        elif analysis == "chis":
            chi_plot_id, chi_message = self.compute_plot_all_chi_angles(
                traj, self_path_registry, self_sim_id
            )
            return f"Chis plot with ID {chi_plot_id}, message: {chi_message}"

        # ================ ALL =================
        elif analysis == "all":
            # First do phi-psi
            phi_plot_id, phi_message = self.compute_and_plot_phi_psi(
                traj, self_path_registry, self_sim_id
            )
            if "Failed." in phi_message:
                return phi_message

            # Then do chi1-chi2
            chi_plot_id, chi_message = self.compute_plot_all_chi_angles(
                traj, self_path_registry, self_sim_id
            )
            if "Failed." in chi_message:
                return chi_message

            return (
                "Succeeded. All analyses completed. "
                f"Ramachandran plot with ID {phi_plot_id}, message: {phi_message} "
                f"Chis plot with ID {chi_plot_id}, message: {chi_message}"
            )

        else:
            # Unknown analysis type
            return f"Failed. Unknown analysis type: {analysis}"

    def validate_input(self, **input):
        input = input.get("action_input", input)
        input = input.get("input", input)
        trajectory_id = input.get("trajectory_fileid", None)
        topology_id = input.get("topology_fileid", None)
        analysis = input.get("analysis", "all")
        selection = input.get("selection", "backbone and sidechain")
        if not trajectory_id:
            raise ValueError("Incorrect Inputs: trajectory_fileid is required")
        if not topology_id:
            raise ValueError("Incorrect Inputs: topology_fileid is required")
        # check if trajectory id is valid
        fileids = self.path_registry.list_path_names()
        error = ""
        system_message = "Tool Messages:"
        if trajectory_id not in fileids:
            error += " Trajectory File ID not in path registry"
        if topology_id not in fileids:
            error += " Topology File ID not in path registry"

        if analysis.lower() not in [
            "all",
            "phi-psi",
            "chis",
        ]:
            analysis = "all"
            system_message += (
                " analysis arg not recognized, using analysis = 'all' as default"
            )

        if selection not in [
            "backbone",
            "name CA",
            "backbone and name CA",
            "protein",
            "backbone and sidechain",
            "sidechain",
            "all",
        ]:
            selection = "all"  # just alpha carbons
        # get all the kwargs:
        keys = input.keys()
        for key in keys:
            if key not in [
                "trajectory_fileid",
                "topology_fileid",
                "analysis",
                "selection",
            ]:
                system_message += f"{key} is not part of admitted tool inputs"
        if error == "":
            error = None
        return {
            "trajectory_fileid": trajectory_id,
            "topology_fileid": topology_id,
            "analysis": analysis,
            "selection": selection,
            "error": error,
            "system_message": system_message,
        }

    def get_values(self, input):
        traj_id = input.get("trajectory_fileid")
        top_id = input.get("topology_fileid")
        analysis = input.get("analysis")
        sel = input.get("selection")
        error = input.get("error")
        syst_mes = input.get("system_message")

        return traj_id, top_id, analysis, sel, error, syst_mes


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
