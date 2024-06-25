import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from langchain.tools import BaseTool

from mdagent.utils import FileType, PathRegistry, load_single_traj


def write_raw_x(
    x: str, values: np.ndarray, traj_id: str, path_registry: PathRegistry
) -> str:
    """
    Writes raw x values to a file and saves the file to the path registry.

    Args:
        x: The name of the analysis tool that produced the values (e.g., "dssp")
        values: The x values to save.
        traj_id: The id of the trajectory the values are associated with.
        path_registry: The path registry to save the file to.

        Returns:
            The file id of the saved file.
    """
    file_name = path_registry.write_file_name(
        FileType.RECORD,
        record_type=x,
    )
    file_id = path_registry.get_fileid(file_name, FileType.RECORD)

    file_path = f"{path_registry.ckpt_records}/{x}_{traj_id}.npy"
    np.save(file_path, values)

    path_registry.map_path(
        file_id,
        file_name,
        description=f"{x} values for trajectory with id: {traj_id}",
    )
    return file_id


class ComputeDSSP(BaseTool):
    name = "ComputeDSSP"
    description = """Compute the DSSP (secondary structure) assignment
    for a protein trajectory. Input is a trajectory file (e.g., .xtc, .
    trr) and an optional topology file (e.g., .pdb, .prmtop). The output
    is an array with the DSSP code for each residue at each time point."""
    path_registry: PathRegistry = PathRegistry.get_instance()
    simplified: bool = True

    def __init__(self, path_registry: PathRegistry, simplified: bool = True):
        super().__init__()
        self.path_registry = path_registry
        self.simplified = simplified

    def _dssp_codes(self) -> list[str]:
        """
        Returns the DSSP codes used by MDTraj. If simplified is True, only
        the codes H, E, and C are used. Otherwise, the full set of codes is
        used."""
        if self.simplified:
            return ["H", "E", "C"]
        return ["H", "B", "E", "G", "I", "T", "S", " "]

    def _dssp_natural_language(self) -> dict[str, str]:
        """
        Returns a dictionary mapping DSSP codes to their natural language
        descriptions. If simplified is True, only the codes H, E, and C are
        used. Otherwise, the full set of codes is used."""
        if self.simplified:
            return {"H": "helix", "E": "strand", "C": "coil"}
        return {
            "H": "alpha helix",
            "B": "beta bridge",
            "E": "extended strand",
            "G": "three helix",
            "I": "five helix",
            "T": "hydrogen bonded turn",
            "S": "bend",
            " ": "loop or irregular",
        }

    def _convert_dssp_counts(self, dssp_counts: dict) -> dict:
        """
        Converts a dictionary of DSSP codes to their counts into a dictionary
        of natural language descriptions to their counts.
        is used.

        Args:
            dssp_counts: A dictionary mapping DSSP codes to their counts.

        Returns:
            A dictionary mapping natural language descriptions to their counts.
        """
        code_to_description = self._dssp_natural_language()

        descriptive_counts = {
            code_to_description[code]: count for code, count in dssp_counts.items()
        }
        return descriptive_counts

    def _summarize_dssp(self, dssp_array: np.ndarray) -> dict[str, int]:
        """
        Summarizes the DSSP assignments for a trajectory. Returns a dictionary
        mapping DSSP codes to their counts.

        Args:
            dssp_array: An array of DSSP codes for each residue at each time point.

        Returns:
            A dictionary mapping natural language descriptions to their counts.
        """
        dssp_codes = self._dssp_codes()
        dssp_dict = {code: 0 for code in dssp_codes}
        for frame in dssp_array:
            for code in frame:
                dssp_dict[code] += 1
        return self._convert_dssp_counts(dssp_dict)

    def _compute_dssp(self, traj: md.Trajectory) -> np.ndarray:
        """
        Computes the DSSP assignments for a trajectory.

        Args:
            traj: The trajectory to compute DSSP assignments for.

        Returns:
            An array of DSSP codes for each residue at each time point.
        """
        return md.compute_dssp(traj, simplified=self.simplified)

    def _run(self, traj_file: str, top_file: str | None = None) -> str:
        """
        Computes the DSSP assignments for a trajectory and saves the results
        to a file.

        Args:
            traj_file: The file id of the trajectory to compute DSSP assignments for.
            top_file: The file id of the topology file to use.

        Returns:
            A summary of the DSSP assignments.
        """
        try:
            traj = load_single_traj(
                path_registry=self.path_registry,
                traj_fileid=traj_file,
                top_fileid=top_file,
            )
            if not traj:
                raise Exception("Trajectory could not be loaded.")
        except Exception as e:
            return str(e)

        dssp_array = self._compute_dssp(traj)
        write_raw_x("dssp", dssp_array, traj_file, self.path_registry)
        summary = self._summarize_dssp(dssp_array)
        return str(summary)

    async def _arun(self, traj_file, top_file):
        """Runs the tool asynchronously."""
        raise NotImplementedError("Async version not implemented")


class ComputeGyrationTensor(BaseTool):
    name = "ComputeGyrationTensor"
    description = """Compute the gyration tensor for each frame in a
    molecular dynamics trajectory. Input is a trajectory file (e.g., .
    xtc, .trr) and an optional topology file (e.g., .pdb, .prmtop).
    The output is an array of gyration tensors for each frame of the
    trajectory."""
    path_registry: PathRegistry = PathRegistry.get_instance()

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _compute_gyration_tensor(self, traj: md.Trajectory) -> np.ndarray:
        """
        Computes the gyration tensor for a trajectory.

        Args:
            traj: The trajectory to compute the gyration tensor for.

        Returns:
            An array of gyration tensors for each frame of the trajectory.
        """
        return md.compute_gyration_tensor(traj)

    def _run(self, traj_file: str, top_file: str | None = None) -> str:
        """
        Computes the gyration tensor for a trajectory and saves the results
        to a file.

        Args:
            traj_file: The file id of the trajectory to compute the gyration tensor for.
            top_file: The file id of the topology file to use.

        Returns:
            A message indicating the success of the computation
        """
        try:
            traj = load_single_traj(
                path_registry=self.path_registry,
                traj_fileid=traj_file,
                top_fileid=top_file,
            )
            if not traj:
                raise Exception("Trajectory could not be loaded.")
        except Exception as e:
            return str(e)

        gyration_tensors = self._compute_gyration_tensor(traj)
        # check if there is 1 frame only
        if traj.n_frames == 1:
            return (
                "Gyration tensor computed for "
                "a single frame, no file saved."
                f"Gyrations tensor: {gyration_tensors}"
            )

        file_id = write_raw_x(
            "gyration_tensor", gyration_tensors, traj_file, self.path_registry
        )
        return "Gyration tensor computed successfully, " f"saved to {file_id}"

    async def _arun(self, traj_file, top_file=None):
        """Runs the tool asynchronously."""
        raise NotImplementedError("Async version not implemented")


def plot_x_over_time(
    x: str, values: np.ndarray, traj_id: str, path_registry: PathRegistry
) -> str:
    """
    Plots the values of x over time and saves the plot to a file.

    Args:
        x: The name of the analysis tool that produced the values (e.g., "dssp")
        values: The x values to plot.
        traj_id: The id of the trajectory the values are associated with.
        path_registry: The path registry to save the file to.

    Returns:
        The file id of the saved file.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(values)
    plt.xlabel("Frame")
    plt.ylabel(x)
    plt.title(f"{x} Over Time")
    plt.grid(True)

    file_name = path_registry.write_file_name(
        FileType.FIGURE,
        file_format="png",
    )
    file_id = path_registry.get_fileid(file_name, FileType.RECORD)

    file_path = f"{path_registry.ckpt_figures}/{x}_over_time_{traj_id}.png"
    plt.savefig(file_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()

    path_registry.map_path(
        file_id,
        file_name,
        description=(f"{x} plot for trajectory " f"with id: {traj_id}"),
    )
    return file_id


class ComputeAsphericity(BaseTool):
    name = "ComputeAsphericity"
    description = """Compute the asphericity for each frame in a
    molecular dynamics trajectory. Input is a trajectory file (e.g., .
    xtc, .trr) and an optional topology file (e.g., .pdb, .prmtop).
    The output is asphericity values for each frame of the
    trajectory."""
    path_registry: PathRegistry = PathRegistry.get_instance()

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _compute_asphericity(self, traj: md.Trajectory) -> np.ndarray:
        """
        Computes the asphericity for a trajectory.

        Args:
            traj: The trajectory to compute the asphericity for.

        Returns:
            An array of asphericity values for each frame of the trajectory."""
        return md.asphericity(traj)

    def _run(self, traj_file: str, top_file: str | None = None) -> str:
        """
        Computes the asphericity for a trajectory and saves the results
        to a file.

        Args:
            traj_file: The file id of the trajectory to compute the asphericity for.
            top_file: The file id of the topology file to use.

        Returns:
            A message indicating the success of the computation.
        """
        try:
            traj = load_single_traj(
                path_registry=self.path_registry,
                traj_fileid=traj_file,
                top_fileid=top_file,
            )
            if not traj:
                raise Exception("Trajectory could not be loaded.")
        except Exception as e:
            return str(e)
        asphericity_values = self._compute_asphericity(traj)
        if traj.n_frames == 1:
            return (
                "Asphericity computed for "
                "a single frame, no file saved."
                f"Asphericity: {asphericity_values}"
            )
        raw_file_id = write_raw_x(
            "asphericity", asphericity_values, traj_file, self.path_registry
        )
        plot_file_id = plot_x_over_time(
            "Asphericity", asphericity_values, traj_file, self.path_registry
        )
        return (
            "asphericity_values saved to "
            f"{raw_file_id}, plot saved to "
            f"{plot_file_id}"
        )

    async def _arun(self, traj_file, top_file):
        """Runs the tool asynchronously."""
        raise NotImplementedError("Async version not implemented")


class ComputeAcylindricity(BaseTool):
    name = "ComputeAcylindricity"
    description = """Compute the acylindricity for each frame in a
    molecular dynamics trajectory. Input is a trajectory file (e.g., .
    xtc, .trr) and an optional topology file (e.g., .pdb, .prmtop). The
    output is an array of acylindricity values for each frame of the
    trajectory."""
    path_registry: PathRegistry = PathRegistry.get_instance()

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _compute_acylindricity(self, traj: md.Trajectory) -> np.ndarray:
        """
        Computes the acylindricity for a trajectory.

        Args:
            traj: The trajectory to compute the acylindricity for.

        Returns:
            An array of acylindricity values for each frame of the trajectory.
        """
        return md.acylindricity(traj)

    def _run(self, traj_file: str, top_file: str | None = None) -> str:
        """
        Computes the acylindricity for a trajectory and saves the results
        to a file.

        Args:
            traj_file: The file id of the trajectory to compute the acylindricity for.
            top_file: The file id of the topology file to use.

        Returns:
            A message indicating the success of the computation.
        """
        try:
            traj = load_single_traj(
                path_registry=self.path_registry,
                traj_fileid=traj_file,
                top_fileid=top_file,
            )
            if not traj:
                raise Exception("Trajectory could not be loaded.")
        except Exception as e:
            return str(e)
        acylindricity_values = self._compute_acylindricity(traj)
        if traj.n_frames == 1:
            return (
                "Acylindricity computed for "
                "a single frame, no file saved."
                f"Acylindricity: {acylindricity_values}"
            )
        raw_file_id = write_raw_x(
            "acylindricity", acylindricity_values, traj_file, self.path_registry
        )
        plot_file_id = plot_x_over_time(
            "acylindricity", acylindricity_values, traj_file, self.path_registry
        )
        return (
            "acylindricity_values saved to "
            f"{raw_file_id}, plot saved to "
            f"{plot_file_id}"
        )

    async def _arun(self, traj_file, top_file):
        """Runs the tool asynchronously."""
        raise NotImplementedError("Async version not implemented")


class ComputeRelativeShapeAntisotropy(BaseTool):
    name = "ComputeRelativeShapeAntisotropy"
    description = """Compute the relative shape antisotropy for each
    frame in a molecular dynamics trajectory. Input is a trajectory
    file (e.g., .xtc, .trr) and an optional topology file (e.g., .pdb, .
    prmtop). The output is an array of relative shape antisotropy values
    for each frame of the trajectory."""
    path_registry: PathRegistry = PathRegistry.get_instance()

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _compute_relative_shape_antisotropy(self, traj: md.Trajectory) -> np.ndarray:
        """
        Computes the relative shape antisotropy for a trajectory.

        Args:
            traj: The trajectory to compute the relative shape antisotropy for.

        Returns:
            An array of relative shape antisotropy
                values for each frame of the trajectory.
        """
        return md.relative_shape_antisotropy(traj)

    def _run(self, traj_file: str, top_file: str | None = None) -> str:
        """
        Computes the relative shape antisotropy for a trajectory and saves the results
        to a file.

        Args:
            traj_file: The file id of the trajectory to
                compute the relative shape antisotropy for.
            top_file: The file id of the topology file to use.

        Returns:
            A message indicating the success of the computation.
        """
        try:
            traj = load_single_traj(
                path_registry=self.path_registry,
                traj_fileid=traj_file,
                top_fileid=top_file,
            )
            if not traj:
                raise Exception("Trajectory could not be loaded.")
        except Exception as e:
            return str(e)
        relative_shape_antisotropy_values = self._compute_relative_shape_antisotropy(
            traj
        )
        if traj.n_frames == 1:
            return (
                "Relative shape antisotropy computed for "
                "a single frame, no file saved."
                f"Relative shape antisotropy: {relative_shape_antisotropy_values}"
            )

        raw_file_id = write_raw_x(
            "relative_shape_antisotropy",
            relative_shape_antisotropy_values,
            traj_file,
            self.path_registry,
        )
        plot_file_id = plot_x_over_time(
            "relative_shape_antisotropy",
            relative_shape_antisotropy_values,
            traj_file,
            self.path_registry,
        )
        return (
            "relative_shape_antisotropy_values saved to "
            f"{raw_file_id}, plot saved to "
            f"{plot_file_id}"
        )

    async def _arun(self, traj_file, top_file):
        """Runs the tool asynchronously."""
        raise NotImplementedError("Async version not implemented")


class AnalyzeProteinStructure(BaseTool):
    name = "AnalyzeProtein"
    description = (
        "Analyze a protein trajectory. "
        "Input is a trajectory file "
        "and an optional topology file, "
        "along with a list of analyses to perform."
        "enter the analyses you want to perform as "
        "a string, separated by commas. "
        "The output is a dictionary "
        "containing the requested analyses."
    )
    path_registry: PathRegistry = PathRegistry.get_instance()

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def analyze_protein(self, traj, requested_analyses: list):
        result = {}
        if "n_atoms" in requested_analyses:
            result["n_atoms"] = traj.n_atoms
        if "n_residues" in requested_analyses:
            result["n_residues"] = traj.n_residues
        if "n_chains" in requested_analyses:
            result["n_chains"] = traj.n_chains
        if "n_frames" in requested_analyses:
            result["n_frames"] = traj.n_frames
        if "time" in requested_analyses:
            result["time"] = traj.time
        if "time_step" in requested_analyses:
            result["time_step"] = traj.time_step
        if "atoms" in requested_analyses:
            result["atoms"] = traj.topology.atoms
        if "bonds" in requested_analyses:
            result["bonds"] = traj.topology.bonds
        if "chains" in requested_analyses:
            result["chains"] = traj.topology.chains
        if "residues" in requested_analyses:
            result["residues"] = traj.topology.residues
        return result

    def _run(
        self, traj_file: str, requested_analyses: str, top_file: str | None = None
    ) -> str:
        try:
            traj = load_single_traj(
                path_registry=self.path_registry,
                traj_fileid=traj_file,
                top_fileid=top_file,
            )
            if not traj:
                raise Exception("Trajectory could not be loaded.")
        except Exception as e:
            return str(e)
        requested_analyses_list = requested_analyses.split(",")
        result = self.analyze_protein(traj, requested_analyses_list)
        return str(result)
