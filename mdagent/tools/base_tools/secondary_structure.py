import mdtraj as md
import numpy as np
from langchain.tools import BaseTool

from mdagent.utils import FileType, PathRegistry, load_single_traj


class ComputeDSSP(BaseTool):
    name = "ComputeDSSP"
    description = """Compute the DSSP (secondary structure) assignment
    for a protein trajectory. Input is a trajectory file (e.g., .xtc, .
    trr) and an optional topology file (e.g., .pdb, .prmtop). The output
    is an array with the DSSP code for each residue at each time point."""
    path_registry: PathRegistry | None = None
    simplified: bool = True

    def __init__(self, path_registry, simplified):
        super().__init__()
        self.path_registry = path_registry
        self.simplified = simplified

    def _dssp_codes(self):
        if self.simplified:
            return ["H", "E", "C"]
        return ["H", "B", "E", "G", "I", "T", "S", " "]

    def _dssp_natural_language(self):
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

    def _convert_dssp_counts(self, dssp_counts: dict):
        code_to_description = self._dssp_natural_language()

        descriptive_counts = {
            code_to_description[code]: count for code, count in dssp_counts.items()
        }
        return descriptive_counts

    def _summarize_dssp(self, dssp_array):
        dssp_codes = self._dssp_codes()
        # turn into dict where keys are codes
        dssp_dict = {code: 0 for code in dssp_codes}
        for frame in dssp_array:
            for code in frame:
                dssp_dict[code] += 1
        return self._convert_dssp_counts(dssp_dict)

    def _write_raw_dssp(self, dssp_array, traj_id):
        file_name = self.path_registry.write_file_name(
            FileType.RECORD,
            record_type="dssp",
        )
        file_id = self.path_registry.get_fileid(file_name, FileType.RECORD)

        file_path = f"{self.path_registry.ckpt_records}/dssp_{traj_id}.npy"
        np.save(file_path, dssp_array)
        
        self.path_registry.map_path(
            file_id,
            file_name,
            description=f"DSSP codes for trajectory with id: {traj_id}",
        )
        return None

    def _compute_dssp(self, traj):
        return md.compute_dssp(traj, simplified=self.simplified)

    def _run(self, traj_file, top_file=None):
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
        self._write_raw_dssp(dssp_array, traj_file)
        summary = self._summarize_dssp(dssp_array)
        return summary

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")


class ComputeGyrationTensor(BaseTool):
    name = "ComputeGyrationTensor"
    description = """Compute the gyration tensor for each frame in a
    molecular dynamics trajectory. Input is a trajectory file (e.g., .
    xtc, .trr) and an optional topology file (e.g., .pdb, .prmtop).
    The output is an array of gyration tensors for each frame of the
    trajectory."""
    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _write_raw_rgy(self, gyration_tensors, traj_id):
        file_name = self.path_registry.write_file_name(
            FileType.RECORD,
            record_type="gyration_tensor",
        )
        file_id = self.path_registry.get_fileid(file_name, FileType.RECORD)

        file_path = f"{self.path_registry.ckpt_records}/gyration_tensors_{traj_id}.npy"
        np.save(file_path, gyration_tensors)
        
        self.path_registry.map_path(
            file_id,
            file_name,
            description=f"gyration tensors for trajectory with id: {traj_id}",
        )
        return file_id
    
    def _compute_gyration_tensor(self, traj):
        return md.compute_gyration_tensor(traj)

    def _run(self, traj_file, top_file=None):
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
        file_id = self._write_raw_rgy(gyration_tensors, traj_file)
        return ("Gyration tensor computed successfully, "
                f"saved to {file_id}")

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")


class ComputePrincipleMoments(BaseTool):
    name = "ComputePrincipleMoments"
    description = """Compute the principle moments of inertia for each
      frame in a molecular dynamics trajectory. Input is a trajectory
      file (e.g., .xtc, .trr) and an optional topology file (e.g., .pdb, .
      prmtop). The output is an array of principle moments of inertia for
      each frame of the trajectory."""
    # TODO -> should this write to a file or return the array?
    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        traj = load_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Trajectory could not be loaded."
        return md.compute_principal_moments(traj)

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")


class ComputeAsphericity(BaseTool):
    name = "ComputeAsphericity"
    description = """Compute the asphericity for each frame in a
    molecular dynamics trajectory. Input is a trajectory file (e.g., .
    xtc, .trr) and an optional topology file (e.g., .pdb, .prmtop).
    The output is an array of asphericity values for each frame of the
    trajectory."""
    # TODO -> should this write to a file or return the array?
    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        traj = load_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Trajectory could not be loaded."
        return md.asphericity(traj)

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")


class ComputeAcylindricity(BaseTool):
    name = "ComputeAcylindricity"
    description = """Compute the acylindricity for each frame in a
    molecular dynamics trajectory. Input is a trajectory file (e.g., .
    xtc, .trr) and an optional topology file (e.g., .pdb, .prmtop). The
    output is an array of acylindricity values for each frame of the
    trajectory."""
    # TODO -> should this write to a file or return the array?
    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        traj = load_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Trajectory could not be loaded."
        return md.acylindricity(traj)

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")


class ComputeRelativeShapeAntisotropy(BaseTool):
    name = "ComputeRelativeShapeAntisotropy"
    description = """Compute the relative shape antisotropy for each
    frame in a molecular dynamics trajectory. Input is a trajectory
    file (e.g., .xtc, .trr) and an optional topology file (e.g., .pdb, .
    prmtop). The output is an array of relative shape antisotropy values
    for each frame of the trajectory."""
    # TODO -> should this write to a file or return the array?
    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        traj = load_traj(self.path_registry, traj_file, top_file)
        if not traj:
            return "Trajectory could not be loaded."
        return md.relative_shape_antisotropy(traj)

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")
