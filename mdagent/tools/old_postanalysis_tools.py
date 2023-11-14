import MDAnalysis as mda
import MDAnalysis.analysis.distances as mda_dist
import numpy as np
from langchain.tools import BaseTool
from MDAnalysis.analysis import rms


def ppi_distance(pdb_file, binding_site="protein"):
    """
    Calculates minimum heavy-atom distance between peptide (assumed to be
    smallest chain) and protein. Returns average distance between these two.

    Can specify binding site if given (optional)
    Can work with any protein-protein interaction (PPI)
    """
    # load and find smallest chain
    u = mda.Universe(pdb_file)
    peptide = None
    for chain in u.segments:
        if peptide is None or len(chain.residues) < len(peptide):
            peptide = chain.residues
    protein = u.select_atoms(
        f"({binding_site}) and not segid {peptide.segids[0]} and not name H*"
    )
    peptide = peptide.atoms.select_atoms("not name H*")
    all_d = []
    for r in peptide.residues:
        distances = mda_dist.distance_array(r.atoms.positions, protein.positions)
        # get row, column of minimum distance
        i, j = np.unravel_index(distances.argmin(), distances.shape)
        all_d.append(distances[i, j])
    avg_d = np.mean(all_d)
    return avg_d


# 1D RMSD, gives one scalar value
def rmsd_compare(
    pdbfile, ref_file, trajectory=None, ref_trajectory=None, selection="backbone"
):
    """take two files (PDB or PSF) and compute RMSD to compare
    the difference between two conformations. If trajectory files for both
    protein of interest and reference are obtained from either user or openmm,
    include these two trajectory files as well."""
    if trajectory is not None:
        u = mda.Universe(pdbfile, trajectory)
        ref = mda.Universe(ref_file, ref_trajectory)
    else:
        u = mda.Universe(pdbfile)
        ref = mda.Universe(ref_file)

    rmsd = rms.rmsd(
        u.select_atoms(selection).positions,
        ref.select_atoms(selection).positions,
        center=True,
        superposition=True,
    )
    return rmsd


# 1D time-dependent RMSD, gives one scalar value for each timestep
def rmsd_overtime(pdbfile, trajectory, selection="backbone", pdbid=None):
    """take two files: 1) topology in form of PDB or PSF file and
    2) trajectory file from openmm simulation. It computes RMSD for each of
    trajectory frames compared to the reference, which is the initial frame.
    It stores RMSD array in a created file."""
    u = mda.Universe(pdbfile, trajectory)
    R = rms.RMSD(u, select=selection)
    R.run()
    if pdbid is not None:
        filename = f"rmsd_{pdbid}.csv"
    else:
        filename = "rmsd_data.csv"
    np.savetxt(
        filename,
        R.results.rmsd,
        fmt=["%d", "%f", "%f"],
        delimiter=",",
        header="Frame,Time,RMSD",
        comments="",
    )
    print("Calculated RMSD for each timestep with respect to the initial frame.")
    # avg_rmsd = np.mean(R.results.rmsd[2])  # rmsd values are in 3rd column
    # print(f"Average RMSD is {avg_rmsd}.")
    # final_rmsd = R.results.rmsd[2][-1]
    # print(f"Final RMSD is {final_rmsd}.")
    return filename


# 1D RMSD scalar value, average of the entire trajectory
def avg_rmsd_overtime(pdbfile, trajectory, selection="backbone"):
    """take two files: 1) topology in form of PDB or CIF file and
    2) trajectory file from openmm simulation. It computes RMSD for each of
    trajectory frames compared to the reference, which is the initial frame,
    then return the average of all RMSD values over time."""
    u = mda.Universe(pdbfile, trajectory)
    R = rms.RMSD(u, select=selection)
    R.run()
    avg_rmsd = np.mean(R.results.rmsd[2])  # rmsd values are in 3rd column
    return avg_rmsd


class PPIDistanceTool(BaseTool):
    name = "PPIDistance"
    description = """
        This tool will take a PDB file and compute RMSD to compare the
        average distance between protein and peptide or two proteins. The file
        must contain at least two chains to represent two distinct proteins or
        peptides. First, get PDB file instead of CIF file.
        Give this tool the path to PDB file.
    """

    def _run(self, query: str) -> str:
        """Use the tool."""
        try:
            if ".cif" in query:
                return "Please use PDB file, not CIF"
            rmsd = ppi_distance(query)
            return f"RMSD is calculated: {rmsd}"
        except Exception as e:
            return str(e)

    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support async")


class RmsdCompareTool(BaseTool):
    name = "RmsdCompare"
    description = """
        This tool will take two files (PDB or PSF) and compute RMSD to compare
        the difference between two conformations. If trajectory files for both
        protein of interest and reference are obtained from either user or openmm,
        include these two trajectory files as well.
        First, get PDB file instead of CIF file.
        Give this tool the path to the files, separated by commas only (no spaces).
        Make sure to give the files in the following order:
        pdb file, reference file, trajectory (Optional), reference trajectory
        (Optional).
    """

    def _run(self, query: str) -> str:
        """Use the tool."""
        try:
            if ".cif" in query:
                return "Please use PDB file, not CIF"
            if "," not in query:
                return "Please separate file names with comma(s)"
            filelist = query.split(",")
            if len(filelist) != 4 and len(filelist) != 2:
                return "Unaccepted number of files: either 2 PDB/CIF files or 4\
                    with trajectory files"
            rmsd = rmsd_compare(*filelist)
            return f"RMSD is calculated: {rmsd}"
        except Exception as e:
            return str(e)

    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support async")


class RmsdTrajectoryTool(BaseTool):
    name = "RmsdTrajectory"
    description = """
        This tool will take two files: 1) topology in form of PDB or PSF file and
        2) trajectory file from openmm simulation. It computes RMSD for each of
        trajectory frames compared to the reference, which is the initial frame.
        It stores RMSD array in a created file.
        First, get PDB file instead of CIF file.
        Give this tool the paths to topology file and trajectory file, separated
        by comma without space.
    """

    def _run(self, query: str) -> str:
        """Use the tool."""
        try:
            if ".cif" in query:
                return "Please use PDB file, not CIF"
            if "," not in query:
                return "Please separate file names with a comma"
            pdb, traj = query.split(",")
            rmsdfile = rmsd_overtime(pdb, traj)
            return f"Sucessfully created a file with RMSD values: {rmsdfile}."
        except Exception as e:
            return str(e)

    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support async")


class AvgRmsdTrajectoryTool(BaseTool):
    name = "AvgRmsdTrajectory"
    description = """
        This tool will take two files: 1) topology in form of PDB or CIF file and
        2) trajectory file from openmm simulation. It computes RMSD for each of
        trajectory frames compared to the reference, which is the initial frame,
        then return the average of all RMSD values over time.
        First, get PDB file instead of CIF file.
        Give this tool the paths to topology file and trajectory file, separated
        by comma without space.
    """

    def _run(self, query: str) -> str:
        """Use the tool."""
        try:
            if ".cif" in query:
                return "Please use PDB file, not CIF"
            if "," not in query:
                return "Please separate file names with a comma"
            pdb, traj = query.split(",")
            rmsd = avg_rmsd_overtime(pdb, traj)
            return f"RMSD is calculated: {rmsd}"
        except Exception as e:
            return str(e)

    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support async")
