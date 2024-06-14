import mdtraj as md
from langchain.tools import BaseTool

from mdagent.utils import PathRegistry, load_single_traj


class BakerHubbard(BaseTool):
    name = "Baker_hubbard"
    description = """Identify hydrogen bonds based that are present in at least 10%
     of each frames ( freq =0.1) and provides a list of tuples with each tuples
    containing three integers representing the indices of atoms (donor, hydrogen,
    acceptor) involved in the hydrogen bonding."""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None, freq=0.1):
        try:
            traj = load_single_traj(self.path_registry, top_file, traj_file)
            if not traj:
                return "Failed. Trajectory could not be loaded."

            result = md.baker_hubbard(
                traj, freq, exclude_water=True, periodic=True, sidechain_only=False
            )
            return f"Succeeded. {result}"

        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

    async def _arun(self, traj_file, top_file=None, freq=0.1):
        raise NotImplementedError("Async version not implemented")

    # def plot_hydrogen_bonds(self, traj_file, top_file=None, freq=0.1):
    #     # Run the analysis
    #     result = self._run(traj_file, top_file, freq)

    #     # Check if the result is successful
    #     if result.startswith("Failed."):
    #         print(result)
    #         return

    #     # Extract the result from the "Succeeded." message
    #     result = eval(result.replace("Succeeded. ", ""))

    #     # Load the trajectory to get the number of frames
    #     traj = load_single_traj(self.path_registry, top_file, traj_file)
    #     n_frames = traj.n_frames

    #     # Initialize an array to count hydrogen bonds per frame
    #     hbond_counts = np.zeros(n_frames)

    #     # Process the results to count hydrogen bonds per frame
    #     for bond in result:
    #         # bond[1] should contain the frames where the bond exists
    #         for frame in bond[1]:
    #             hbond_counts[frame] += 1
    #     # Plot the results
    #     plt.figure(figsize=(10, 6))
    #     plt.bar(range(n_frames), hbond_counts, color='b')
    #     plt.xlabel('Frame')
    #     plt.ylabel('Number of Hydrogen Bonds')
    #     plt.title('Hydrogen Bonds Over Time')
    #     plt.grid(True)
    #     plt.show()


class KabschSander(BaseTool):
    name = "kabsch_sander"
    description = """Compute hydrogen bond energy between each pair
    of residues in every frame of the simulation and provides list of indices
    specifying which residues are involved in each hydrogen bond and its hydrogen bond
    energies."""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        try:
            traj = load_single_traj(self.path_registry, top_file, traj_file)
            if not traj:
                return "Failed. Trajectory could not be loaded."

            result = md.kabsch_sander(traj)
            return f"Succeeded. {result}"

        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")


class WernetNilsson(BaseTool):
    name = "wernet_nilsson"
    description = """Identifies hydrogen bonds without frequency parameter, provides
    a list of tuples with indices of donor, hydrogen and acceptor atoms. Prefer this tool over BakerHubbard, except where the user explicitly requests BakerHubbard."""

    path_registry: PathRegistry | None = None

    def __init__(self, path_registry: PathRegistry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, traj_file, top_file=None):
        try:
            traj = load_single_traj(self.path_registry, top_file, traj_file)
            if not traj:
                return "Failed. Trajectory could not be loaded."
            result = md.wernet_nilsson(
                traj, exclude_water=True, periodic=True, sidechain_only=False
            )
            return f"Succeeded. {result}"
        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

    async def _arun(self, traj_file, top_file=None):
        raise NotImplementedError("Async version not implemented")


# 3d visualization?
# heatmap?
# time series plots
# histograms
