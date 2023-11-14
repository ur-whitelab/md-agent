import os

import matplotlib.pyplot as plt
import MDAnalysis as mda
import nglview as nv
import numpy as np
from MDAnalysis.analysis import align, diffusionmap, rms
from nglview.contrib.movie import MovieMaker

# all things related to RMSD as 'standard deviation'
# 1. time-dependent RMSD of the whole trajectory with all or selected atoms
#       (all, backbone, heavy atoms)
# 2. pairwise RMSD
# 3. RMSF

# TO DO:
# test the tool with all RMSD functions separately (use specific function for each)
# test the tool with all RMSD functions together (use calculate_rmsd)
# add brief description for each rmsd method


class RMSDFunctions:
    def __init__(self, trajectory, pdb_file, ref_file=None, ref_trajectory=None):
        self.pdb_file = pdb_file
        self.trajectory = trajectory
        self.pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]
        self.ref_file = ref_file
        self.ref_trajectory = ref_trajectory
        if ref_file:
            self.ref_name = os.path.splitext(os.path.basename(ref_file))[0]
        else:
            self.ref_name = None

    def calculate_rmsd(self, rmsd_type="rmsd", selection="backbone"):
        i = 0
        base_filename = f"{rmsd_type}_{self.pdb_name}"
        filename = base_filename
        while os.path.exists(filename + ".csv"):
            i += 1
            filename = f"{base_filename}_{i}"
        self.filename = filename

        if rmsd_type == "rmsd":
            return self.compute_1d_rmsd(selection=selection)
        elif rmsd_type == "pairwise_rmsd":
            return self.compute_2d_rmsd(selection)
        elif rmsd_type == "rmsf":
            return self.compute_rmsf(selection)
        else:
            raise ValueError(
                "Invalid rmsd_type. Please choose from 'rmsd', 'pairwise_rmsd', 'rmsf'"
            )

    def compute_1d_rmsd(self, selection="backbone", pdbid=None, plot=True):
        # 1D time-dependent RMSD, gives one scalar value for each timestep
        """take two files: 1) topology in form of PDB or PSF file and
        2) trajectory file from openmm simulation. It computes RMSD for each of
        trajectory frames compared to the reference, which is the initial frame.
        It stores RMSD array in a created file."""
        u = mda.Universe(self.pdb_file, self.trajectory)
        R = rms.RMSD(u, select=selection)
        R.run()

        # save to file
        np.savetxt(
            f"{self.filename}.csv",
            R.results.rmsd,
            fmt=["%d", "%f", "%f"],
            delimiter=",",
            header="Frame,Time,RMSD",
            comments="",
        )
        print("Calculated RMSD for each timestep with respect to the initial frame.")
        avg_rmsd = np.mean(R.results.rmsd[2])  # rmsd values are in 3rd column
        print(f"Average RMSD is {avg_rmsd}.")
        final_rmsd = R.results.rmsd[2][-1]
        print(f"Final RMSD is {final_rmsd}.")
        # if plot:
        #     df = pd.DataFrame(R.results.rmsd, columns=["Frame", "Time", "RMSD"])
        #     ax = df.plot(x="Frame", y="RMSD", kind="line", title="RMSD")
        #     ax.set_
        if plot:
            plt.plot(R.results.rmsd[0], R.results.rmsd[2], label=str(selection))
            plt.xlabel("Frame")
            plt.ylabel("RMSD ($\AA$)")
            plt.title("Time-Dependent RMSD")
            plt.legend()
            plt.show()
            plt.savefig(f"{self.filename}.png")
        return "SOME STRING FOR MDAGENT"

    def compute_2d_rmsd(self, selection="backbone", plot_heatmap=True):
        u = mda.Universe(self.pdb_file, self.trajectory)
        if self.ref_file and self.ref_trajectory:
            ref = mda.Universe(self.ref_file, self.ref_trajectory)
        else:
            ref = None

        if ref is None:
            # pairwise RMSD of a trajectory to itself
            align.AlignTraj(u, u, select=selection, in_memory=True).run()
            matrix = diffusionmap.DistanceMatrix(u, select=selection).run()
            pairwise_matrix = matrix.results.dist_matrix
            x_label = y_label = "Frame"
        else:
            pairwise_matrix = np.zeros(
                (len(u.trajectory), len(ref.trajectory))  # y-axis
            )  # x-axis
            for i, frame in enumerate(u.trajectory):
                r = rms.RMSD(ref, u, select=selection, ref_frame=i).run()
                pairwise_matrix[i] = r.results.rmsd[:, 2]
            x_label = f"Frame ({self.ref_name})"
            y_label = f"Frame ({self.pdb_name}))"
        np.savetxt(
            f"{self.filename}.csv",
            pairwise_matrix,
            delimiter=",",
        )
        if plot_heatmap:
            plt.imshow(pairwise_matrix, cmap="viridis")
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.colorbar(label=r"RMSD ($\AA$)")
            plt.savefig(f"{self.filename}.png")

    def compute_rmsf(self, selection="backbone", plot=True, movie=False):
        # Load the universe
        u = mda.Universe(self.pdb_file, self.trajectory)

        # Align the trajectory to the first frame for better RMSF computation
        # use averages as a reference for aligning
        average = align.AverageStructure(u, u, select=selection, ref_frame=0).run()
        align_ref = average.results.universe
        align.AlignTraj(u, align_ref, select=selection, in_memory=True).run()

        # Compute RMSF
        atoms = u.select_atoms(selection)
        R = rms.RMSF(atoms).run()
        rmsf = R.results.rmsf

        # Save to a text file
        rmsf_data = np.column_stack((atoms.resids, rmsf))
        np.savetxt(
            f"{self.filename}.csv",
            rmsf_data,
            delimiter=",",
            header="Residue_ID,RMSF",
            comments="",
        )

        # Plot RMSF
        if plot:
            plt.figure(figsize=(5, 3))
            plt.plot(atoms.resnums, rmsf, label=str(selection))
            plt.xlabel("Residue Number")
            plt.ylabel("RMSF ($\AA$)")
            plt.title("Root Mean Square Fluctuation")
            plt.legend()
            plt.show()
            plt.savefig(f"{self.filename}.png")

        # Create a movie with nglview
        if movie:
            u.add_TopologyAttr("tempfactors")  # add empty attribute for all atoms
            protein = u.select_atoms("protein")  # select protein atoms
            for residue, r_value in zip(protein.residues, R.results.rmsf):
                residue.atoms.tempfactors = r_value
            view = nv.show_mdanalysis(u)
            view.update_representation(color_scheme="bfactor")
            view
            movie = MovieMaker(
                view,
                step=100,  # keep every 100th frame
                output=f"images/{self.filename}.gif",
                render_params={"factor": 3},  # set to 4 for highest quality
            )
            movie.make()
            u.atoms.write(f"{self.filename}_tempfactors.pdb")

        return "SOME STRING FOR MDAGENT"
