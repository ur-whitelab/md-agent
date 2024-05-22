import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from scipy.integrate import simpson

from mdagent.utils import FileType, PathRegistry


class TimeCorrelation:
    def __init__(self, path_registry, csv_file_id, time_step, property):
        """
        Initialize the TimeCorrelationAnalysis class with path registry.

        Parameters:
        path_registry (PathRegistry): mapping file IDs to file paths.
        csv_file_id (str): File ID for the CSV file containing the property time series.
        time_step (float): The time step between frames in the trajectory.
        property (str): The property for which to calculate the time correlation.
        """
        self.path_registry = path_registry
        self.property = property
        self.time_step = time_step
        self.autocorrelation = None
        self.tau = None

        all_file_ids = self.path_registry.list_path_names()
        if csv_file_id not in all_file_ids:
            raise ValueError("File ID not found in path registry")
        self.file_path = self.path_registry.get_mapped_path(csv_file_id)

    def calculate_time_correlation(self):
        """
        Calculate the time correlation function and tau for the property.

        Returns:
        str: Completion message and file path where the time correlation is saved.
        """
        time_series = np.loadtxt(self.file_path, delimiter=",")
        num_frames = time_series.shape[0]
        autocorrelation = np.zeros(num_frames)

        mean_value = np.mean(time_series)
        fluctuation = time_series - mean_value

        for t in range(num_frames):
            dot_products = np.sum(
                fluctuation[: num_frames - t] * fluctuation[t:], axis=0
            )
            autocorrelation[t] = np.mean(dot_products)
        autocorrelation /= autocorrelation[0]  # Normalize

        # integrate the autocorrelation function to obtain tau
        self.tau = simpson(autocorrelation, dx=self.time_step)
        self.autocorrelation = autocorrelation

        # save to file
        autocorr_file = f"{self.property}_time_correlation.csv"
        i = 0
        while os.path.exists(f"{self.path_registry.ckpt_figures}/{autocorr_file}"):
            i += 1
            autocorr_file = f"{self.property}_time_correlation_{i}.csv"
        np.savetxt(
            f"{self.path_registry.ckpt_figures}/{autocorr_file}",
            autocorrelation,
            delimiter=",",
            header=f"Autocorrelation for {self.property}. Tau = {self.tau}",
        )
        self.path_registry.map_path(
            f"{self.property}_autocorrelation_{i}",
            f"{self.path_registry.ckpt_figures}/{autocorr_file}",
            description=f"Autocorrelation for {self.property}",
        )
        return f"Time correlation function calculated and saved to {autocorr_file}."

    def plot_time_correlation(self):
        """
        Plot the time correlation function.

        Returns:
        str: Completion message with file ID information.
        """
        message = ""
        if self.autocorrelation is None:
            message += self.calculate_time_correlation()
        fig_analysis = f"{self.property}_time_correlation"
        fig_name = self.path_registry.write_file_name(
            type=FileType.FIGURE, fig_analysis=fig_analysis, file_format="png"
        )
        fig_id = self.path_registry.get_fileid(file_name=fig_name, type=FileType.FIGURE)
        plt.plot(self.autocorrelation)
        plt.xlabel("Time Lag")
        plt.ylabel("Autocorrelation")
        plt.title(f"Time Correlation Function, Tau = {self.tau:.2f}")
        plt.savefig(f"{self.path_registry.ckpt_figures}/{fig_name}")
        plt.close()
        self.path_registry.map_path(
            fig_id,
            f"{self.path_registry.ckpt_figures}/{fig_name}",
            description=f"Plot of time correlation function for {self.property}",
        )
        message += (
            f"Time correlation function for {self.property} saved to "
            f"{fig_name} with plot ID {fig_id}. Time correlation tau: {self.tau}"
        )
        return message


class TimeCorrelationAnalysisInput(BaseModel):
    file_id: str = Field(None, description="File ID for file containing time series.")
    timestep: float = Field(None, description="Time step between frames in trajectory.")
    property: str = Field(
        None, description="The property for which to calculate the time correlation."
    )


class TimeCorrelationAnalysis(BaseTool):
    name = "TimeCorrelationAnalysis"
    description = (
        "Calculate the time correlation function and correlation"
        "time (tau) for a given time series of a property."
    )
    args_schema = TimeCorrelationAnalysisInput
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, file_id: str, timestep: float, property: str) -> str:
        try:
            time_corr = TimeCorrelation(self.path_registry, file_id, timestep, property)
            return f"Succeeded. {time_corr.plot_time_correlation()}"
        except Exception as e:
            return f"Failed. Error: {e}"


###### EXAMPLES OF TIME SERIES DATA ######
# note that time_step is required - include in returned messages & file descriptions
# proposed FileType: TIME_SERIES, with file starting "TIME_{property}"

# # RMSD
# traj = md.load(trajectory_file, top=pdb_file)
# time_step = traj.timestep / 1000  # Assuming timestep is in picoseconds
# rmsd_series = md.rmsd(traj, traj, 0)
# np.savetxt("rmsd_series.csv", rmsd_series, delimiter=",")
