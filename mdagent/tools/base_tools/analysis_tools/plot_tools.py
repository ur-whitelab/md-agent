import csv
import re
from typing import Optional

import matplotlib.pyplot as plt
from langchain.tools import BaseTool

from mdagent.utils import PathRegistry


class PlottingTools:
    def __init__(
        self,
        path_registry,
    ):
        self.path_registry = path_registry
        self.data = None
        self.headers = None
        self.matched_headers = None
        self.file_name = None
        self.file_path = None

    def _find_file(self, file_name: str) -> None:
        self.file_name = file_name
        self.file_path = self.path_registry.get_mapped_path(file_name)
        if not self.file_path:
            raise FileNotFoundError("File not found.")
        return None

    def process_csv(self) -> None:
        with open(self.file_path, "r") as f:
            reader = csv.DictReader(f)
            self.headers = reader.fieldnames if reader.fieldnames is not None else []
            self.data = list(reader)

        self.matched_headers = [
            (i, header)
            for i, header in enumerate(self.headers)
            if re.search(r"(step|time)", header, re.IGNORECASE)
        ]

        if not self.matched_headers or not self.headers or not self.data:
            raise ValueError("File could not be processed.")
        return None

    def plot_data(self) -> str:
        if self.matched_headers:
            time_or_step = self.matched_headers[0][1]
            xlab = "step" if "step" in time_or_step.lower() else "time"
        else:
            return "No 'step' or 'time' headers found."

        failed_headers = []
        created_plots = []
        for header in self.headers:
            if header != time_or_step:
                try:
                    x = [float(row[time_or_step]) for row in self.data]
                    y = [float(row[header]) for row in self.data]

                    header_lab = (
                        header.split("(")[0].strip() if "(" in header else header
                    ).lower()
                    plot_name = f"{self.file_name}_{xlab}_vs_{header_lab}.png"

                    # Generate and save the plot
                    plt.figure()
                    plt.plot(x, y)
                    plt.xlabel(xlab)
                    plt.ylabel(header)
                    plt.title(f"{self.file_name}_{xlab} vs {header_lab}")
                    plt.savefig(plot_name)
                    self.path_registry.map_path(
                        plot_name,
                        plot_name,
                        (
                            "Post Simulation Figure for "
                            "{self.file_name} - {header_lab} vs {xlab}"
                        ),
                    )
                    plt.close()

                    created_plots.append(plot_name)
                except ValueError:
                    failed_headers.append(header)

        if (
            len(failed_headers) == len(self.headers) - 1
        ):  # -1 to account for time_or_step header
            raise Exception("All plots failed due to non-numeric data.")

        return ", ".join(created_plots)


class SimulationOutputFigures(BaseTool):
    name = "PostSimulationFigures"
    description = """This tool will take
    a csv file id output from an openmm
    simulation and create figures for
    all physical parameters
    versus timestep of the simulation.
    Give this tool the path to the
    csv file output from the simulation."""

    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry] = None):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, file_name: str) -> str:
        """use the tool."""
        try:
            plotting_tools = PlottingTools(self.path_registry)
            plotting_tools._find_file(file_name)
            plotting_tools.process_csv()
            plot_result = plotting_tools.plot_data()
            if type(plot_result) == str:
                return "Figures created: " + plot_result
            else:
                return "No figures created."
        except ValueError:
            return "File could not be processed."
        except FileNotFoundError:
            return "Issue with CSV file, file not found."
        except Exception as e:
            return str(e)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
