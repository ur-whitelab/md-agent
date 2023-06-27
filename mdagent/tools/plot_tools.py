import csv
import inspect

import matplotlib.pyplot as plt
import numpy as np
from langchain.tools import BaseTool
from openmm.app import StateDataReporter


def _csv_reader(csv_filename):
    try:
        with open(csv_filename, "r") as file:
            reader = csv.reader(file)
            return reader
    except Exception:
        raise FileNotFoundError("File not found.")


def _get_data_params():
    try:
        # Get the parameters of the StateDataReporter's initialization function
        parameters = inspect.signature(StateDataReporter.__init__).parameters
        parameter_names = [name for name in parameters.keys() if name != "self"]
    except Exception:
        return []
    return parameter_names


def _find_data_params(csv_filename):
    reader = _csv_reader(csv_filename)
    header = next(reader)  # Read the first row (header)
    # Normalize the header names
    header = [name.replace(" ", "").lower() for name in header]

    # Normalize the parameter names
    parameter_list = _get_data_params()
    if len(parameter_list) == 0:
        return [param for param in header]
    parameter_list = [param.replace(" ", "").lower() for param in parameter_list]

    # Find and return the parameters that exist in the CSV file
    return [param for param in header if param in parameter_list]


def _plan_plots(csv_filename):
    physical_params = {
        "potentialenergy": "kJ/mol",
        "temperature": "K",
        "kineticenergy": "kJ/mol",
        "totalenergy": "kJ/mol",
        "volume": "nm³",
        "density": "g/cm³",
        "systemmass": "Da",
        "step": "",
        "time": "ps",
    }
    params = _find_data_params(csv_filename)
    timestep = "step" if "step" in params else "time" if "time" in params else None
    if timestep is None:
        raise ValueError("No timestep data found in csv file.")

    data = np.array(list(_csv_reader(csv_filename)), dtype=float)

    data_dict = {}
    for i, param in enumerate(params):
        if param in physical_params:
            data_dict[param] = (data[:, i], physical_params[param])

    return data_dict


def create_plots(data_dict, timestep, x_unit, timestep_choice):
    for param, (plot_data, unit) in data_dict.items():
        plt.plot(timestep, plot_data)
        plt.xlabel(timestep_choice.capitalize() + f" ({x_unit})")
        plt.ylabel(f"{param.capitalize()} ({unit})")
        plt.title(f"{param.capitalize()} vs. {timestep_choice.capitalize()}")
        plt.savefig(f"{param}_vs_{timestep_choice}.png")
        plt.close()


class SimulationOutputFigures(BaseTool):
    name = "PostSimulationFigures"
    description = """This tool will take
    a csv file output from an openmm
    simulation and create figures for
    all physical parameters
    versus timestep of the simulation.
    Give this tool the path to the
    csv file output from the simulation."""

    def _run(self, file_path: str) -> str:
        """use the tool."""
        try:
            data_dict = _plan_plots(file_path)
            if "step" in data_dict:
                timestep = data_dict["step"][0]
                timestep_choice = "step"
                x_unit = ""
            elif "time" in data_dict:
                timestep = data_dict["time"][0]
                timestep_choice = "time"
                x_unit = data_dict["time"][1]
            create_plots(data_dict, timestep, x_unit, timestep_choice)
            return "Figures created."
        except ValueError:
            return "No timestep data found in csv file."
        except FileNotFoundError:
            return "Issue with CSV file, file not found."

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
