import csv
import re

import matplotlib.pyplot as plt
from langchain.tools import BaseTool


def process_csv(file_name):
    with open(file_name, "r") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        data = list(reader)

    matched_headers = [
        (i, header)
        for i, header in enumerate(headers)
        if re.search(r"(step|time)", header, re.IGNORECASE)
    ]

    return data, headers, matched_headers


def plot_data(data, headers, matched_headers):
    # Get the first matched header
    if matched_headers:
        time_or_step = matched_headers[0][1]
    else:
        print("No 'step' or 'time' headers found.")
        return

    failed_headers = []

    created_plots = []
    # For each header (except the time/step one), plot time/step vs that header
    header_count = 0
    for header in headers:
        if header != time_or_step:
            header_count += 1
            try:
                # Extract the data for the x and y axes
                x = [float(row[time_or_step]) for row in data]
                y = [float(row[header]) for row in data]

                if "step" in time_or_step.lower():
                    xlab = "step"
                if "(" in header:
                    header_lab = (header.split("(")[0]).strip()
                # Generate the plot
                plt.figure()
                plt.plot(x, y)
                plt.xlabel(xlab)
                plt.ylabel(header)
                plt.title(f"{xlab} vs {header_lab}")

                # Save the figure
                plt.savefig(f"{xlab}_vs_{header_lab}.png")
                plt.close()
                created_plots.append(f"{xlab}_vs_{header_lab}.png")
            except ValueError:  # If data cannot be converted to float
                failed_headers.append(header)

    # If all plots failed, raise an exception
    if len(failed_headers) == len(headers) - header_count:
        raise Exception("All plots failed due to non-numeric data.")
    return ", ".join(created_plots)


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
            data, headers, matched_headers = process_csv(file_path)
            plot_result = plot_data(data, headers, matched_headers)
            if type(plot_result) == str:
                return "Figures created: " + plot_result
            else:
                return "No figures created."
        except ValueError:
            return "No timestep data found in csv file."
        except FileNotFoundError:
            return "Issue with CSV file, file not found."
        except Exception as e:
            return str(e)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
