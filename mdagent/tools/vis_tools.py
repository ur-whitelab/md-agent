import os
import subprocess
from typing import Optional

import nbformat as nbf
from langchain.tools import BaseTool

from .registry import PathRegistry


class VisFunctions:
    def list_files_in_directory(self, directory):
        files = [
            f
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]
        return ", ".join(files)

    def run_molrender(self, cif_path):
        """Function to run molrender,
        it requires node.js to be installed
        and the molrender package to be
        installed globally.
        This will save .png
        files in the current
        directory."""

        cmd = ["molrender", "all", cif_path, ".", "--format", "png"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return Exception(f"Error running molrender: {result.stderr}")
        else:
            print(f"Output: {result.stdout}")

    def create_notebook(self, query, PathRegistry):
        """This is for plan B
        tool, it will create
        a notebook
        with the code to
        install nglview and
        display the cif/pdb file."""
        # Create a new notebook
        nb = nbf.v4.new_notebook()

        # Code to install NGLview
        install_code = "!pip install -q nglview"

        # Code to import NGLview and display a file
        import_code = f"""
        import nglview as nv
        view = nv.show_file("{query}")
        view
        """

        # Create new code cells
        install_cell = nbf.v4.new_code_cell(source=install_code)
        import_cell = nbf.v4.new_code_cell(source=import_code)

        # Add the cells
        nb.cells.extend([install_cell, import_cell])

        # Write the notebook to a file
        with open("Visualization.ipynb", "w") as f:
            nbf.write(nb, f)
        # add filename to registry
        file_description = "Notebook to visualize cif/pdb files"
        PathRegistry.map_path(
            "visualize_notebook", "Visualization.ipynb", file_description
        )
        return "Visualization Complete"


class VisualizationToolRender(BaseTool):
    """For this tool
    to work you need
    to instal molrender
    https://github.com/molstar/molrender/tree/master"""

    name = "PDBVisualization"
    description = """This tool will create
                    a visualization of a cif
                    file as a png file in
                    the same directory. if
                    cif file doesnt exist
                    you should look for
                    alternatives in the directory"""

    def _run(self, query: str) -> str:
        """use the tool."""
        try:
            vis = VisFunctions()
            vis.run_molrender(query)
            return "Visualization created"
        except Exception as e:
            return f"An error occurred while running molrender: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class CheckDirectoryFiles(BaseTool):
    name = "ListDirectoryFiles"
    description = """This tool will
    give you a list of comma
    separated files in the
    current directory."""

    def _run(self, query: str) -> str:
        """use the tool."""
        try:
            vis = VisFunctions()
            return vis.list_files_in_directory(".")
        except Exception:
            return "An error occurred while listing files in directory"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class PlanBVisualizationTool(BaseTool):
    """This tool will create
    a .ipynb file with the
    visualization of the
    file. It is intended
    to be used only
    if VisualizationToolRender fails"""

    name = "PlanBVisualizationTool"
    description = """This tool will create a .ipynb
                    file with the visualization
                    of the file. It is intended
                    to be used only if
                    VisualizationToolRender fails.
                    Give this tool the saved
                    name of the file
                    and the output
                    will be a notebook the
                    user can use to visualize
                    the file."""
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, query: str) -> str:
        """use the tool."""
        try:
            if self.path_registry is None:  # this should not happen
                return "Path registry not initialized"
            vis = VisFunctions()
            vis.create_notebook(query, self.path_registry)
            return "Visualization Complete"
        except Exception:
            return "An error occurred while creating the notebook"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
