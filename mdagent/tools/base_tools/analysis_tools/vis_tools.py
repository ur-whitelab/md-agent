import subprocess
from typing import Optional

import nbformat as nbf
from langchain.tools import BaseTool

from mdagent.utils import PathRegistry


class VisFunctions:
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


class VisualizeProtein(BaseTool):
    """To get a png, you must install molrender
    https://github.com/molstar/molrender/tree/master
    Otherwise, you will get a notebook where you
    can visualize the protein."""

    name = "PDBVisualization"
    description = """This tool will create
                    a visualization of a cif
                    file as a png file in
                    the same directory OR
                    it will create
                    a .ipynb file with the
                    visualization of the
                    file, depending on the
                    packages available.
                    If a notebook is created,
                    the user can open the
                    notebook and visualize the
                    system."""
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, query: str) -> str:
        """use the tool."""
        vis = VisFunctions()
        try:
            vis.run_molrender(query)
            return "Visualization created as png"
        except Exception:
            try:
                vis.create_notebook(query, self.path_registry)
                return "Visualization created as notebook"
            except Exception as e:
                return f"An error occurred while running molrender: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
