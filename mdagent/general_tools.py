import os
import subprocess

import nbformat as nbf
import pqapi
from langchain.tools import BaseTool

"""Functions """


def list_files_in_directory(directory):
    files = [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]
    return ", ".join(files)


def run_molrender(cif_path):
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


"""This is for plan B
tool, it will create
a notebook
with the code to
install nglview and
display the cif/pdb file."""


def create_notebook(query):
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
    return "Visualization Complete"


class VisualizationToolRender(BaseTool):
    """For this tool
    to work you need
    to instal molrender
    https://github.com/molstar/molrender/tree/master"""

    name = "Visualization of PDB files"
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
            run_molrender(query)
            return "Visualization created"
        except Exception as e:
            return f"An error occurred while running molrender: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class CheckDirectoryFiles(BaseTool):
    name = "List files in directory"
    description = """This tool will
    give you a list of comma
    separated files in the
    current directory."""

    def _run(self, query: str) -> str:
        """use the tool."""
        return list_files_in_directory(".")

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

    name = "Plan B for Visualization of PDB or cif files"
    description = """This tool will create a .ipynb
    file with the visualization
    of the file. It is intended
    to be used only if
    VisualizationToolRender fails.
    Give this tool the path to
    the file and the output
    will be a notebook the
    user can use to visualize
    the file."""

    def _run(self, query: str) -> str:
        """use the tool."""
        create_notebook(query)
        return "Visualization Complete"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class Scholar2ResultLLM(BaseTool):
    name = "LiteratureSearch"
    description = """Input a specific question,
                returns an answer from literature search."""
    pqa_key: str = ""

    def __init__(self, pqa_key: str):
        super().__init__()
        self.pqa_key = pqa_key

    def _run(self, question: str) -> str:
        """Use the tool"""
        response = pqapi.agent_query("default", question)
        return response.answer

    async def _arun(self, question: str) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError
