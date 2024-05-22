import os
import shutil
import subprocess
from typing import Optional

import nbformat as nbf
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from mdagent.utils import PathRegistry


class VisFunctions:
    def __init__(self, path_registry):
        self.path_registry = path_registry
        self.starting_files = os.listdir(".")

    def _find_png(self):
        current_files = os.listdir(".")
        new_files = [f for f in current_files if f not in self.starting_files]
        png_files = [f for f in new_files if f.endswith(".png")]
        return png_files

    def run_molrender(self, cif_path: str) -> str:
        """Function to run molrender,
        it requires node.js to be installed
        and the molrender package to be
        installed globally.
        This will save .png
        files in the current
        directory."""
        self.cif_file_name = os.path.basename(cif_path)

        cmd = ["molrender", "all", cif_path, ".", "--format", "png"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            raise RuntimeError("molrender package not found")
        file_name = self._find_png()
        if not file_name:
            raise FileNotFoundError("No .png files were created")
        shutil.move(file_name[0], f"{self.path_registry.ckpt_figures}/{file_name[0]}")
        self.path_registry.map_path(
            f"mol_render_{self.cif_file_name}",
            f"{self.path_registry.ckpt_figures}/{file_name[0]}",
            (
                f"Visualization of cif file {self.cif_file_name}"
                "as png file. using molrender."
            ),
        )

        if result.returncode != 0:
            raise RuntimeError(f"Error running molrender: {result.stderr}")
        else:
            print(f"Output: {result.stdout}")
        return (
            "Visualization using molrender complete, "
            f"saved as: mol_render_{self.cif_file_name}"
        )

    def create_notebook(self, top_file: str, traj=None) -> str:
        """This is for plan B
        tool, it will create
        a notebook
        with the code to
        install nglview and
        display the cif/pdb file."""
        self.cif_file_name = os.path.basename(top_file)

        # Create a new notebook
        nb = nbf.v4.new_notebook()

        # Code to install NGLview
        install_code = "!pip install -q nglview"
        disclaimer = (
            "#Note: Is possible the agent misses the \n"
            "#correct topology file, and/or trajectory file.\n"
            "#Check if the files are correct beforehand.\n"
        )
        if traj:
            # Code to import NGLview and display a file
            install_code += "\n!pip install -q mdtraj"

            import_code = (
                "import nglview as nv\nimport mdtraj as md\n"
                f"traj = md.load('{traj}',top='{top_file}')\n"
                "view=nv.show_mdtraj(traj)\nview"
            )
        # Code to import NGLview and display a file
        import_code = f"import nglview as nv\nview=nv.show_file('{top_file}')\nview"

        # Create new code cells
        install_cell = nbf.v4.new_code_cell(source=install_code)
        disclaimer_cell = nbf.v4.new_markdown_cell(disclaimer)
        import_cell = nbf.v4.new_code_cell(source=import_code)

        # Add the cells
        nb.cells.extend([install_cell, disclaimer_cell, import_cell])

        # Write the notebook to a file
        notebook_name = (
            f"{self.path_registry.ckpt_figures}"
            f"/{self.cif_file_name.split('.')[0]}_vis.ipynb"
        )
        with open(notebook_name, "w") as f:
            nbf.write(nb, f)

        self.path_registry.map_path(
            notebook_name,
            notebook_name,
            f"Notebook to visualize cif/pdb file {self.cif_file_name} using nglview.",
        )
        return "Visualization Complete"


class visProteinSchema(BaseModel):
    topology_fileid: str = Field(
        decription="The fileid of the protein file to visualize"
    )
    trajectory_fileid: Optional[str] = Field(
        description="The fileid of the trajectory"
        " file to visualize if type is 'movie'"
    )
    type: Optional[str] = Field(
        "static",
        description=(
            "The type of visualization to create."
            "Options are 'static' (default) or 'movie'"
        ),
    )


class VisualizeProtein(BaseTool):
    """To get a png, you must install molrender
    https://github.com/molstar/molrender/tree/master
    Otherwise, you will get a notebook where you
    can visualize the protein."""

    name = "PDBVisualization"
    args_schema = visProteinSchema
    description = (
        "This tool will create"
        " a visualization of a protein"
        " file as a png file OR"
        " it will create"
        " a .ipynb file with the"
        " visualization of the"
        " file, depending on the"
        " packages available."
        " If a notebook is created,"
        " the user can open the"
        " notebook and visualize the"
        " system."
    )
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, **input):
        """use the tool."""
        input = self.validate_input(input)
        topology_id = input["topology_fileid"]
        if not self.path_registry:
            return "Failed. Error: Path registry is not set"
        topology_path = self.path_registry.get_mapped_path(topology_id)
        type = input["type"]
        if not self.path_registry:
            return "Failed. Error: Path registry is not set"
        top_path = self.path_registry.get_mapped_path(topology_path)
        if not top_path:
            return f"Failed. File not found: {topology_id}"
        if type == "movie":
            trajectory_id = input["trajectory_fileid"]
            if not trajectory_id:
                print("no trajectory fileid, using static visualization")
                type = "static"
            else:
                traj_path = self.path_registry.get_mapped_path(trajectory_id)
        vis = VisFunctions(self.path_registry)
        try:
            if type == "static":
                return "Succeeded" + vis.run_molrender(top_path)
            if type == "movie":
                return "Succeeded" + vis.create_notebook(top_path, traj=traj_path)
        except (RuntimeError, FileNotFoundError) as e:
            print(f"Error running molrender: {str(e)}. Using NGLView instead.")
            try:
                vis.create_notebook(top_path)
                return "Succeeded. Visualization created as notebook"
            except Exception as e:
                return f"Failed. {type(e).__name__}: {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

    def validate_input(self, input):
        input = input.get("input", input)
        input = input.get("action_input", input)
        error = ""
        top_id = input.get("topology_fileid")
        if not top_id:
            error += "topology_fileid field is required. "

        # check if trajectory id is valid
        fileids = self.path_registry.list_path_names()

        if top_id not in fileids:
            error += "trajectory_fileid not in path registry"

        trajectory_id = input.get("trajectory_fileid", None)
        if trajectory_id:
            if trajectory_id not in fileids:
                error += "trajectory_fileid not in path registry"

        type = input.get("type", "static")
        if type not in ["static", "movie"]:
            type = "static"

        if error == "":
            error = None
        return {
            "protein_fileid": top_id,
            "trajectory_fileid": trajectory_id,
            "type": type,
        }
