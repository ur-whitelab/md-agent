import os
import re
from typing import Optional

from langchain.tools import BaseTool
from openmm.app import PDBFile, PDBxFile
from pdbfixer import PDBFixer

from mdagent.utils import PathRegistry


class CleaningTools:
    def _extract_path(self, user_input: str, path_registry: PathRegistry) -> str:
        """Extract file path from user input."""

        # Remove any leading or trailing white space
        user_input = user_input.strip()

        # Remove single and double quotes from the user_input
        user_input = user_input.replace("'", "")
        user_input = user_input.replace('"', "")

        # First check the path registry
        mapped_path = path_registry.get_mapped_path(user_input)
        if mapped_path != "Name not found in path registry.":
            return mapped_path

        # If not found in registry, check if it is a valid path
        match = re.search(r"[a-zA-Z0-9_\-/\\:.]+(?:\.pdb|\.cif)", user_input)

        if match:
            return match.group(0)
        else:
            raise ValueError("No valid file path found in user input.")

    def _standard_cleaning(self, pdbfile: str, path_registry: PathRegistry):
        pdbfile = self._extract_path(pdbfile, path_registry)
        name, end = os.path.splitext(os.path.basename(pdbfile))
        end = end.lstrip(".")
        fixer = PDBFixer(filename=pdbfile)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(True)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        tidy_filename = f"tidy_{name}.{end}"
        if end == "pdb":
            PDBFile.writeFile(fixer.topology, fixer.positions, open(tidy_filename, "a"))
        elif end == "cif":
            PDBxFile.writeFile(
                fixer.topology, fixer.positions, open(tidy_filename, "a")
            )
        # add filename to registry
        short_name = f"tidy_{name}"
        file_description = "Cleaned File. Standard cleaning."
        path_registry.map_path(short_name, tidy_filename, file_description)
        return f"{file_description} Written to {tidy_filename}"

    def _remove_water(self, pdbfile: str, path_registry: PathRegistry):
        pdbfile = self._extract_path(pdbfile, path_registry)
        name, end = os.path.splitext(os.path.basename(pdbfile))
        end = end.lstrip(".")
        fixer = PDBFixer(filename=pdbfile)
        fixer.removeHeterogens(False)
        tidy_filename = f"tidy_{name}.{end}"
        if end == "pdb":
            PDBFile.writeFile(fixer.topology, fixer.positions, open(tidy_filename, "a"))
        elif end == "cif":
            PDBxFile.writeFile(
                fixer.topology, fixer.positions, open(tidy_filename, "a")
            )
        # add filename to registry
        short_name = f"tidy_{name}"
        file_description = "Cleaned File. Removed water."
        path_registry.map_path(short_name, tidy_filename, file_description)
        return f"{file_description} Written to {tidy_filename}"

    def _add_hydrogens_and_remove_water(
        self, pdbfile: str, path_registry: PathRegistry
    ):
        pdbfile = self._extract_path(pdbfile, path_registry)
        name, end = os.path.splitext(os.path.basename(pdbfile))
        end = end.lstrip(".")
        fixer = PDBFixer(filename=pdbfile)
        fixer.removeHeterogens(False)
        tidy_filename = f"tidy_{name}.{end}"
        if end == "pdb":
            PDBFile.writeFile(fixer.topology, fixer.positions, open(tidy_filename, "a"))
        elif end == "cif":
            PDBxFile.writeFile(
                fixer.topology, fixer.positions, open(tidy_filename, "a")
            )
        # add filename to registry
        short_name = f"tidy_{name}"
        file_description = "Cleaned File. Missing Hydrogens added and water removed."
        path_registry.map_path(short_name, tidy_filename, file_description)
        return f"{file_description} Written to {tidy_filename}"

    def _add_hydrogens(self, pdbfile: str, path_registry: PathRegistry):
        pdbfile = self._extract_path(pdbfile, path_registry)
        name, end = os.path.splitext(os.path.basename(pdbfile))
        end = end.lstrip(".")
        fixer = PDBFixer(filename=pdbfile)
        fixer.addMissingHydrogens(7.0)
        tidy_filename = f"tidy_{name}.{end}"
        if end == "pdb":
            PDBFile.writeFile(fixer.topology, fixer.positions, open(tidy_filename, "a"))
        elif end == "cif":
            PDBxFile.writeFile(
                fixer.topology, fixer.positions, open(tidy_filename, "a")
            )
        # add filename to registry
        short_name = f"tidy_{name}"
        file_description = "Cleaned File. Missing Hydrogens added."
        path_registry.map_path(short_name, tidy_filename, file_description)
        return f"{file_description} Written to {tidy_filename}"


class SpecializedCleanTool(BaseTool):
    """Standard Cleaning of PDB or CIF files"""

    name = "StandardCleaningTool"
    description = """
    This tool will perform a complete cleaning of a PDB or CIF file.
    Input: PDB or CIF file.
    Output: Cleaned PDB file
    Youl will remove heterogens, add missing atoms and hydrogens, and add solvent."""
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, query: str) -> str:
        """use the tool."""
        try:
            if self.path_registry is None:  # this should not happen
                return "Path registry not initialized"
            clean_tools = CleaningTools()
            return clean_tools._standard_cleaning(query, self.path_registry)
        except FileNotFoundError:
            return "Check your file path. File not found."
        except Exception as e:
            return f"Something went wrong. {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class RemoveWaterCleaningTool(BaseTool):
    """Removes water and adds hydrogens"""

    name = """RemoveWaterAddHydrogensCleaningTool"""
    description = """This is the default cleaning tool.
    If and only if the human wants
    to remove water and heterogens, and add hydrogens.
    This tool will remove water
    and add hydrogens in a pdb or cif file.
    Input: PDB or CIF file.
    Output: Cleaned PDB file
    """

    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, query: str) -> str:
        """use the tool."""
        try:
            if self.path_registry is None:  # this should not happen
                return "Path registry not initialized"
            clean_tools = CleaningTools()
            return clean_tools._add_hydrogens_and_remove_water(
                query, self.path_registry
            )
        except FileNotFoundError:
            return "Check your file path. File not found."
        except Exception as e:
            return f"Something went wrong. {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class AddHydrogensCleaningTool(BaseTool):
    """Adds hydrogens"""

    name = "AddHydrogensCleaningTool"
    description = """
]   This tool only adds hydrogens to a pdb or cif file.
    in a pdb or cif file
    Input: PDB or CIF file.
    Output: Cleaned PDB file
    """

    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, query: str) -> str:
        """use the tool."""
        try:
            if self.path_registry is None:  # this should not happen
                return "Path registry not initialized"
            clean_tools = CleaningTools()
            return clean_tools._add_hydrogens(query, self.path_registry)
        except FileNotFoundError:
            return "Check your file path. File not found."
        except Exception as e:
            return f"Something went wrong. {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
