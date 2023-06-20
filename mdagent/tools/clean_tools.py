import re

from langchain.tools import BaseTool
from openmm.app import PDBFile, PDBxFile
from pdbfixer import PDBFixer


def _extract_path(user_input: str) -> str:
    """Extract file path from user input."""

    # Remove any leading or trailing white space
    user_input = user_input.strip()

    # Remove single and double quotes from the user_input
    user_input = user_input.replace("'", "")
    user_input = user_input.replace('"', "")

    # Use a regex to find a sequence of non-space characters ending with .pdb or .cif
    match = re.search(r"\b[\w/\\:.]+\.(?:pdb|cif)\b", user_input)

    if match:
        return match.group(0)
    else:
        raise ValueError("No valid file path found in user input.")


def _standard_cleaning(pdbfile: str):
    pdbfile = _extract_path(pdbfile)
    name = pdbfile.split(".")[0]
    end = pdbfile.split(".")[1]
    fixer = PDBFixer(filename=pdbfile)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    if end == "pdb":
        PDBFile.writeFile(
            fixer.topology, fixer.positions, open(f"tidy_{name}.pdb", "a")
        )
    elif end == "cif":
        PDBxFile.writeFile(
            fixer.topology, fixer.positions, open(f"tidy_{name}.cif", "a")
        )
    return f"Cleaned File written to tidy_{name}.pdb"


def _remove_water(pdbfile: str):
    pdbfile = _extract_path(pdbfile)
    name = pdbfile.split(".")[0]
    end = pdbfile.split(".")[1]
    fixer = PDBFixer(filename=pdbfile)
    fixer.removeHeterogens(False)
    if end == "pdb":
        PDBFile.writeFile(
            fixer.topology, fixer.positions, open(f"tidy_{name}.pdb", "a")
        )
    elif end == "cif":
        PDBxFile.writeFile(
            fixer.topology, fixer.positions, open(f"tidy_{name}.cif", "a")
        )
    return f"Cleaned File. Standard cleaning. Written to tidy_{name}.pdb"


def _add_hydrogens_and_remove_water(pdbfile: str):
    pdbfile = _extract_path(pdbfile)
    name = pdbfile.split(".")[0]
    end = pdbfile.split(".")[1]
    fixer = PDBFixer(filename=pdbfile)
    fixer.removeHeterogens(False)
    fixer.addMissingHydrogens(7.0)
    if end == "pdb":
        PDBFile.writeFile(
            fixer.topology, fixer.positions, open(f"tidy_{name}.pdb", "w")
        )
    elif end == "cif":
        PDBxFile.writeFile(
            fixer.topology, fixer.positions, open(f"tidy_{name}.cif", "w")
        )
    return f"""Cleaned File. Missing Hydrogens added, and water
            removed. Written to tidy_{name}.pdb"""


def _add_hydrogens(pdbfile: str):
    pdbfile = _extract_path(pdbfile)
    name = pdbfile.split(".")[0]
    end = pdbfile.split(".")[1]
    fixer = PDBFixer(filename=pdbfile)
    fixer.addMissingHydrogens(7.0)
    if end == "pdb":
        PDBFile.writeFile(
            fixer.topology, fixer.positions, open(f"tidy_{name}.pdb", "a")
        )
    elif end == "cif":
        PDBxFile.writeFile(
            fixer.topology, fixer.positions, open(f"tidy_{name}.cif", "a")
        )
    return f"Cleaned File. Missing Hydrogens added. Written to tidy_{name}.pdb"


class SpecializedCleanTool(BaseTool):
    """Standard Cleaning of PDB or CIF files"""

    name = "Standard Cleaning of PDB or CIF files"
    description = """
    This tool will perform a complete cleaning of a PDB or CIF file.
    Input: PDB or CIF file.
    Output: Cleaned PDB file
    Youl will remove heterogens, add missing atoms and hydrogens, and add solvent."""

    def _run(self, query: str) -> str:
        """use the tool."""
        return _standard_cleaning(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class RemoveWaterCleaningTool(BaseTool):
    """Removes water and adds hydrogens"""

    name = """Cleaning tools that removes water
    and add hydrogens in a pdb or cif file"""
    description = """This is the default cleaning tool.
    If and only if the human wants
    to remove water and heterogens, and add hydrogens.
    This tool will remove water
    and add hydrogens in a pdb or cif file.
    Input: PDB or CIF file.
    Output: Cleaned PDB file
    """

    def _run(self, query: str) -> str:
        return _add_hydrogens_and_remove_water(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class addHydrogensCleaningTool(BaseTool):
    """Adds hydrogens"""

    name = """Cleaning tools that adds hydrogens
    in a pdb or cif file"""
    description = """
]   This tool only adds hydrogens to a pdb or cif file.
    in a pdb or cif file
    Input: PDB or CIF file.
    Output: Cleaned PDB file"""

    def _run(self, query: str) -> str:
        return _add_hydrogens(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
