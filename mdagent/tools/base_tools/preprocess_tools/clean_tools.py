import os
from typing import Optional, Type

from langchain.tools import BaseTool
from openmm.app import PDBFile, PDBxFile
from pdbfixer import PDBFixer
from pydantic import BaseModel, Field

from mdagent.utils import FileType, PathRegistry


class CleaningTools:
    def __init__(self, path_registry):
        self.path_registry = path_registry

    def _standard_cleaning(self, pdbfile: str) -> str:
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
        self.path_registry.map_path(short_name, tidy_filename, file_description)
        return f"{file_description} Written to {tidy_filename}"

    def _remove_water(self, pdbfile: str) -> str:
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
        self.path_registry.map_path(short_name, tidy_filename, file_description)
        return f"{file_description} Written to {tidy_filename}"

    def _add_hydrogens_and_remove_water(self, pdbfile: str) -> str:
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
        self.path_registry.map_path(short_name, tidy_filename, file_description)
        return f"{file_description} Written to {tidy_filename}"

    def _add_hydrogens(self, pdbfile: str) -> str:
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
        self.path_registry.map_path(short_name, tidy_filename, file_description)
        return f"{file_description} Written to {tidy_filename}"


class SpecializedCleanTool(BaseTool):
    """Standard Cleaning of PDB or CIF files"""

    name = "StandardCleaningTool"
    description = """
    This tool will perform a complete cleaning of a PDB or CIF file.
    Input: PDB or CIF file name
    Output: Cleaned PDB file
    You will remove heterogens, add missing atoms and hydrogens, and add solvent."""
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, file_name: str) -> str:
        """use the tool."""
        if self.path_registry is None:
            return "Path registry not initialized"
        try:
            file_path = self.path_registry.get_mapped_path(file_name)
            if file_path is None:
                return "File not found"
            clean_tools = CleaningTools(self.path_registry)
            return clean_tools._standard_cleaning(file_path)
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
    Input: PDB or CIF file name.
    Output: Cleaned PDB file
    """

    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, file_name: str) -> str:
        """use the tool."""
        if self.path_registry is None:
            return "Path registry not initialized"
        try:
            file_path = self.path_registry.get_mapped_path(file_name)
            if file_path is None:
                return "File not found"
            clean_tools = CleaningTools(self.path_registry)
            return clean_tools._add_hydrogens_and_remove_water(file_path)
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
    Input: PDB or CIF file name.
    Output: Cleaned PDB file
    """

    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, file_name: str) -> str:
        """use the tool."""
        if self.path_registry is None:
            return "Path registry not initialized"
        try:
            file_path = self.path_registry.get_mapped_path(file_name)
            if file_path is None:
                return "File not found"
            clean_tools = CleaningTools(self.path_registry)
            return clean_tools._add_hydrogens(file_path)
        except FileNotFoundError:
            return "Check your file path. File not found."
        except Exception as e:
            return f"Something went wrong. {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class CleaningToolFunctionInput(BaseModel):
    """Input model for CleaningToolFunction"""

    pdb_id: str = Field(..., description="ID of the pdb/cif file in the path registry")
    replace_nonstandard_residues: bool = Field(
        True, description="Whether to replace nonstandard residues with standard ones. "
    )
    add_missing_atoms: bool = Field(
        True,
        description="Whether to add missing atoms to the file from the SEQRES records.",
    )
    remove_heterogens: bool = Field(
        True, description="Whether to remove heterogens from the file."
    )
    remove_water: bool = Field(
        True,
        description="""Whether to remove water from the file.
        remove_heterogens must be True.""",
    )
    add_hydrogens: bool = Field(
        True, description="Whether to add hydrogens to the file."
    )
    add_hydrogens_ph: int = Field(7.0, description="pH at which hydrogens are added.")


class CleaningToolFunction(BaseTool):
    name = "CleaningToolFunction"
    description = """
    This tool performs various cleaning operations on a PDB or CIF file.
    Operations can include removing heterogens,
    adding missing atoms and hydrogens,
    replacing nonstandard residues, and/or removing water.

    """
    args_schema: Type[BaseModel] = CleaningToolFunctionInput

    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, **input_args) -> str:
        """Use the tool with specified operations."""
        if self.path_registry is None:
            return "Path registry not initialized"
        try:
            if "input_args" in input_args.keys():
                input_args = input_args["input_args"]
            else:
                input_args = input_args
            pdbfile_id = input_args.get("pdb_id", None)
            if pdbfile_id is None:
                return """No file was provided.
                The input has to be a dictionary with the key 'pdb_id'"""
            remove_heterogens = input_args.get("remove_heterogens", True)
            remove_water = input_args.get("remove_water", True)
            add_hydrogens = input_args.get("add_hydrogens", True)
            add_hydrogens_ph = input_args.get("add_hydrogens_ph", 7.0)
            replace_nonstandard_residues = input_args.get(
                "replace_nonstandard_residues", True
            )
            add_missing_atoms = input_args.get("add_missing_atoms", True)
            input_args.get("output_path", None)

            if self.path_registry is None:
                return "Path registry not initialized"
            file_description = "Cleaned File: "
            try:
                pdbfile_path = self.path_registry.get_mapped_path(pdbfile_id)
                if "/" in pdbfile_path:
                    pdbfile = pdbfile_path.split("/")[-1]
                else:
                    pdbfile = pdbfile_path
                name, end = pdbfile.split(".")

            except Exception as e:
                print(f"error retrieving from path_registry, trying to read file {e}")
                return "File not found in path registry. "
            print(f"file path: {pdbfile_path}")
            fixer = PDBFixer(filename=pdbfile_path)
            try:
                fixer.findMissingResidues()
            except Exception:
                print("error at findMissingResidues")
            try:
                fixer.findNonstandardResidues()
            except Exception:
                print("error at findNonstandardResidues")
            try:
                if remove_heterogens and remove_water:
                    fixer.removeHeterogens(False)
                    file_description += " Removed Heterogens, and Water Removed. "
                elif remove_heterogens and not remove_water:
                    fixer.removeHeterogens(True)
                    file_description += " Removed Heterogens, and Water Kept. "
            except Exception:
                print("error at removeHeterogens")

            try:
                if replace_nonstandard_residues:
                    fixer.replaceNonstandardResidues()
                    file_description += " Replaced Nonstandard Residues. "
            except Exception:
                print("error at replaceNonstandardResidues")
            try:
                fixer.findMissingAtoms()
            except Exception:
                print("error at findMissingAtoms")
            try:
                if add_missing_atoms:
                    fixer.addMissingAtoms()
            except Exception:
                print("error at addMissingAtoms")
            try:
                if add_hydrogens:
                    fixer.addMissingHydrogens(add_hydrogens_ph)
                    file_description += f"Added Hydrogens at pH {add_hydrogens_ph}. "
            except Exception:
                print("error at addMissingHydrogens")

            file_description += (
                "Missing Atoms Added and replaces nonstandard residues. "
            )
            file_mode = "w" if add_hydrogens else "a"
            file_name = self.path_registry.write_file_name(
                type=FileType.PROTEIN,
                protein_name=name.split("_")[0],
                description="Clean",
                file_format=end,
            )
            file_id = self.path_registry.get_fileid(file_name, FileType.PROTEIN)
            directory = "files/pdb"
            if not os.path.exists(directory):
                os.makedirs(directory)
            if end == "pdb":
                PDBFile.writeFile(
                    fixer.topology,
                    fixer.positions,
                    open(f"{directory}/{file_name}", file_mode),
                )
            elif end == "cif":
                PDBxFile.writeFile(
                    fixer.topology,
                    fixer.positions,
                    open(f"{directory}/{file_name}", file_mode),
                )

            self.path_registry.map_path(
                file_id, f"{directory}/{file_name}", file_description
            )
            return f"File cleaned!\nFile ID:{file_id}\nPath:{directory}/{file_name}"
        except FileNotFoundError:
            return "Check your file path. File not found."
        except Exception as e:
            print(e)
            return f"Something went wrong. {e}"

    async def _arun(
        self, query: str, remove_water: bool = False, add_hydrogens: bool = False
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Asynchronous operation not supported yet.")
