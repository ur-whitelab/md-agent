import os
import re
import subprocess
import sys
import typing
from typing import Any, Dict, List, Optional, Type, Union

import requests
import streamlit as st
from langchain.tools import BaseTool
from pdbfixer import PDBFixer
from pydantic import BaseModel, Field, ValidationError, root_validator
from rdkit import Chem

from mdagent.utils import FileType, PathRegistry

from .elements import list_of_elements


def get_pdb(query_string, path_registry=None):
    """
    Search RSCB's protein data bank using the given query string
    and return the path to pdb file in either CIF or PDB format
    """
    if path_registry is None:
        path_registry = PathRegistry.get_instance()
    url = "https://search.rcsb.org/rcsbsearch/v2/query?json={search-request}"
    query = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": query_string},
        },
        "return_type": "entry",
    }
    r = requests.post(url, json=query)
    if r.status_code == 204:
        return None
    if "cif" in query_string or "CIF" in query_string:
        filetype = "cif"
    else:
        filetype = "pdb"
    if "result_set" in r.json() and len(r.json()["result_set"]) > 0:
        pdbid = r.json()["result_set"][0]["identifier"]
        print(f"PDB file found with this ID: {pdbid}")
        st.markdown(f"PDB file found with this ID: {pdbid}", unsafe_allow_html=True)
        url = f"https://files.rcsb.org/download/{pdbid}.{filetype}"
        pdb = requests.get(url)
        filename = path_registry.write_file_name(
            FileType.PROTEIN,
            protein_name=pdbid,
            description="raw",
            file_format=filetype,
        )
        file_id = path_registry.get_fileid(filename, FileType.PROTEIN)
        directory = "files/pdb"
        # Create the directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(f"{directory}/{filename}", "w") as file:
            file.write(pdb.text)

        return filename, file_id
    return None


class ProteinName2PDBTool(BaseTool):
    name = "PDBFileDownloader"
    description = (
        "This tool downloads PDB (Protein Data Bank) or"
        "CIF (Crystallographic Information File) files using"
        "a protein's common name (NOT a small molecule)."
        "When a specific file type, either PDB or CIF,"
        "is requested, add file type to the query string with space."
        "Input: Commercial name of the protein or file without"
        "file extension"
        "Output: Corresponding PDB or CIF file"
    )
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, query: str) -> str:
        """Use the tool."""
        try:
            if self.path_registry is None:  # this should not happen
                return "Path registry not initialized"
            filename, pdbfile_id = get_pdb(query, self.path_registry)
            if pdbfile_id is None:
                return "Name2PDB tool failed to find and download PDB file."
            else:
                self.path_registry.map_path(
                    pdbfile_id,
                    f"files/pdb/{filename}",
                    f"PDB file downloaded from RSCB, PDBFile ID: {pdbfile_id}",
                )
                return f"Name2PDB tool successful. downloaded the PDB file:{pdbfile_id}"
        except Exception as e:
            return f"Something went wrong. {e}"

    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("this tool does not support async")


"""validate_pdb_format: validates a pdb file against the pdb format specification
   packmol_wrapper: takes in a list of pdb files, a
   list of number of molecules, a list of instructions, and a list of small molecules
   and returns a packed pdb file
   Molecule: class that represents a molecule (helpful for packmol
   PackmolBox: class that represents a box of molecules (helpful for packmol)
   summarize_errors: function that summarizes the errors found by validate_pdb_format
   _extract_path: function that extracts a file path from a string
   _standard_cleaning: function that cleans a pdb file using pdbfixer)"""

########PDB Validation#########


def validate_pdb_format(fhandle):
    """
    Compare each ATOM/HETATM line with the format defined on the
    official PDB website.

    Parameters
    ----------
    fhandle : a line-by-line iterator of the original PDB file.

    Returns
    -------
    (int, list)
        - 1 if error was found, 0 if no errors were found.
        - List of error messages encountered.
    """
    # check if filename is in directory
    if not os.path.exists(fhandle):
        return (1, ["File not found. Packmol failed to write the file."])
    errors = []
    _fmt_check = (
        ("Atm. Num.", (slice(6, 11), re.compile(r"[\d\s]+"))),
        ("Alt. Loc.", (slice(11, 12), re.compile(r"\s"))),
        ("Atm. Nam.", (slice(12, 16), re.compile(r"\s*[A-Z0-9]+\s*"))),
        ("Spacer #1", (slice(16, 17), re.compile(r"[A-Z0-9 ]{1}"))),
        ("Res. Nam.", (slice(17, 20), re.compile(r"\s*[A-Z0-9]+\s*"))),
        ("Spacer #2", (slice(20, 21), re.compile(r"\s"))),
        ("Chain Id.", (slice(21, 22), re.compile(r"[A-Za-z0-9 ]{1}"))),
        ("Res. Num.", (slice(22, 26), re.compile(r"\s*[\d\-]+\s*"))),
        ("Ins. Code", (slice(26, 27), re.compile(r"[A-Z0-9 ]{1}"))),
        ("Spacer #3", (slice(27, 30), re.compile(r"\s+"))),
        ("Coordn. X", (slice(30, 38), re.compile(r"\s*[\d\.\-]+\s*"))),
        ("Coordn. Y", (slice(38, 46), re.compile(r"\s*[\d\.\-]+\s*"))),
        ("Coordn. Z", (slice(46, 54), re.compile(r"\s*[\d\.\-]+\s*"))),
        ("Occupancy", (slice(54, 60), re.compile(r"\s*[\d\.\-]+\s*"))),
        ("Tmp. Fac.", (slice(60, 66), re.compile(r"\s*[\d\.\-]+\s*"))),
        ("Spacer #4", (slice(66, 72), re.compile(r"\s+"))),
        ("Segm. Id.", (slice(72, 76), re.compile(r"[\sA-Z0-9\-\+]+"))),
        ("At. Elemt", (slice(76, 78), re.compile(r"[\sA-Z0-9\-\+]+"))),
        ("At. Charg", (slice(78, 80), re.compile(r"[\sA-Z0-9\-\+]+"))),
    )

    def _make_pointer(column):
        col_bg, col_en = column.start, column.stop
        pt = ["^" if c in range(col_bg, col_en) else " " for c in range(80)]
        return "".join(pt)

    for iline, line in enumerate(fhandle, start=1):
        line = line.rstrip("\n").rstrip("\r")  # CR/LF
        if not line:
            continue

        if line[0:6] in ["ATOM  ", "HETATM"]:
            # ... [rest of the code unchanged here]
            linelen = len(line)
            if linelen < 80:
                emsg = "[!] Line {0} is short: {1} < 80\n"
                sys.stdout.write(emsg.format(iline, linelen))

            elif linelen > 80:
                emsg = "[!] Line {0} is long: {1} > 80\n"
                sys.stdout.write(emsg.format(iline, linelen))

            for fname, (fcol, fcheck) in _fmt_check:
                field = line[fcol]
                if not fcheck.match(field):
                    pointer = _make_pointer(fcol)
                    emsg = "[!] Offending field ({0}) at line {1}\n".format(
                        fname, iline
                    )
                    emsg += repr(line) + "\n"
                    emsg += pointer + "\n"
                    errors.append(emsg)

        else:
            # ... [rest of the code unchanged here]
            linelen = len(line)
            # ... [rest of the code unchanged here]
            linelen = len(line)
            skip_keywords = (
                "END",
                "ENDMDL",
                "HEADER",
                "TITLE",
                "REMARK",
                "CRYST1",
                "MODEL",
            )

            if any(keyword in line for keyword in skip_keywords):
                continue

            if linelen < 80:
                emsg = "[!] Line {0} is short: {1} < 80\n"
                sys.stdout.write(emsg.format(iline, linelen))
            elif linelen > 80:
                emsg = "[!] Line {0} is long: {1} > 80\n"
                sys.stdout.write(emsg.format(iline, linelen))

    """
    map paths to files in path_registry before you return the string
    same for all other functions you want to save files for next tools
    Don't forget to import PathRegistry and add path_registry
    or PathRegistry as an argument
    """
    if errors:
        msg = "\nTo understand your errors, read the format specification:\n"
        msg += "http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM\n"
        errors.append(msg)
        return (1, errors)
    else:
        return (0, ["It *seems* everything is OK."])


##########################PACKMOL###############################


def summarize_errors(errors):
    error_summary = {}

    # Regular expression pattern to capture the error type and line number
    pattern = r"\[!\] Offending field \((.+?)\) at line (\d+)"

    for error in errors:
        match = re.search(pattern, error)
        if match:
            error_type, line_number = match.groups()
            # If this error type hasn't been seen before,
            # initialize it in the dictionary
            if error_type not in error_summary:
                error_summary[error_type] = {"lines": []}
            error_summary[error_type]["lines"].append(line_number)

    # Format the summarized errors for display
    summarized_strings = []
    for error_type, data in error_summary.items():
        line_count = len(data["lines"])
        if line_count > 3:
            summarized_strings.append(f"{error_type}: total {line_count} lines")
        else:
            summarized_strings.append(f"{error_type}: lines: {','.join(data['lines'])}")

    return summarized_strings


class Molecule:
    def __init__(self, filename, file_id, number_of_molecules=1, instructions=None):
        self.filename = filename
        self.id = file_id
        self.number_of_molecules = number_of_molecules
        self.instructions = instructions if instructions else []
        self.load()

    def load(self):
        # load the molecule data (optional)
        pass

    def get_number_of_atoms(self):
        # return the number of atoms in this molecule
        pass


class PackmolBox:
    def __init__(
        self, file_number=1, file_description="PDB file for simulation with: \n"
    ):
        self.molecules = []
        self.file_number = 1
        self.file_description = file_description
        self.final_name = None

    def add_molecule(self, molecule):
        self.molecules.append(molecule)
        self.file_description += f"""{molecule.number_of_molecules} of
        {molecule.filename} as {molecule.instructions} \n"""

    def generate_input_header(self):
        # Generate the header of the input file in .inp format
        orig_pdbs_ids = [
            f"{molecule.number_of_molecules}_{molecule.id}"
            for molecule in self.molecules
        ]

        _final_name = f'{"_and_".join(orig_pdbs_ids)}'

        self.file_description = (
            "Packed Structures of the following molecules:\n"
            + "\n".join(
                [
                    f"Molecule ID: {molecule.id}, "
                    f"Number of Molecules: {molecule.number_of_molecules}"
                    for molecule in self.molecules
                ]
            )
        )
        while os.path.exists(f"files/pdb/{_final_name}_v{self.file_number}.pdb"):
            self.file_number += 1

        self.final_name = f"{_final_name}_v{self.file_number}.pdb"
        with open("packmol.inp", "w") as out:
            out.write("##Automatically generated by LangChain\n")
            out.write("tolerance 2.0\n")
            out.write("filetype pdb\n")
            out.write(
                f"output {self.final_name}\n"
            )  # this is the name of the final file
            out.close()

    def generate_input(self):
        input_data = []
        for molecule in self.molecules:
            input_data.append(f"structure {molecule.filename}")
            input_data.append(f"  number {molecule.number_of_molecules}")
            for idx, instruction in enumerate(molecule.instructions):
                input_data.append(f"  {molecule.instructions[idx]}")
            input_data.append("end structure")

        # Convert list of input data to a single string
        return "\n".join(input_data)

    def run_packmol(self, PathRegistry):
        # Use the generated input to execute Packmol
        input_string = self.generate_input()
        # Write the input to a file
        with open("packmol.inp", "a") as f:
            f.write(input_string)
        # Here, run Packmol using the subprocess module or similar
        cmd = "packmol < packmol.inp"
        result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        if result.returncode != 0:
            print("Packmol failed to run with 'packmol < packmol.inp' command")
            result = subprocess.run(
                "./" + cmd, shell=True, text=True, capture_output=True
            )
            if result.returncode != 0:
                print("Packmol failed to run with './packmol < packmol.inp' command")
                return (
                    "Packmol failed to run. Please check the input file and try again."
                )

        # validate final pdb
        pdb_validation = validate_pdb_format(f"{self.final_name}")
        if pdb_validation[0] == 0:
            # delete .inp files
            os.remove("packmol.inp")
            for molecule in self.molecules:
                os.remove(molecule.filename)
            # name of packed pdb file
            time_stamp = PathRegistry.get_timestamp()[-6:]
            os.rename(self.final_name, f"files/pdb/{self.final_name}")
            PathRegistry.map_path(
                f"PACKED_{time_stamp}",
                f"files/pdb/{self.final_name}",
                self.file_description,
            )
            # move file to files/pdb
            print("succesfull!")
            return f"PDB file validated successfully. FileID: PACKED_{time_stamp}"
        elif pdb_validation[0] == 1:
            # format pdb_validation[1] list of errors
            errors = summarize_errors(pdb_validation[1])
            # delete .inp files

            os.remove("packmol.inp")
            print("errors:", f"{errors}")
            return "PDB file not validated, errors found {}".format(("\n").join(errors))


# define function that takes in a list of
#  molecules and a list of instructions and returns a pdb file


def packmol_wrapper(
    PathRegistry,
    pdbfiles: List,
    files_id: List,
    number_of_molecules: List,
    instructions: List[List],
):
    """Useful when you need to create a box
    of different types of molecules molecules"""

    # create a box
    box = PackmolBox()
    # add molecules to the box
    for (
        pdbfile,
        file_id,
        number_of_molecules,
        instructions,
    ) in zip(pdbfiles, files_id, number_of_molecules, instructions):
        molecule = Molecule(pdbfile, file_id, number_of_molecules, instructions)
        box.add_molecule(molecule)
    # generate input header
    box.generate_input_header()
    # generate input
    # run packmol
    print("Packing:", box.file_description, "\nThe file name is:", box.final_name)
    return box.run_packmol(PathRegistry)


"""Args schema for packmol_wrapper tool. Useful for OpenAI functions"""
##TODO


class PackmolInput(BaseModel):
    pdbfiles_id: typing.Optional[typing.List[str]] = Field(
        ..., description="List of PDB files id (path_registry) to pack into a box"
    )
    number_of_molecules: typing.Optional[typing.List[int]] = Field(
        ..., description="List of number of molecules to pack into a box"
    )
    instructions: typing.Optional[typing.List[List[str]]] = Field(
        ...,
        description=(
            "List of instructions for each molecule. "
            "One List per Molecule. "
            "Every instruction should be one string like:\n"
            "'inside box 0. 0. 0. 90. 90. 90.'"
        ),
    )
    small_molecules: typing.Optional[typing.List[str]] = Field(
        ..., description=("List of small molecules to be downloaded from MolPDB")
    )


class PackMolTool(BaseTool):
    name: str = "packmol_tool"
    description: str = (
        "Useful when you need to create a box "
        "of different types of molecules.\n"
        "Three different examples:\n"
        "pdbfiles_id: ['1a2b_123456', 'water_000000']\n"
        "number_of_molecules: [1, 1000]\n"
        "instructions: [['fixed 0. 0. 0. 0. 0. 0. \n centerofmass'], "
        "['inside box 0. 0. 0. 90. 90. 90.']]\n"
        "will pack 1 molecule of 1a2b_123456 at the origin "
        "and 1000 molecules of water_000000. \n"
        "pdbfiles_id: ['1a2b_123456']\n"
        "number_of_molecules: [1]\n"
        "instructions: [['fixed  0. 0. 0. 0. 0. 0.' \n center]]\n"
        "This will fix the barocenter of protein 1a2b_123456 at "
        "the center of the box with no rotation.\n"
        "pdbfiles_id: ['1a2b_123456']\n"
        "number_of_molecules: [1]\n"
        "instructions: [['outside sphere 2.30 3.40 4.50 8.0]]\n"
        "This will place the protein 1a2b_123456 outside a sphere "
        "centered at 2.30 3.40 4.50 with radius 8.0\n"
    )

    args_schema: Type[BaseModel] = PackmolInput

    path_registry: typing.Optional[PathRegistry]

    def __init__(self, path_registry: typing.Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _get_sm_pdbs(self, small_molecules):
        all_files = self.path_registry.list_path_names()
        print(all_files)
        for molecule in small_molecules:
            # check path registry for molecule.pdb
            if molecule not in all_files:
                # download molecule using small_molecule_pdb from MolPDB
                molpdb = MolPDB()
                molpdb.small_molecule_pdb(molecule, self.path_registry)

    def _run(self, **values) -> str:
        """use the tool."""

        if self.path_registry is None:  # this should not happen
            raise ValidationError("Path registry not initialized")
        try:
            values = self.validate_input(values)
        except ValidationError as e:
            return str(e)
        error_msg = values.get("error", None)
        pdbfile_ids = values.get("pdbfiles_id", [])
        pdbfiles = [
            self.path_registry.get_mapped_path(pdbfile) for pdbfile in pdbfile_ids
        ]
        pdbfile_names = [pdbfile.split("/")[-1] for pdbfile in pdbfiles]
        # copy them to the current directory with temp_ names
        for pdbfile, pdbfile_name in zip(pdbfiles, pdbfile_names):
            os.system(f"cp {pdbfile} temp_{pdbfile_name}")
        pdbfile_names = [f"temp_{pdbfile_name}" for pdbfile_name in pdbfile_names]
        number_of_molecules = values.get("number_of_molecules", [])
        instructions = values.get("instructions", [])
        small_molecules = values.get("small_molecules", [])
        # make sure small molecules are all downloaded
        self._get_sm_pdbs(small_molecules)
        if error_msg:
            return error_msg
        # check if packmol is installed
        cmd = "command -v packmol"
        result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        if result.returncode != 0:
            result = subprocess.run(
                "./" + cmd, shell=True, text=True, capture_output=True
            )
            if result.returncode != 0:
                return (
                    "Packmol is not installed. Please install"
                    "packmol at "
                    "'https://m3g.github.io/packmol/download.shtml'"
                    "and try again."
                )

        return packmol_wrapper(
            self.path_registry,
            pdbfiles=pdbfile_names,
            files_id=pdbfile_ids,
            number_of_molecules=number_of_molecules,
            instructions=instructions,
        )

    def validate_input(cls, values: Union[str, Dict[str, Any]]) -> Dict:
        # check if is only a string
        if isinstance(values, str):
            print("values is a string", values)
            raise ValidationError("Input must be a dictionary")
        pdbfiles = values.get("pdbfiles_id", [])
        number_of_molecules = values.get("number_of_molecules", [])
        instructions = values.get("instructions", [])

        if not (len(pdbfiles) == len(number_of_molecules) == len(instructions)):
            return {
                "error": (
                    "The lengths of pdbfiles, number_of_molecules, "
                    "and instructions must be equal to use this tool."
                )
            }

        for instruction in instructions:
            if len(instruction) != 1:
                return {
                    "error": (
                        "Each instruction must be a single string. "
                        "If necessary, use newlines in a instruction string."
                    )
                }
            # TODO enhance this validation with more packmol instructions
            first_word = instruction[0].split(" ")[0]
            if first_word == "center":
                if len(instruction[0].split(" ")) == 1:
                    return {
                        "error": (
                            "The instruction 'center' must be accompanied by more "
                            "instructions. Example 'fixed 0. 0. 0. 0. 0. 0.' "
                            "The complete instruction would be: 'center \n fixed 0. 0. "
                            "0. 0. 0. 0.' with a newline separating the two "
                            "instructions."
                        )
                    }
            elif first_word not in [
                "inside",
                "outside",
                "fixed",
            ]:
                return {
                    "error": (
                        "The first word of each instruction must be one of "
                        "'inside' or 'outside' or 'fixed' \n"
                        "examples: center \n fixed 0. 0. 0. 0. 0. 0.,\n"
                        "inside box -10. 0. 0. 10. 10. 10. \n"
                    )
                }

        # Further validation, e.g., checking if files exist
        registry = PathRegistry()
        file_ids = registry.list_path_names()

        for pdbfile_id in pdbfiles:
            if "_" not in pdbfile_id:
                return {
                    "error": (
                        f"{pdbfile_id} is not a valid pdbfile_id" "in the path_registry"
                    )
                }
            if pdbfile_id not in file_ids:
                # look for files in the current directory
                # that match some part of the pdbfile
                ids_w_description = registry.list_path_names_and_descriptions()

                return {
                    "error": (
                        f"PDB file ID {pdbfile_id} does not exist "
                        "in the path registry.\n"
                        f"This are the files IDs: {ids_w_description} "
                    )
                }
        return values

    async def _arun(self, values: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


########VALIDATION AND FIXING PDB FILES########################


class PDBsummarizerfxns:
    def __init__(self):
        self.list_of_elements = list_of_elements

    def _record_inf(self, pdbfile):
        with open(pdbfile, "r") as f:
            lines = f.readlines()
            remarks = [
                record_lines
                for record_lines in lines
                if record_lines.startswith("REMARK")
            ]
            atoms = [
                record_lines
                for record_lines in lines
                if record_lines.startswith("ATOM")
            ]
            box = [
                record_lines
                for record_lines in lines
                if record_lines.startswith("CRYST1")
            ]
            HETATM = [
                record_lines
                for record_lines in lines
                if record_lines.startswith("HETATM")
            ]

        return remarks, atoms, box, HETATM

    def _num_of_dif_residues(self, pdbfile):
        remarks, atoms, box, HETATM = self._record_inf(pdbfile)
        residues = [atom[17:20] for atom in atoms]
        residues = list(set(residues))
        return len(residues)

    # diagnosis
    """Checks for the elements names in the pdb file.
    Positions 76-78 of the ATOM and HETATM records"""

    def _atoms_have_elements(self, pdbfile):
        _, atoms, _, _ = self._record_inf(pdbfile)
        print(atoms)
        elements = [atom[76:78] for atom in atoms if atom not in [" ", "", "  ", "   "]]
        print(elements)
        if len(elements) != len(atoms):
            print(
                (
                    "No elements in the ATOM records there are"
                    "{len(elements)} elements and {len(atoms)}"
                    "atoms records"
                )
            )
            return False
        elements = list(set(elements))
        for element in elements:
            if element not in self.list_of_elements:
                print("Element not in the list of elements")
                return False
        return True

    def _atoms_have_tempFactor(self, pdbfile):
        _, atoms, _, _ = self._record_inf(pdbfile)
        tempFactor = [
            atom[60:66]
            for atom in atoms
            if atom[60:66] not in [" ", "", "  ", "   ", "  ", "      "]
        ]
        if len(tempFactor) != len(atoms):
            return False
        return True

    def _atoms_have_occupancy(self, pdbfile):
        _, atoms, _, _ = self._record_inf(pdbfile)
        occupancy = [
            atom[54:60]
            for atom in atoms
            if atom[54:60] not in [" ", "", "  ", "   ", "  ", "      "]
        ]
        if len(occupancy) != len(atoms):
            return False
        return True

    def _hetatom_have_occupancy(self, pdbfile):
        _, _, _, HETATM = self._record_inf(pdbfile)
        occupancy = [
            atom[54:60]
            for atom in HETATM
            if atom[54:60] not in [" ", "", "  ", "   ", "  ", "      "]
        ]
        if len(occupancy) != len(HETATM):
            return False
        return True

    def _hetatm_have_elements(self, pdbfile):
        _, _, _, HETATM = self._record_inf(pdbfile)
        elements = [
            atom[76:78] for atom in HETATM if atom[76:78] not in [" ", "", "  ", "   "]
        ]
        if len(elements) != len(HETATM):
            print("No elements in the HETATM records")
            return False
        return True

    def _hetatm_have_tempFactor(self, pdbfile):
        _, _, _, HETATM = self._record_inf(pdbfile)
        tempFactor = [
            atom[60:66] for atom in HETATM if atom not in [" ", "", "  ", "   "]
        ]
        if len(tempFactor) != len(HETATM):
            return False
        return True

    """Checks for the residue names in the pdb file.
      Positions 17-20 of the ATOM and HETATM records"""

    def _atoms_hetatm_have_residue_names(self, pdbfile):
        _, atoms, _, HETATM = self._record_inf(pdbfile)
        residues = [atom[17:20] for atom in atoms]
        residues = list(set(residues))
        if len(residues) != len(atoms):
            return False
        residues = [atom[17:20] for atom in HETATM]
        residues = list(set(residues))
        if len(residues) != len(HETATM):
            return False
        return True

    def _atoms_hetatm_have_occupancy(self, pdbfile):
        _, atoms, _, HETATM = self._record_inf(pdbfile)
        occupancy = [
            atom[54:60]
            for atom in atoms
            if atom not in [" ", "", "  ", "   ", "  ", "      "]
        ]
        if len(occupancy) != len(atoms):
            return False
        occupancy = [
            HET[54:60]
            for HET in HETATM
            if HET not in [" ", "", "  ", "   ", "  ", "      "]
        ]
        if len(occupancy) != len(HETATM):
            return False
        return True

    def _non_standard_residues(self, pdbfile):
        fixer = PDBFixer(file_name=pdbfile)
        fixer.findNonstandardResidues()
        len(fixer.nonstandardResidues)


def pdb_summarizer(pdb_file):
    pdb = PDBsummarizerfxns()
    pdb.remarks, pdb.atoms, pdb.box, pdb.HETATM = pdb._record_inf(pdb_file)
    pdb.atoms_elems = pdb._atoms_have_elements(pdb_file)
    pdb.HETATM_elems = pdb._hetatm_have_elements(pdb_file)
    pdb.residues = pdb._atoms_hetatm_have_residue_names(pdb_file)
    pdb.atoms_tempFact = pdb._atoms_have_tempFactor(pdb_file)
    pdb.num_of_residues = pdb._num_of_dif_residues(pdb_file)
    pdb.HETATM_tempFact = pdb._hetatm_have_tempFactor(pdb_file)

    output = (
        f"PDB file: {pdb_file} has the following properties:"
        "Number of residues: {pdb.num_of_residues}"
        "Are elements identifiers present: {pdb.atoms}"
        "Are HETATM elements identifiers present: {pdb.HETATM}"
        "Are residue names present: {pdb.residues}"
        "Are box dimensions present: {pdb.box}"
        "Non-standard residues: {pdb.HETATM}"
    )
    return output


def _fix_element_column(pdb_file, custom_element_dict=None):
    records = ("ATOM", "HETATM", "ANISOU")
    corrected_lines = []
    for line in pdb_file:
        if line.startswith(records):
            atom_name = line[12:16]

            if atom_name[0].isalpha() and not atom_name[2:].isdigit():
                element = atom_name.strip()
            else:
                atom_name = atom_name.strip()
                if atom_name[0].isdigit():
                    element = atom_name[1]
                else:
                    element = atom_name[0]

            if element not in set(list_of_elements):
                element = "  "  # empty element in case we cannot assign

            line = line[:76] + element.rjust(2) + line[78:]
            corrected_lines.append(line)

    return corrected_lines


def fix_element_column(pdb_file, custom_element_dict=None):
    """Fixes the Element columns of a pdb file"""

    # extract Title, Header, Remarks, and Cryst1 records
    file_name = pdb_file.split(".")[0]
    # check if theres a file-name-fixed.pdb file
    if os.path.isfile(file_name + "-fixed.pdb"):
        pdb_file = file_name + "-fixed.pdb"
    assert isinstance(pdb_file, str), "pdb_file must be a string"
    with open(pdb_file, "r") as f:
        print("I read the initial file")
        pdb_file_lines = f.readlines()
        # validate if pdbfile has element records
        pdb = PDBsummarizerfxns()
        atoms_have_elems, HETATM_have_elems = pdb._atoms_have_elements(
            pdb_file
        ), pdb._hetatm_have_elements(pdb_file)
        if atoms_have_elems and HETATM_have_elems:
            f.close()
            return (
                "Element's column already filled with"
                "elements, no fix needed for elements"
            )
        print("I closed the initial file")
        f.close()

    # fix element column
    records = ("TITLE", "HEADER", "REMARK", "CRYST1", "HET", "LINK", "SEQRES")
    final_records = ("CONECT", "MASTER", "END")
    _unchanged_records = []
    _unchanged_final_records = []
    print("pdb_file", pdb_file)
    for line in pdb_file_lines:
        if line.startswith(records):
            _unchanged_records.append(line)
        elif line.startswith(final_records):
            _unchanged_final_records.append(line)
    print("_unchanged_records", _unchanged_records)
    new_pdb = _fix_element_column(pdb_file_lines, custom_element_dict)
    # join the linees
    new_pdb = "".join(new_pdb)
    # write new pdb file as pdb_file-fixed.pdb
    new_pdb_file = file_name.split(".")[0] + "-fixed.pdb"
    print("name of fixed pdb file", new_pdb_file)
    # write the unchanged records first and then the new pdb file
    assert isinstance(new_pdb_file, str), "new_pdb_file must be a string"
    with open(new_pdb_file, "w") as f:
        print("I wrote the new file")
        f.writelines(_unchanged_records)
        f.write(new_pdb)
        f.writelines(_unchanged_final_records)
        f.close()
    try:
        # read the new pdb file and check if it has element records
        with open(new_pdb_file, "r") as f:
            pdb_file_lines = f.readlines()
            pdb = PDBsummarizerfxns()
            atoms_have_elems, HETATM_have_elems = pdb._atoms_have_elements(
                new_pdb_file
            ), pdb._hetatm_have_elements(new_pdb_file)
            if atoms_have_elems and HETATM_have_elems:
                f.close()
                return "Element's column fixed successfully"
            else:
                f.close()
                return "Element's column not fixed, and i dont know why"
    except Exception as e:
        return f"Element's column not fixed error: {e}"


class FixElementColumnArgs(BaseTool):
    # arguments of fix_element_column
    pdb_file: str = Field(..., description="PDB file to be fixed")
    custom_element_dict: dict = Field(
        None,
        description=(
            "Custom element dictionary. If None," "the default dictionary is used"
        ),
    )


def pad_line(line):
    """Pad line to 80 characters in case it is shorter."""
    size_of_line = len(line)
    if size_of_line < 80:
        padding = 80 - size_of_line + 1
        line = line.strip("\n") + " " * padding + "\n"
    return line[:81]  # 80 + newline character


def _fix_temp_factor_column(pdbfile, bfactor, only_fill):
    """Set the temperature column in all ATOM/HETATM records to a given value.

    This function is a generator.

    Parameters
    ----------
    fhandle : a line-by-line iterator of the original PDB file.

    bfactor : float
        The desired bfactor.

    Yields
    ------
    str (line-by-line)
        The modified (or not) PDB line."""
    _pad_line = pad_line
    records = ("ATOM", "HETATM")
    corrected_lines = []
    bfactor = "{0:>6.2f}".format(bfactor)

    for line in pdbfile:
        if line.startswith(records):
            line = _pad_line(line)
            if only_fill:
                if line[60:66].strip() == "":
                    corrected_lines.append(line[:60] + bfactor + line[66:])
            else:
                corrected_lines.append(line[:60] + bfactor + line[66:])
        else:
            corrected_lines.append(line)

    return corrected_lines


def fix_temp_factor_column(pdb_file, bfactor=1.00, only_fill=True):
    """Fixes the tempFactor columns of a pdb file"""

    # extract Title, Header, Remarks, and Cryst1 records
    # get name from pdb_file
    if isinstance(pdb_file, str):
        file_name = pdb_file.split(".")[0]
    else:
        return "pdb_file must be a string"
    file_name = pdb_file.split(".")[0]

    if os.path.isfile(file_name + "-fixed.pdb"):
        file_name = file_name + "-fixed.pdb"

    assert isinstance(file_name, str), "pdb_file must be a string"
    with open(file_name, "r") as f:
        print("im reading the files temp factor")
        pdb_file_lines = f.readlines()
        # validate if pdbfile has temp factors
        pdb = PDBsummarizerfxns()
        atoms_have_bfactor, HETATM_have_bfactor = pdb._atoms_have_tempFactor(
            pdb_file
        ), pdb._hetatm_have_tempFactor(pdb_file)
        if atoms_have_bfactor and HETATM_have_bfactor and only_fill:
            # print("Im closing the file temp factor")
            f.close()
            return (
                "TempFact column filled with bfactor already,"
                "no fix needed for temp factor"
            )
        f.close()
    # fix element column
    records = ("TITLE", "HEADER", "REMARK", "CRYST1", "HET", "LINK", "SEQRES")
    final_records = ("CONECT", "MASTER", "END")
    _unchanged_final_records = []
    _unchanged_records = []
    for line in pdb_file_lines:
        if line.startswith(records):
            _unchanged_records.append(line)
        elif line.startswith(final_records):
            _unchanged_final_records.append(line)

    new_pdb = _fix_temp_factor_column(pdb_file_lines, bfactor, only_fill)
    # join the linees
    new_pdb = "".join(new_pdb)
    # write new pdb file as pdb_file-fixed.pdb
    new_pdb_file = file_name + "-fixed.pdb"
    # organize columns HEADER, TITLE, REMARKS, CRYST1, ATOM, HETATM, CONECT, MASTER, END

    assert isinstance(new_pdb_file, str), "new_pdb_file must be a string"
    # write new pdb file as pdb_file-fixed.pdb
    with open(new_pdb_file, "w") as f:
        f.writelines(_unchanged_records)
        f.write(new_pdb)
        f.writelines(_unchanged_final_records)
        f.close()
    try:
        # read the new pdb file and check if it has element records
        with open(new_pdb_file, "r") as f:
            pdb_file = f.readlines()
            pdb = PDBsummarizerfxns()
            atoms_have_bfactor, HETATM_have_bfactor = pdb._atoms_have_tempFactor(
                new_pdb_file
            ), pdb._hetatm_have_tempFactor(new_pdb_file)
            if atoms_have_bfactor and HETATM_have_bfactor:
                f.close()
                return "TempFact fixed successfully"
            else:
                f.close()
                return "TempFact column not fixed"
    except Exception as e:
        return f"Couldnt read written file TempFact column not fixed error: {e}"


class FixTempFactorColumnArgs(BaseTool):
    # arguments of fix_element_column
    pdb_file: str = Field(..., description="PDB file to be fixed")
    bfactor: float = Field(1.0, description="Bfactor value to use")
    only_fill: bool = Field(
        True,
        description=(
            "Only fill empty bfactor columns."
            "Avoids replacing existing values."
            "False if you want to replace all values"
            "with the bfactor value"
        ),
    )


def _fix_occupancy_column(pdbfile, occupancy, only_fill):
    """
    Set the occupancy column in all ATOM/HETATM records to a given value.

    Non-ATOM/HETATM lines are give as are. This function is a generator.

    Parameters
    ----------
    fhandle : a line-by-line iterator of the original PDB file.

    occupancy : float
        The desired occupancy value

    Yields
    ------
    str (line-by-line)
        The modified (or not) PDB line.
    """

    records = ("ATOM", "HETATM")
    corrected_lines = []
    occupancy = "{0:>6.2f}".format(occupancy)
    for line in pdbfile:
        if line.startswith(records):
            line = pad_line(line)
            if only_fill:
                if line[54:60].strip() == "":
                    corrected_lines.append(line[:54] + occupancy + line[60:])
            else:
                corrected_lines.append(line[:54] + occupancy + line[60:])
        else:
            corrected_lines.append(line)

    return corrected_lines


def fix_occupancy_columns(pdb_file, occupancy=1.0, only_fill=True):
    """Fixes the occupancy columns of a pdb file"""
    # extract Title, Header, Remarks, and Cryst1 records
    # get name from pdb_file
    file_name = pdb_file.split(".")[0]
    if os.path.isfile(file_name + "-fixed.pdb"):
        file_name = file_name + "-fixed.pdb"

    assert isinstance(pdb_file, str), "pdb_file must be a string"
    with open(file_name, "r") as f:
        pdb_file_lines = f.readlines()
        # validate if pdbfile has occupancy
        pdb = PDBsummarizerfxns()
        atoms_have_bfactor, HETATM_have_bfactor = pdb._atoms_have_occupancy(
            file_name
        ), pdb._hetatom_have_occupancy(file_name)
        if atoms_have_bfactor and HETATM_have_bfactor and only_fill:
            f.close()
            return (
                "Occupancy column filled with occupancy"
                "already, no fix needed for occupancy"
            )
        f.close()
    # fix element column
    records = ("TITLE", "HEADER", "REMARK", "CRYST1", "HET", "LINK", "SEQRES")
    final_records = ("CONECT", "MASTER", "END")
    _unchanged_records = []
    _unchanged_final_records = []
    for line in pdb_file_lines:
        if line.startswith(records):
            _unchanged_records.append(line)
        elif line.startswith(final_records):
            _unchanged_final_records.append(line)

    new_pdb = _fix_occupancy_column(pdb_file_lines, occupancy, only_fill)
    # join the linees
    new_pdb = "".join(new_pdb)
    # write new pdb file as pdb_file-fixed.pdb
    new_pdb_file = file_name + "-fixed.pdb"

    # write new pdb file as pdb_file-fixed.pdb
    assert isinstance(new_pdb_file, str), "new_pdb_file must be a string"
    with open(new_pdb_file, "w") as f:
        f.writelines(_unchanged_records)
        f.write(new_pdb)
        f.writelines(_unchanged_final_records)
        f.close()
    try:
        # read the new pdb file and check if it has element records
        with open(new_pdb_file, "r") as f:
            pdb_file = f.readlines()
            pdb = PDBsummarizerfxns()
            atoms_have_bfactor, HETATM_have_bfactor = pdb._atoms_have_tempFactor(
                new_pdb_file
            ), pdb._hetatm_have_tempFactor(new_pdb_file)
            if atoms_have_bfactor and HETATM_have_bfactor:
                f.close()
                return "Occupancy fixed successfully"
            else:
                f.close()
                return "Occupancy column not fixed"
    except Exception:
        return "Couldnt read file Occupancy's column not fixed"


class FixOccupancyColumnArgs(BaseTool):
    # arguments of fix_element_column
    pdb_file: str = Field(..., description="PDB file to be fixed")
    occupancy: float = Field(1.0, description="Occupancy value to be set")
    only_fill: bool = Field(
        True,
        description=(
            "Only fill empty occupancy columns."
            "Avoids replacing existing values."
            "False if you want to replace all"
            "values with the occupancy value"
        ),
    )


# Define a mapping between query keys and functions.
# If a function requires additional arguments from the query, define it as a lambda.
FUNCTION_MAP = {
    "ElemColum": lambda pdbfile, params: fix_element_column(pdbfile),
    "tempFactor": lambda pdbfile, params: fix_temp_factor_column(pdbfile, *params),
    "Occupancy": lambda pdbfile, params: fix_occupancy_columns(pdbfile, *params),
}


def apply_fixes(pdbfile, query):
    # Iterate through the keys and functions in FUNCTION_MAP.
    for key, func in FUNCTION_MAP.items():
        # Check if the current key is in the query and is not None.
        params = query.get(key)
        if params is not None:
            # If it is, call the function with
            # pdbfile and the parameters from the query.
            func(pdbfile, params)

            return "PDB file fixed"


class PDBFilesFixInp(BaseModel):
    pdbfile: str = Field(..., description="PDB file to be fixed")
    ElemColum: typing.Optional[bool] = Field(
        False,
        description=(
            "List of fixes to be applied. If None, a"
            "validation of what fixes are needed is performed."
        ),
    )
    tempFactor: typing.Optional[typing.Tuple[float, bool]] = Field(
        (...),
        description=(
            "Tuple of     ( float, bool)"
            "first arg is the"
            "value to be set as the tempFill, and third arg indicates"
            "if only empty TempFactor columns have to be filled"
        ),
    )
    Occupancy: typing.Optional[typing.Tuple[float, bool]] = Field(
        (...),
        description=(
            "Tuple of (bool, float, bool)"
            "where first arg indicates if Occupancy"
            "fix has to be applied, second arg is the"
            "value to be set, and third arg indicates"
            "if only empty Occupancy columns have to be filled"
        ),
    )

    @root_validator
    def validate_input(cls, values: Union[str, Dict[str, Any]]) -> Dict:
        if isinstance(values, str):
            print("values is a string", values)
            raise ValidationError("Input must be a dictionary")

        pdbfile = values.get("pdbfiles", "")
        occupancy = values.get("occupancy")
        tempFactor = values.get("tempFactor")
        ElemColum = values.get("ElemColum")

        if occupancy is None and tempFactor is None and ElemColum is None:
            if pdbfile == "":
                return {"error": "No inputs given, failed use of tool."}
            else:
                return values
        else:
            if occupancy:
                if len(occupancy) != 2:
                    return {
                        "error": (
                            "if you want to fix the occupancy"
                            "column argument must be a tuple of (bool, float)"
                        )
                    }
                if not isinstance(occupancy[0], float):
                    return {"error": "occupancy first arg must be a float"}
                if not isinstance(occupancy[1], bool):
                    return {"error": "occupancy second arg must be a bool"}
            if tempFactor:
                if len(tempFactor != 2):
                    return {
                        "error": (
                            "if you want to fix the tempFactor"
                            "column argument must be a tuple of (float, bool)"
                        )
                    }
                if not isinstance(tempFactor[0], bool):
                    return {"error": "occupancy first arg must be a float"}
                if not isinstance(tempFactor[1], float):
                    return {"error": "tempFactor second arg must be a float"}
            if ElemColum is not None:
                if not isinstance(ElemColum[1], bool):
                    return {"error": "ElemColum must be a bool"}
            return values


class FixPDBFile(BaseTool):
    name: str = "PDBFileFixer"
    description: str = "Fixes PDB files columns if needed"
    args_schema: Type[BaseModel] = PDBFilesFixInp

    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, query: Dict):
        """use the tool."""
        if self.path_registry is None:
            raise ValidationError("Path registry not initialized")
        error_msg = query.get("error")
        if error_msg:
            return error_msg
        pdbfile = query.pop("pdbfile")
        if len(query.keys()) == 0:
            validation = validate_pdb_format(pdbfile)
            if validation[0] == 0:
                return "PDB file is valid, no need to fix it"

            if validation[0] == 1:
                # Convert summarized_errors into a set for O(1) lookups
                error_set = set(validation[1])

                # Apply Fixes
                if "At. Elem." in error_set:
                    fix_element_column(pdbfile)
                if "Tmp. Fac." in error_set:
                    fix_temp_factor_column(pdbfile)
                if "Occupancy" in error_set:
                    fix_occupancy_columns(pdbfile)

                validate = validate_pdb_format(pdbfile + "-fixed.pdb")
                if validate[0] == 0:
                    name = pdbfile + "-fixed.pdb"
                    description = "PDB file fixed"
                    self.path_registry.map_path(name, name, description)
                    return "PDB file fixed"
                else:
                    return "PDB not fully fixed"
        else:
            apply_fixes(pdbfile, query)
            validate = validate_pdb_format(pdbfile + "-fixed.pdb")
            if validate[0] == 0:
                name = pdbfile + "-fixed.pdb"
                description = "PDB file fixed"
                self.path_registry.map_path(name, name, description)
                return "PDB file fixed"
            else:
                return "PDB not fully fixed"


class MolPDB:
    def is_smiles(self, text: str) -> bool:
        try:
            m = Chem.MolFromSmiles(text, sanitize=False)
            if m is None:
                return False
            return True
        except Exception:
            return False

    def largest_mol(
        self, smiles: str
    ) -> (
        str
    ):  # from https://github.com/ur-whitelab/chemcrow-public/blob/main/chemcrow/utils.py
        ss = smiles.split(".")
        ss.sort(key=lambda a: len(a))
        while not self.is_smiles(ss[-1]):
            rm = ss[-1]
            ss.remove(rm)
        return ss[-1]

    def molname2smiles(
        self, query: str
    ) -> (
        str
    ):  # from https://github.com/ur-whitelab/chemcrow-public/blob/main/chemcrow/tools/databases.py
        url = " https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}"
        r = requests.get(url.format(query, "property/IsomericSMILES/JSON"))
        # convert the response to a json object
        data = r.json()
        # return the SMILES string
        try:
            smi = data["PropertyTable"]["Properties"][0]["IsomericSMILES"]
        except KeyError:
            return (
                "Could not find a molecule matching the text."
                "One possible cause is that the input is incorrect, "
                "input one molecule at a time."
            )
        # remove salts
        return Chem.CanonSmiles(self.largest_mol(smi))

    def smiles2name(self, smi: str) -> str:
        try:
            smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
        except Exception:
            return "Invalid SMILES string"
        # query the PubChem database
        r = requests.get(
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
            + smi
            + "/synonyms/JSON"
        )
        data = r.json()
        try:
            name = data["InformationList"]["Information"][0]["Synonym"][0]
        except KeyError:
            return "Unknown Molecule"
        return name

    def small_molecule_pdb(self, mol_str: str, path_registry) -> str:
        # takes in molecule name or smiles (converts to smiles if name)
        # writes pdb file name.pdb (gets name from smiles if possible)
        # output is done message
        ps = Chem.SmilesParserParams()
        ps.removeHs = False
        try:
            if self.is_smiles(mol_str):
                m = Chem.MolFromSmiles(mol_str)
                mol_name = self.smiles2name(mol_str)
            else:  # if input is not smiles, try getting smiles
                smi = self.molname2smiles(mol_str)
                m = Chem.MolFromSmiles(smi)
                mol_name = mol_str
            try:  # only if needed
                m = Chem.AddHs(m)
            except Exception:
                pass
            Chem.AllChem.EmbedMolecule(m)
            file_name = f"{mol_name}.pdb"
            Chem.MolToPDBFile(m, file_name)
            # add to path registry
            if path_registry:
                _ = path_registry.map_path(
                    file_name, file_name, f"pdb file for the small molecule {mol_name}"
                )
            return (
                f"PDB file for {mol_str} successfully created and saved to {file_name}."
            )
        except Exception:
            return (
                "There was an error getting pdb. Please input a single molecule name."
            )


class SmallMolPDB(BaseTool):
    name = "SmallMoleculePDB"
    description = (
        "Creates a PDB file for a small molecule"
        "Use this tool when you need to use a small molecule in a simulation."
        "Input can be a molecule name or a SMILES string."
    )
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, mol_str: str) -> str:
        """use the tool."""
        mol_pdb = MolPDB()
        output = mol_pdb.small_molecule_pdb(mol_str, self.path_registry)
        return output
