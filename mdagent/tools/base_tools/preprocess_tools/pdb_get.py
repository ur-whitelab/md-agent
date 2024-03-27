import os
from typing import Optional

import requests
import streamlit as st
from langchain.tools import BaseTool
from rdkit import Chem

from mdagent.utils import FileType, PathRegistry


def get_pdb(query_string: str, path_registry: PathRegistry):
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
        directory = f"{path_registry.ckpt_pdb}"
        # Create the directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(f"{directory}/{filename}", "w") as file:
            file.write(pdb.text)
        path_registry.map_path(
            file_id,
            f"{path_registry.ckpt_pdb}/{filename}",
            "PDB file downloaded from RSCB",
        )

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
                    f"{self.path_registry.ckpt_pdb}/{filename}",
                    f"PDB file downloaded from RSCB, PDBFile ID: {pdbfile_id}",
                )
                return f"Name2PDB tool successful. downloaded the PDB file:{pdbfile_id}"
        except Exception as e:
            return f"Something went wrong. {e}"

    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("this tool does not support async")


class MolPDB:
    def __init__(self, path_registry):
        self.path_registry = path_registry

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

    def small_molecule_pdb(self, mol_str: str) -> str:
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
            file_name = f"{self.path_registry.ckpt_pdb}/{mol_name}.pdb"
            Chem.MolToPDBFile(m, file_name)
            print("finished writing pdb file")
            self.path_registry.map_path(
                mol_name, file_name, f"pdb file for the small molecule {mol_name}"
            )
            return (
                f"PDB file for {mol_str} "
                "successfully created and saved "
                f"to {mol_name}.pdb."
            )
        except Exception as e:
            print(
                "There was an error getting pdb. Please input a single molecule name."
                f"{mol_str},{mol_name}"
            )
            return (
                "There was an error getting pdb. Please input a single molecule name."
                "Error: " + str(e)
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
        mol_pdb = MolPDB(self.path_registry)
        output = mol_pdb.small_molecule_pdb(mol_str)
        return output
