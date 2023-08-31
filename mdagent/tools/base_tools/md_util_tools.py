from typing import Optional

import requests
from langchain.tools import BaseTool

from .registry import PathRegistry


def get_pdb(query_string, PathRegistry):
    """
    Search RSCB's protein data bank using the given query string
    and return the path to pdb file
    """

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
        return "No Content Error: PDB ID not found for this substance."
    elif "result_set" in r.json() and len(r.json()["result_set"]) > 0:
        pdbid = r.json()["result_set"][0]["identifier"]
        url = f"https://files.rcsb.org/download/{pdbid}.cif"
        pdb = requests.get(url)
        filename = f"{pdbid}.cif"
        with open(filename, "w") as file:
            file.write(pdb.text)
        # add filename to registry
        file_description = "PDB file downloaded from RSCB"
        PathRegistry.map_path(filename, filename, file_description)
        return filename


class Name2PDBTool(BaseTool):
    name = "PDBFileDownloader"
    description = """This tool downloads PDB (Protein Data Bank) or
                    CIF (Crystallographic Information File)
                    files using commercial chemical names.
                    It’s ideal for situations where you
                    need to directly retrieve these files
                    using a chemical’s commercial name.
                    Input: Commercial name of the chemical
                    Output: Corresponding PDB or CIF file"""

    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, query: str) -> str:
        """Use the tool."""
        try:
            if self.path_registry is None:  # this should not happen
                return "Path registry not initialized"
            pdb = get_pdb(query, self.path_registry)
            if pdb is None:
                return "Name2PDB tool failed to download the PDB file."
            else:
                return f"Name2PDB tool successfully downloaded the PDB file: {pdb}"
        except Exception as e:
            return f"Something went wrong. {e}"

    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Name2PDB does not support async")
