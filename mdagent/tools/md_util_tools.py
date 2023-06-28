from typing import Optional

import requests
from langchain.tools import BaseTool

from mdagent.utils import PathRegistry


def get_pdb(query_string, PathRegistry):
    """
    Search RSCB's protein data bank using the given query string
    and return the path to pdb file in either CIF or PDB format
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
    if r.status_code != 204: # need to check this
        return None
    if "cif" in query_string or "CIF" in query_string:
        filetype = "cif"
    else:
        filetype = "pdb"

    if "result_set" in r.json() and len(r.json()["result_set"]) > 0:
        pdbid = r.json()["result_set"][0]["identifier"]
        print(f"PDB file found with this ID: {pdbid}")
        url = f"https://files.rcsb.org/download/{pdbid}.{filetype}"
        pdb = requests.get(url)
        filename = f"{pdbid}.{filetype}"
        with open(filename, "w") as file:
            file.write(pdb.text)
        print(f"{filename} is created.")
        file_description = f"PDB file downloaded from RSCB, PDB ID: {pdbid}"
        PathRegistry.map_path(filename, filename, file_description)
        return filename
    return None


class Name2PDBTool(BaseTool):
    name = "PDBFileDownloader"
    description = """This tool downloads PDB (Protein Data Bank) or
                    CIF (Crystallographic Information File) files using
                    commercial chemical names. It’s ideal for situations where
                    you need to directly retrieve these file using a chemical’s
                    commercial name. When a specific file type, either PDB or CIF,
                    is requested, add file type to the query string with space.
                    Input: Commercial name of the chemical or file without
                    file extension
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
                return "Name2PDB tool failed to find and download PDB file."
            else:
                return f"Name2PDB tool successfully downloaded the PDB file: {pdb}"
        except Exception as e:
            return f"Something went wrong. {e}"

    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("this tool does not support async")
