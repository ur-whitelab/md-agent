import requests
from langchain.tools import BaseTool


def get_pdb(query_string):
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
        print("No Content Error: PDB ID not found for this substance.")
        return None
    elif "result_set" in r.json() and len(r.json()["result_set"]) > 0:
        pdbid = r.json()["result_set"][0]["identifier"]
        print(f"PDB file found for fibronectin: {pdbid}")
        url = f"https://files.rcsb.org/download/{pdbid}.cif"
        pdb = requests.get(url)
        filename = f"{pdbid}.cif"
        with open(filename, "w") as file:
            file.write(pdb.text)
        print(f"{filename} is created")
        return filename
    return None


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

    def _run(self, query: str) -> str:
        """Use the tool."""
        get_pdb(query)
        return "PDB file was downlaoded, please check the directory."

    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Name2PDB does not support async")
