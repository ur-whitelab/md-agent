from langchain.tools import BaseTool


def get_pdb(query_string):
    """
    Search RSCB's protein data bank using the given query string
    and return the path to temp pdb file
    """
    import requests

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
    if "result_set" in r.json() and len(r.json()["result_set"]) > 0:
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
    name = "name2pdb"
    description = "useful for when you need to retrieve PDB file using chemical names"
    # change to mention pdb name must be given

    def _run(self, query: str) -> str:
        """Use the tool."""
        return get_pdb(query)

    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Name2PDB does not support async")
