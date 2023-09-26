import os
from typing import Optional

import requests
from langchain.tools import BaseTool

from mdagent.utils import PathRegistry


def download_pdb_file(pdb_id, path_registry):
    """This tool downloads a PDB file from the RCSB Protein Data Bank (PDB) website.
    The function takes a PDB ID as input and checks if a JSON file containing the PDB
    IDs exists. If the JSON file does not exist, it prompts the user to enter a PDB ID.
    The function then uses the PDB ID to construct a URL and sends a GET request to
    download the PDB file. The downloaded file is saved with the same name as the PDB
    ID and returned as the output. Finally, the function prints the path to the
    downloaded file."""
    try:
        # Check if the JSON file exists
        if not os.path.exists("pdb_ids.json"):
            # Prompt the user to enter a pdb id
            pdb_id = input("Enter a pdb id: ")

        # Use the pdb id to download the mmcif or pdb file
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url)

        # Save the downloaded file
        file_path = f"{pdb_id}.pdb"
        with open(file_path, "wb") as file:
            file.write(response.content)

        # Return the path to the downloaded file
        return file_path
    except Exception as e:
        return str(e)


class DownloadPdbFile(BaseTool):
    name = "DownloadPdbFile"
    description = """This tool downloads a PDB file from the RCSB
    Protein Data Bank (PDB) website. The function takes a PDB ID as
    input and checks if a JSON file containing the PDB IDs exists. If
    the JSON file does not exist, it prompts the user to enter a PDB ID.
    The function then uses the PDB ID to construct a URL and sends a GET
    request to download the PDB file. The downloaded file is saved with
    the same name as the PDB ID and returned as the output.
    Finally, the function prints the path to the downloaded file."""
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, pdb_id: str) -> str:
        """Use the tool."""
        try:
            if self.path_registry is None:
                return "Path registry not initialized"
            output = download_pdb_file(pdb_id, self.path_registry)
            if output is None:
                return "This tool fails to produce the expected output"
            else:
                return f"This tool completed its task. Output: {output}"
        except Exception as e:
            return "Something went wrong: " + str(e)

    async def _async(self, pdb_id) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support async")
