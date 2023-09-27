import os

import requests


def download_pdb_file(pdb_id):
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


pdb_id = "3pqr"
downloaded_file = download_pdb_file(pdb_id)
print(f"Downloaded file: {downloaded_file}")
