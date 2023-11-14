from typing import Type

import MDAnalysis as mda
import MDAnalysis.analysis.distances as mda_dist
import numpy as np
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


def ppi_distance(pdb_file, binding_site="protein"):
    """
    Calculates minimum heavy-atom distance between peptide (assumed to be
    smallest chain) and protein. Returns average distance between these two.

    Can specify binding site if given (optional)
    Can work with any protein-protein interaction (PPI)
    """
    # load and find smallest chain
    u = mda.Universe(pdb_file)
    peptide = None
    for chain in u.segments:
        if peptide is None or len(chain.residues) < len(peptide):
            peptide = chain.residues
    protein = u.select_atoms(
        f"({binding_site}) and not segid {peptide.segids[0]} and not name H*"
    )
    peptide = peptide.atoms.select_atoms("not name H*")
    all_d = []
    for r in peptide.residues:
        distances = mda_dist.distance_array(r.atoms.positions, protein.positions)
        # get row, column of minimum distance
        i, j = np.unravel_index(distances.argmin(), distances.shape)
        all_d.append(distances[i, j])
    avg_dist = np.mean(all_d)
    return avg_dist


class PPIDistanceInputSchema(BaseModel):
    pdb_file: str = Field(
        description="name of PDB file containing protein-protein interaction"
    )
    binding_site: str = Field(
        description="a list of residues as the binding site of the protein"
    )


class PPIDistance(BaseTool):
    name: str = "ppi_distance"
    description: str = """Use to calculate minimum heavy-atom distance between
    peptide and protein. Can use any PDB file with protein-protein interaction."""
    arg_schema: Type[BaseModel] = PPIDistanceInputSchema

    def _run(self, pdb_file: str, binding_site: str = "protein"):
        avg_dist = ppi_distance(pdb_file, binding_site=binding_site)
        return avg_dist

    def _arun(self, pdb_file: str, binding_site: str = "protein"):
        raise NotImplementedError("This tool does not support async")
