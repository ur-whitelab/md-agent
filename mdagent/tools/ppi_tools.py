from typing import Any, Dict, Optional, Type

import MDAnalysis as mda
import MDAnalysis.analysis.distances as mda_dist
import numpy as np
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, root_validator


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
        description="file with .pdb extension containing protein-protein interaction"
    )
    binding_site: Optional[str] = Field(
        description="""a list of selected residues as the binding site
        of the protein using MDAnalysis selection syntax."""
    )

    @root_validator
    def validate_query(cls, values: Dict[str, Any]) -> Dict:
        pdb_file = values.get("pdb_file")
        if not pdb_file:
            values["error"] = "PDB file must be provided"
        elif not pdb_file.endswith(".pdb"):
            values["error"] = "PDB file must have .pdb extension"
        return values


class PPIDistance(BaseTool):
    name: str = "ppi_distance"
    description: str = """Useful for calculating minimum heavy-atom distance
    between peptide and protein. First, make sure you have valid PDB file with
    any protein-protein interaction."""
    args_schema: Type[BaseModel] = PPIDistanceInputSchema

    def _run(
        self, pdb_file: str, binding_site: str = "protein", error: Optional[str] = ""
    ):
        if error:  # this doesn't work
            return f"error: {error}"
        try:
            avg_dist = ppi_distance(pdb_file, binding_site=binding_site)
        except ValueError as e:
            return (
                f"ValueError: {e}. \nMake sure to provide valid PBD "
                "file and binding site using MDAnalysis selection syntax."
            )
        except Exception as e:
            return f"Something went wrong. {type(e).__name__}: {e}"
        return f"{avg_dist}\n"

    def _arun(self, pdb_file: str, binding_site: str = "protein"):
        raise NotImplementedError("This tool does not support async")
