from typing import Optional, Type

import MDAnalysis as mda
import MDAnalysis.analysis.distances as mda_dist
import numpy as np
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from mdagent.utils import PathRegistry, validate_tool_args


def ppi_distance(file_path, binding_site="protein"):
    """
    Calculates minimum heavy-atom distance between peptide (assumed to be
    smallest chain) and protein. Returns average distance between these two.

    Can specify binding site if given (optional)
    Can work with any protein-protein interaction (PPI)
    """
    # load and find smallest chain
    u = mda.Universe(file_path)
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


class PPIDistance(BaseTool):
    name: str = "ppi_distance"
    description: str = """Useful for calculating minimum heavy-atom distance
    between peptide and protein. First, make sure you have valid PDB file with
    any protein-protein interaction. Give this tool the name of the file. The
    tool will find the path."""
    args_schema: Type[BaseModel] = PPIDistanceInputSchema
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    @validate_tool_args(args_schema=args_schema)
    def _run(self, pdb_file: str, binding_site: str = "protein"):
        if not self.path_registry:
            return "Failed. Error: Path registry is not set"
        file_path = self.path_registry.get_mapped_path(pdb_file)
        if not file_path:
            return f"Failed. File not found: {pdb_file}"
        if not file_path.endswith(".pdb"):
            return "Failed. Error with input: PDB file must have .pdb extension"
        try:
            avg_dist = ppi_distance(file_path, binding_site=binding_site)
        except ValueError as e:
            return (
                f"Failed. ValueError: {e}. \nMake sure to provide valid PBD "
                "file and binding site using MDAnalysis selection syntax."
            )
        except Exception as e:
            return f"Failed. Something went wrong. {type(e).__name__}: {e}"
        return f"Succeeded: PPI average distance is {avg_dist}\n"

    def _arun(self, pdb_file: str, binding_site: str = "protein"):
        raise NotImplementedError("This tool does not support async")
