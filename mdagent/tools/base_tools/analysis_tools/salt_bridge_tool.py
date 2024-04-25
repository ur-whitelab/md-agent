import os
from typing import Optional

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from langchain.tools import BaseTool

from mdagent.utils import FileType, PathRegistry

 
class SaltBridgeFunction: #this class defines a method called find_salt_bridge
    #using MD traj and top files and threshold distance default, residue pair list
    # used to account for salt bridge analysis
    def __init__(self, path_registry):
        self.path_registry = path_registry
        self.includes_top = [".h5", ".lh5", ".pdb"] # those are the files we need for this analysis
        self.paired_salt_bridges=[] #stores paired salt bridges
        self.unpaired_residues=set() #store unpaired residues

    def find_salt_bridges(self, traj_file, top_file , threshold_distance=0.4, residue_pairs=None):
        salt_bridges = []
        # load trajectory using MDTraj
        traj = md.load(traj_file, top= top_file)
        if residue_pairs is None:
            residue_pairs = [("ARG", "ASP"), ("ARG", "GLU"), ("LYS", "ASP"), ("LYS", "GLU")]

        for pair in residue_pairs:
            donor_residues = traj.topology.select(f'residue_name == "{pair[0]}"')
            acceptor_residues = traj.topology.select(f'residue_name == "{pair[1]}"')

            for donor_idx in donor_residues:
                for acceptor_idx in acceptor_residues:
                    distances = md.compute_distances(traj, [[donor_idx, acceptor_idx]])
                    if any(d <= threshold_distance for d in distances):
                        salt_bridges.append((donor_idx, acceptor_idx))

 # Check if the donor and acceptor form a salt bridge
                        if any(d <= threshold_distance for d in distances):
                            # If yes, remove them from the unpaired set
                            self.unpaired_residues.discard(donor_idx)  # Remove donor from unpaired residues set
                            self.unpaired_residues.discard(acceptor_idx)  # Remove acceptor from unpaired residues set
                        else:
                            # If not, add them to the unpaired set
                            self.unpaired_residues.add(donor_idx)  # Add donor to unpaired residues set
                            self.unpaired_residues.add(acceptor_idx)  # Add acceptor to unpaired residues set
        print("Salt bridges found:")
        for bridge in salt_bridges:
            print(
                f"Residue {traj.topology.atom(bridge[0]).residue.index + 1} ({traj.topology.atom(bridge[0]).residue.name}) - "
                f"Residue {traj.topology.atom(bridge[1]).residue.index + 1} ({traj.topology.atom(bridge[1]).residue.name})"
            )

            #Print unpaired residues
        print("Unpaired_residues:")
    
        for residue_idx in self.unpaired_residues:
            print(f"Residue {traj.topology.atom(residue_idx).residue.index + 1} ({traj.topology.atom(residue_idx).residue.name})")

        return salt_bridges, list(self.unpaired_residues), list(residue_pairs)



class SaltBridgeTool(BaseTool):
    name = "salt_bridge_tool"
    description = "A tool to find salt bridge in a protein trajectory"
    path_registry: Optional[PathRegistry]

    
    def __init__(self, path_registry:Optional[PathRegistry]):
        super().___init__()
        self.path_registry = path_registry


         # This line is not correct self.salt_bridge_function = SaltBridgeFunction(path_registry)

    def _run(self, traj_file, top_file, threshold_distance=0.4, residue_pairs=None):
         # Load trajectory using MDTraj
        traj = md.load(traj_file, top=top_file)
        #calls the salt bridge function
        salt_bridges = [self.salt_bridge_function.find_salt_bridges(traj_file, top_file,
        threshold_distance, residue_pairs)]
        return salt_bridges

    def _agg_result(self, result):
        return result

    def _call__(self, traj_file, top_file, threshold_distance=0.4, residue_pairs=None):
        result = self._run(traj_file, top_file, threshold_distance, residue_pairs)
        return self._agg_result(result)




