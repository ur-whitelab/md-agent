import os
from typing import Optional

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from langchain.tools import BaseTool

from mdagent.utils import FileType, PathRegistry

 # Load trajectory using MDTraj
        traj = md.load("trajectory.dcd", top="topology.pdb") # or
        traj = md.load(traj_file, top= top_file)

class SaltBridgeFunction:
    def __init__(self, path_registry):
        self.path_registry = path_registry
        self.includes_top = [".h5", ".lh5", ".pdb"]
        self.paired_salt_bridges=[] #stores paired salt bridges
        self.unpaired_residues=set() #store unpaired residues

    def find_salt_bridges(traj_file, top_file, threshold_distance=0.4, residue_pairs=None):
        salt_bridges = []

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
    unpaired_residues.discard(donor_idx)  # Remove donor from unpaired residues set
    unpaired_residues.discard(acceptor_idx)  # Remove acceptor from unpaired residues set
else:
    # If not, add them to the unpaired set
    unpaired_residues.add(donor_idx)  # Add donor to unpaired residues set
    unpaired_residues.add(acceptor_idx)  # Add acceptor to unpaired residues set


        return salt_bridges, list(unpaired_residues), list(residue_pairs)


 # Perform salt bridge analysis
        salt_bridges = find_salt_bridges(traj)

class SaltBridgeTool(BaseTool): #why cant I expand or condense this class like other class?
        name = "salt_bridge_tool"
        description = "A tool to find salt bridge in a protein trajectory"

        def __init_(self, path_registry):
            self.salt_bridge_function = SaltBridgeFunction(path_registry)

        def _run(self, traj_file, top_file, threshold_distance=0.4, residue_pairs=None):
            #i need to make sure this tool will find lone pairs too
            salt_bridges = [self.salt_bridge_function.find_salt_bridges(traj_file, top_file,
        threshold_distance, residue_pairs)]
            return salt_bridges

        def _agg_result(self, result):
            return result

        def _call__(self, traj_file, top_file, threshold_distance=0.4, residue_pairs=None):
            result = self._run(traj_file, top_file, threshold_distance, residue_pairs)
            return self._agg_result(result)


  # Load trajectory using MDTraj
        traj = md.load("trajectory.dcd", top="topology.pdb") # or
        traj = md.load(traj_file, top= top_file)

#create an instance (files?) of the salt bridge tool

path_registry = PathRegistry()
salt_bridge_tool = SaltBridgeTool(path_registry)

#to use tool to find salt bridges

salt_bridges = salt_bridge_tool(traj, "topology.pdb")

# Print identified salt bridges
print("Salt bridges found:")
for bridge in salt_bridges:
    print(
        f"Residue {traj.topology.atom(bridge[0]).residue.index + 1} ({traj.topology.atom(bridge[0]).residue.name}) - "
        f"Residue {traj.topology.atom(bridge[1]).residue.index + 1} ({traj.topology.atom(bridge[1]).residue.name})"
    )

    #Print unpaired residues
print("Unpaired_residues:")
salt_bridge_function = salt_bridge_tool.salt_bridge_function
for residue_idx in salt_bridge_function.unpaired_residues:
    print(f"Residue {traj.topology.atom(residue_idx).residue.index + 1} ({traj.topology.atom(residue_idx).residue.name})")
