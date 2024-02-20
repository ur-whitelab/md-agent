from typing import List, Optional

import matplotlib.pyplot as plt
import mdtraj as md
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from mdagent.utils import PathRegistry


class RDFToolInput(BaseModel):
    trajectory_fileid: str = Field(
        None, description="Trajectory file. Either dcd, hdf5, xtc oe xyz"
    )

    topology_fileid: Optional[str] = Field(None, description="Topology file")
    stride: Optional[int] = Field(None, description="Stride for reading trajectory")
    selections: Optional[List[List[str]]] = Field(
        [["protein", "water"]],
        description="Selections for RDF. Do not use for now. As "
        "it will only calculate RDF for protein and water molecules.",
    )
    # atom_indices: Optional[List[int]] = Field(
    #    None, description="Atom indices to load in the trajectory"
    # )
    # TODO: Add pairs of atoms to calculate RDF within the tool
    ##pairs: Optional[str] = Field(None, description="Pairs of atoms to calculate RDF ")


class RDFutils:
    # get the expression for select pairs
    pass


class RDFTool(BaseTool):
    name = "RDFTool"
    description = (
        "Calculate the radial distribution function (RDF) of a trajectory "
        "of a protein with respect to water molecules. \n\nInput Example 1: \n"
        "trajectory_fileid: 'rec0_142404' \n"
        "topology_fileid: 'top_sim0_142401' \n"
        "stride: 2 \n"
        "selections: None\n"
        "Input Example 2: \n"
        "trajectory_fileid: 'rec0_142404' \n"
        "topology_fileid: 'top_sim0_142401' \n"
        "\n\n"
        "As you can see, the stride and selections are optional. "
    )
    args_schema = RDFToolInput
    path_registry: Optional[PathRegistry]

    def _run(self, input):
        try:
            inputs = self.validate_input(input)
        except ValueError as e:
            if "Incorrect Inputs" in str(e):
                print("Error in Inputs in RDF tool: ", str(e))
                return ("Error in Inputs", str(e))
            elif "Invalid file extension" in str(e):
                print("File Extension Not Supported in RDF tool: ", str(e))
                return ("File Extension Not Supported", str(e))
            elif "Missing Inputs" in str(e):
                print("Missing Inputs in RDF tool: ", str(e))
                return ("Missing Inputs", str(e))
            else:
                raise ValueError(f"Error during inputs in RDF tool {e}")

        trajectory_id = inputs["trajectory_fileid"]
        topology_id = inputs["topology_fileid"]
        stride = inputs["stride"]
        inputs["selections"]  # not used at the moment

        path_to_traj = self.path_registry.get_mapped_path(trajectory_id)
        ending = path_to_traj.split(".")[-1]
        if ending in ["dcd", "xtc", "xyz"]:
            path_to_top = self.path_registry.get_mapped_path(topology_id)
            traj = md.load(path_to_traj, top=path_to_top, stride=stride)
        else:
            # hdf5, h5, pdb already checked in validation of inputs
            traj = md.load(path_to_traj, stride=stride)
        try:
            r, gr = md.compute_rdf(
                traj,
                traj.topology.select_pairs(
                    ("protein and backbone and " "(name C or name N or name CA)"),
                    "water and name O",
                ),
                r_range=(0.1, 2),  # Adjust these values based on your system
                bin_width=0.005,
            )
        except Exception as e:
            # not sure what exceptions to catch for now, will handle them as they come
            print("Error in RDF calculation:", str(e))
            raise ("Error in RDF calculation: ", str(e))
        # save plot
        fig, ax = plt.subplots()
        ax.plot(r, gr)
        ax.set_xlabel(r"$r$ (nm)")
        ax.set_ylabel(r"$g(r)$")
        ax.set_title("RDF")
        num = 0
        image_name = "rdf{}_{}.png".format(num, trajectory_id)
        while image_name in self.path_registry.list_path_names():
            num += 1
            image_name = "rdf_{}_{}.png".format(trajectory_id, num)

        plt.savefig(image_name)
        plt.close()
        return (
            "RDF calculated successfully"
            f"{image_name} has been saved in the current directory"
        )
        # path_to_top = self.path_registry.get_mapped_path(topology_id)

    def _arun(self, input):
        pass

    def validate_input(self, input):
        if "action_input" in input:
            input = input["action_input"]

        trajectory_id = input.get("trajectory_fileid", None)

        topology_id = input.get("topology_fileid", None)

        stride = input.get("stride", None)

        selections = input.get("selections", [])

        if not trajectory_id:
            raise ValueError("Missing Inputs: Trajectory file ID is required")

        # check if trajectory id is valid
        fileids = self.path_registry.list_path_names()
        print("fileids: ", fileids)
        if trajectory_id not in fileids:
            raise ValueError("Trajectory File ID not in path registry")

        path_to_traj = self.path_registry.get_mapped_path(trajectory_id)

        ending = path_to_traj.split(".")[-1]
        if ending in ["dcd", "xtc", "xyz"]:
            # requires topology
            if not topology_id:
                raise ValueError(
                    "Missing Inputs: "
                    "Topology file is required for trajectory "
                    "file with extension {}".format(ending)
                )
            if topology_id not in fileids:
                raise ValueError("Topology File ID not in path registry")

        elif ending in ["hdf5", "h5", "pdb"]:
            # does not require topology
            pass

        else:
            raise ValueError(
                "Invalid file extension for trajectory file. "
                "For the moment only supported extensions are: "
                "dcd, xtc, hdf5, h5, xyz, pdb"
            )

        if stride:
            if type(stride) != int:
                try:
                    stride = int(stride)
                    if stride <= 0:
                        raise ValueError(
                            "Incorrect Inputs: "
                            "Stride must be a positive integer "
                            "or None for default value of 1"
                        )
                except ValueError:
                    raise ValueError(
                        "Incorrect Inputs: Stride must be an integer "
                        "or None for default value of 1"
                    )
            else:
                if stride <= 0:
                    raise ValueError(
                        "Incorrect Inputs: " "Stride must be a positive integer"
                    )

        if selections:
            try:
                selections = list(
                    map(str, selection.split(",")) for selection in selections
                )
            except ValueError:
                raise ValueError(
                    "Incorrect Inputs: Selections must be a list of comma "
                    "separated lists of  or None for all atoms"
                )
        inputs = {
            "trajectory_fileid": trajectory_id,
            "topology_fileid": topology_id,
            "stride": stride,
            "selections": selections,
        }

        return inputs
