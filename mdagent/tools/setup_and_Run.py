import json
import os

from langchain.tools import BaseTool
from openmm import PME, HBonds, LangevinIntegrator, VerletIntegrator
from openmm.app import (
    ForceField,
    Modeller,
    PDBFile,
    PDBReporter,
    PDBxFile,
    Simulation,
    StateDataReporter,
)
from openmm.unit import femtoseconds, kelvin, nanometers, picosecond, picoseconds

from .clean_tools import _extract_path


def _setup_simulation_from_json(file_name):
    # Open the json file and load the parameters
    with open(file_name, "r") as f:
        params = json.load(f)
    return params


def _SetUpAndRunSimmulation(query):
    # Load the force field
    # ask for inputs from the user
    params = _setup_simulation_from_json(query)
    params["Forcefield"] = params["Forcefield"].replace("(default)", "")
    Forcefield = params["Forcefield"].split(",")[0]
    Water_model = params["Forcefield"].split(",")[1].strip()
    print(Forcefield, Water_model)
    # check if forcefields end in .xml
    if Forcefield.endswith(".xml") and Water_model.endswith(".xml"):
        print("yes")
        forcefield = ForceField(Forcefield, Water_model)

        # Load the PDB file
    pdbfile = _extract_path(params["File Path"])
    name = pdbfile.split(".")[0]
    end = pdbfile.split(".")[1]
    if end == "pdb":
        pdb = PDBFile(params["File Path"])
    elif end == "cif":
        pdb = PDBxFile(params["File Path"])

    modeller = Modeller(pdb.topology, pdb.positions)
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * nanometers,
        constraints=HBonds,
    )

    _integrator = params["Integrator"].split(" ")[0].strip()
    _temp = params["Temperature"].split(" ")[0].strip()
    _friction_coef = params["Friction"].split(" ")[0].strip()
    _timestep = params["Timestep"].split(" ")[0].strip()

    print(_integrator)
    if _integrator == "Langevin":
        integrator = LangevinIntegrator(
            float(_temp) * kelvin,
            float(_friction_coef) / picosecond,
            float(_timestep) * femtoseconds,
        )
    elif _integrator == "Verlet":
        integrator = VerletIntegrator(float(_timestep) * picoseconds)

    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()
    simulation.reporters.append(PDBReporter(f"{name}.pdb", 1000))
    simulation.reporters.append(
        StateDataReporter(
            f"{name}.csv", 1000, step=True, potentialEnergy=True, temperature=True
        )
    )
    simulation.step(int(params["Number of Steps"].split(" ")[0].strip()))
    return simulation


def _extract_parameters_path():
    """Check directory for parameters.json file."""
    # Check if there is a parameters.json file in the directory.
    if os.path.exists("parameters.json"):
        return "parameters.json"
    # If there's no exact match, check for
    # any JSON file that contains 'parameters' in its name.
    else:
        for file in os.listdir("."):
            if "parameters" in file and file.endswith(".json"):
                return file
        # If no matching file is found, raise an exception.
        raise ValueError("No parameters.json file found in directory.")


class SetUpAndRunTool(BaseTool):
    name = "Set Up simmulation objects and run simmulation"
    description = """This tool will set up the simmulation objects
                    and run the simmulation.
                    It will ask for the parameters path.
                    """

    def _run(self, query: str) -> str:
        # find the parameters in the directory
        parameters = _extract_parameters_path()
        _SetUpAndRunSimmulation(parameters)
        return "Simmulation Complete"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
