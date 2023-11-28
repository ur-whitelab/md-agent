import ast
import json
import os
from typing import Optional

import langchain
from langchain.base_language import BaseLanguageModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from openmm import (
    AndersenThermostat,
    LangevinIntegrator,
    MonteCarloBarostat,
    VerletIntegrator,
    app,
)
from openmm.app import (
    ForceField,
    Modeller,
    PDBFile,
    PDBReporter,
    PDBxFile,
    Simulation,
    StateDataReporter,
)
from openmm.unit import bar, femtoseconds, kelvin, nanometers, picosecond, picoseconds

from mdagent.utils import PathRegistry

from ..preprocess_tools.clean_tools import CleaningTools


class SimulationFunctions:
    llm = langchain.chat_models.ChatOpenAI(
        temperature=0.05, model_name="gpt-4", request_timeout=1000, max_tokens=2000
    )

    def _prompt_summary(self, query: str, llm: BaseLanguageModel = llm):
        prompt_template = """Your input is the original query. Your
                            task is to parse through the user query.
                            and provide a summary of the file path input,
                            the type of preprocessing needed (this is the
                            same as cleaning the file), the forcefield
                            used for the simulation,
                            the ensemble of the simulation, the integrator needed,
                            the number of steps, the timestep, the temperature,
                            and other instructions.
                            and follow the format "name: description.

                            File Path: what is the file path of the file
                            you are using? it must include a .cif or .pdb extension.
                            Preprocessing: what preprocessing is needed?
                            you can choose from the following: standard cleaning,
                            remove water, add hydrogens, add hydrogens and remove
                            water. The default is add hydrogens and remove water.
                            Forcefield: what forcefields are you using?
                            you can choose from the following: AMBER, CHARMM,
                            OPLS, GROMACS. Default -->  "amber14-all.xml, tip3p.xml".
                            Ensemble: what ensemble are you using?
                            you can choose from the following:
                            NPT, NVT, NVE. Default --> "NVT".
                            Integrator: what integrator are you using?
                            you can choose from the following:
                            Langevin, Verlet, Brownian.
                            The default depends on the ensemble
                            (NPT -> Langevin, NVT -> Langevin, NVE -> Verlet).
                            Number of Steps: how many steps
                            are you using? The default is 10000.
                            Timestep: what is the timestep?
                            Default --> "1 fs".
                            Temperature: what is the temperature?
                            Default --> "300 K".
                            Pressure: What is the pressure?
                            If NPT ensemble, the default is 1.0 bar, otherwise None.
                            Friction: what is the friction coefficient?
                            Default --> "1.0"
                            record_params: what parameters do you want to record?
                            you can choose from the following:
                            step, time, potentialEnergy, kineticEnergy,
                            totalEnergy, temperature, volume, density,
                            progress, remainingTime, speed, elapsedTime,
                            separator, systemMass, totalSteps, append.
                            Default --> ["step", "potentialEnergy", "temperature"].
                            Other Instructions: what other instructions do you have?
                            The default is none.
                            Example of the final output:
                            File Path: 1a1p.pdb
                            Preprocessing: standard cleaning
                            Forcefield: amber14-all.xml, tip3p.xml
                            Ensemble: NPT
                            Integrator: Langevin
                            Number of Steps: 10000
                            Timestep: 1 fs
                            Temperature: 300 K
                            Pressure: 1.0 bar
                            Friction: 1.0
                            record_params: ["step", "potentialEnergy", "temperature"]
                            Other Instructions: none
                            If there is not enough information in a category,
                            you may fill in with the default, but explicitly state so.
                            Here is the information:{query}"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["query"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        return llm_chain.run(" ".join(query))

    def _save_to_file(self, summary: str, filename: str, PathRegistry):
        """Parse the summary string and
        save it to a file in JSON format."""
        # Split the summary into lines
        lines = summary.strip().split("\n")

        # Parse each line into a key and a value
        summary_dict = {}
        for line in lines:
            key, value = line.split(":")
            summary_dict[key.strip()] = value.strip()

        # Save the dictionary to a file
        with open(filename, "w") as f:
            json.dump(summary_dict, f)

        # add filename to registry
        file_description = "Simulation Parameters"
        PathRegistry.map_path(filename, filename, file_description)

    def _instruction_summary(self, query: str, PathRegistry):
        summary = self._prompt_summary(query)
        self._save_to_file(summary, "simulation_parameters.json", PathRegistry)
        return summary

    def _setup_simulation_from_json(self, file_name):
        # Open the json file and load the parameters
        with open(file_name, "r") as f:
            params = json.load(f)
        return params

    def _setup_and_run_simulation(self, query, PathRegistry):
        # Load the force field
        # ask for inputs from the user
        params = self._setup_simulation_from_json(query)

        # forcefield key can be forcefield_files or Forcefield
        if "forcefield_files" in params:
            params["forcefield_files"] = (
                params["forcefield_files"]
                .replace("(default)", "")
                .replace(" and ", ",")
                .strip()
            )
            Forcefield_files = [
                file.strip() for file in params["forcefield_files"].split(",")
            ]
            Forcefield = Forcefield_files[0]
            Water_model = Forcefield_files[1]
        else:
            params["Forcefield"] = (
                params["Forcefield"]
                .replace("(default)", "")
                .replace(" and ", ",")
                .strip()
            )
            Forcefield_files = [
                file.strip() for file in params["Forcefield"].split(",")
            ]
            Forcefield = Forcefield_files[0]
            Water_model = Forcefield_files[1]
        print("Setting up forcields :", Forcefield, Water_model)
        # check if forcefields end in .xml
        if Forcefield.endswith(".xml") and Water_model.endswith(".xml"):
            forcefield = ForceField(Forcefield, Water_model)
        # adding forcefield to registry

        # Load the PDB file
        cleantools = CleaningTools()
        pdbfile = cleantools._extract_path(params["File Path"])
        print("Starting pdb/cis file :", pdbfile)
        name = pdbfile.split(".")[0]
        end = pdbfile.split(".")[1]
        if end == "pdb":
            pdb = PDBFile(pdbfile)
        elif end == "cif":
            pdb = PDBxFile(pdbfile)

        modeller = Modeller(pdb.topology, pdb.positions)
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * nanometers,
            constraints=app.PME,
        )

        _integrator = params["Integrator"].split(" ")[0].strip()
        _temp = params["Temperature"].split(" ")[0].strip()
        _friction_coef = params["Friction"].split(" ")[0].strip()
        _timestep = params["Timestep"].split(" ")[0].strip()

        if _integrator == "Langevin":
            print(
                "Setting up Langevin integrator with Parameters:",
                _temp,
                "K",
                _friction_coef,
                "1/ps",
                _timestep,
                "fs",
            )
            if params["Ensemble"] == "NPT":
                _pressure = params["Pressure"].split(" ")[0].strip()
                system.addForce(MonteCarloBarostat(_pressure * bar, _temp * kelvin))
            integrator = LangevinIntegrator(
                float(_temp) * kelvin,
                float(_friction_coef) / picosecond,
                float(_timestep) * femtoseconds,
            )
        elif _integrator == "Verlet":
            if params["Ensemble"] == "NPT":
                _pressure = params["Pressure"].split(" ")[0].strip()
                system.addForce(AndersenThermostat(_temp * kelvin, 1 / picosecond))
                system.addForce(MonteCarloBarostat(_pressure * bar, _temp * kelvin))
                print(
                    "Setting up Verlet integrator with Parameters:",
                    _timestep,
                    "fs",
                    _temp,
                    "K",
                    _pressure,
                    "bar",
                )
            print("Setting up Verlet integrator with Parameters:", _timestep, "fs")
            integrator = VerletIntegrator(float(_timestep) * picoseconds)

        simulation = Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)
        simulation.minimizeEnergy()
        simulation.reporters.append(PDBReporter(f"{name}.pdb", 1000))
        # reporter_args = {"reportInterval": 1000}
        reporter_args = {}
        params["record_params"] = ast.literal_eval(params["record_params"])
        for param in params["record_params"]:
            if param in [
                "step",
                "time",
                "potentialEnergy",
                "kineticEnergy",
                "totalEnergy",
                "temperature",
                "volume",
                "density",
                "progress",
                "remainingTime",
                "speed",
                "elapsedTime",
                "separator",
                "systemMass",
                "totalSteps",
                "append",
            ]:
                # The params from the json file should be booleans
                reporter_args[param] = True
        simulation.reporters.append(
            StateDataReporter(f"{name}.csv", 1000, **reporter_args)
        )

        simulation.step(int(params["Number of Steps"].split(" ")[0].strip()))

        # add filenames to registry
        file_name1 = "simulation_trajectory.pdb"
        file_description1 = "Simulation PDB, containing the simulation trajectory"
        PathRegistry.map_path(file_name1, f"{name}.pdb", file_description1)
        file_name2 = "simulation_data.csv"
        file_description2 = (
            "Simulation Data, containing step, potential energy, and temperature"
        )
        PathRegistry.map_path(file_name2, f"{name}.csv", file_description2)

        return simulation

    def _extract_parameters_path(self):
        """Check directory for parameters.json file."""
        # Check if there is a parameters.json file in the directory.
        if os.path.exists("simulation_parameters_summary.json"):
            return "simulation_parameters_summary.json"
        # If there's no exact match, check for
        # any JSON file that contains 'parameters' in its name.
        else:
            for file in os.listdir("."):
                if "parameters" in file and file.endswith(".json"):
                    return file
            # If no matching file is found, raise an exception.
            raise ValueError("No parameters.json file found in directory.")


class SetUpAndRunTool(BaseTool):
    name = "SetUpAndRunTool"
    description = """This tool can only run after InstructionSummary
                    This tool will set up the simulation objects
                    and run the simulation.
                    It will ask for the parameters path.
                    input:  json file
                    """
    path_registry: Optional[PathRegistry]

    def __init__(
        self,
        path_registry: Optional[PathRegistry],
    ):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, query: str) -> str:
        """Use the tool"""
        # find the parameters in the directory
        try:
            if self.path_registry is None:  # this should not happen
                return "Registry not initialized"
            sim_fxns = SimulationFunctions()
            parameters = sim_fxns._extract_parameters_path()

        except ValueError as e:
            return (
                str(e)
                + """\nPlease use the Instruction summary tool with the
                to create a parameters.json file in the directory."""
            )
        self.log("This are the parameters:")
        self.log(parameters)
        # print the parameters in json file
        with open(parameters) as f:
            params = json.load(f)
        for key, value in params.items():
            print(key, ":", value)
        self.log("Are you sure you want to run the simulation? (y/n)")
        response = input("yes or no: ")
        if response.lower() in ["yes", "y"]:
            sim_fxns._setup_and_run_simulation(parameters, self.path_registry)
        else:
            return "Simulation interrupted due to human input"
        return "Simulation Completed, simulation trajectory and data files saved."

    def log(self, text, color="blue"):
        if color == "blue":
            print("\033[1;34m\t{}\033[00m".format(text))
        if color == "red":
            print("\033[31m\t{}\033[00m".format(text))

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class InstructionSummary(BaseTool):
    name = "Instruction Summary"
    description = """This tool will summarize the instructions
     given by the human. This is the first tool you will
       use, unless you dont have a .cif or .pdb file in
       which case you have to download one first.
     Input: Instructions or original query.
     Output: Summary of instructions"""
    path_registry: Optional[PathRegistry]

    def __init__(
        self,
        path_registry: Optional[PathRegistry],
    ):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, query: str) -> str:
        # first check if there is any .cif or .pdb files in the directory
        # if there is, then ask for instructions
        if self.path_registry is None:  # this should not happen
            return "Registry not initialized"
        files = os.listdir(".")
        pdb_cif_files = [f for f in files if f.endswith(".pdb") or f.endswith(".cif")]
        pdb_cif_files_tidy = [
            f
            for f in files
            if (f.endswith(".pdb") or f.endswith(".cif")) and "tidy" in f
        ]
        if len(pdb_cif_files_tidy) != 0:
            path = pdb_cif_files_tidy[0]
        else:
            path = pdb_cif_files[0]
            sim_fxns = SimulationFunctions()
            summary = sim_fxns._prompt_summary(query + "the pdbfile is" + path)
            sim_fxns._save_to_file(
                summary, "simulation_parameters_summary.json", self.path_registry
            )
        return summary

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
