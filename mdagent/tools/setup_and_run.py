import json
import os

import langchain
from langchain import LLMChain, PromptTemplate
from langchain.base_language import BaseLanguageModel
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

from .clean_tools import _extract_path


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
                            OPLS, GROMACS. The default is "amber14-all.xml, tip3p.xml".
                            Ensemble: what ensemble are you using?
                            you can choose from the following:
                            NPT, NVT, NVE. The default is NVT
                            Integrator: what integrator are you using?
                            you can choose from the following:
                            Langevin, Verlet, Brownian.
                            The default depends on the ensemble
                            (NPT -> Langevin, NVT -> Langevin, NVE -> Verlet).
                            Number of Steps: how many steps
                            are you using? The default is 1000.
                            Timestep: what is the timestep?
                            The default is 1 fs.
                            Temperature: what is the temperature?
                            The default is 300 K.
                            Pressure: What is the pressure?
                            If NPT ensemble, the default is 1.0 bar, otherwise None.
                            Friction: what is the friction coefficient?
                            The default is 1.0 (1/ps)
                            Other Instructions: what other instructions do you have?
                            The default is none.

                            If there is not enough information in a category,
                            you may fill in with the default, but explicitly state so.
                            Here is the information:{query}"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["query"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        return llm_chain.run(" ".join(query))

    def _save_to_file(self, summary: str, filename: str):
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

    def _instruction_summary(self, query: str):
        summary = self._prompt_summary(query)
        self._save_to_file(summary, "simulation_parameters.json")
        return summary

    def _setup_simulation_from_json(self, file_name):
        # Open the json file and load the parameters
        with open(file_name, "r") as f:
            params = json.load(f)
        return params

    def _setup_and_run_simulation(self, query):
        # Load the force field
        # ask for inputs from the user
        params = self._setup_simulation_from_json(query)
        params["Forcefield"] = params["Forcefield"].replace("(default)", "")
        Forcefield = params["Forcefield"].split(",")[0]
        Water_model = params["Forcefield"].split(",")[1].strip()
        print("Setting up forcields :", Forcefield, Water_model)
        # check if forcefields end in .xml
        if Forcefield.endswith(".xml") and Water_model.endswith(".xml"):
            forcefield = ForceField(Forcefield, Water_model)

            # Load the PDB file
        pdbfile = _extract_path(params["File Path"])
        print("Starting pdb/cis file :", pdbfile)
        name = pdbfile.split(".")[0]
        end = pdbfile.split(".")[1]
        if end == "pdb":
            pdb = PDBFile(params["File Path"])
        elif end == "cif":
            pdb = PDBxFile(params["File Path"])

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
        simulation.reporters.append(
            StateDataReporter(
                f"{name}.csv", 1000, step=True, potentialEnergy=True, temperature=True
            )
        )
        simulation.step(int(params["Number of Steps"].split(" ")[0].strip()))
        return simulation

    def _extract_parameters_path(self):
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
    name = "SetUpAndRunTool"
    description = """This tool will set up the simulation objects
                    and run the simulation.
                    It will ask for the parameters path.
                    input: parameters.json (if the .json)
                    """

    def _run(self, query: str) -> str:
        """Use the tool"""
        # find the parameters in the directory
        try:
            sim_fxns = SimulationFunctions()
            parameters = sim_fxns._extract_parameters_path()
        except ValueError as e:
            return (
                str(e)
                + f"""\nPlease use the Instruction summary tool with the
                query: {query} to create a parameters.json file in the directory."""
            )
        sim_fxns._setup_and_run_simulation(parameters)
        return "Simulation Completed, saved as .pdb and .csv files"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
