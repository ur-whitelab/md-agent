# Standard Library Imports
import ast
import json
import os
import re

# Third-Party Imports
import textwrap
from typing import Any, Dict, List, Optional, Type

import langchain
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from openmm import (
    AndersenThermostat,
    BrownianIntegrator,
    LangevinIntegrator,
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    OpenMMException,
    Platform,
    VerletIntegrator,
    app,
    unit,
)
from openmm.app import (
    PME,
    AllBonds,
    CutoffNonPeriodic,
    CutoffPeriodic,
    DCDReporter,
    Ewald,
    ForceField,
    HAngles,
    HBonds,
    Modeller,
    NoCutoff,
    PDBFile,
    PDBReporter,
    PDBxFile,
    Simulation,
    StateDataReporter,
)
from openmm.unit import bar, femtoseconds, kelvin, nanometers, picosecond, picoseconds
from pydantic import BaseModel, Field

from mdagent.tools.base_tools.preprocess_tools import CleaningTools

# Local Library/Application Imports
from mdagent.utils import FileType, PathRegistry

# TODO delete files created from the simulation if not needed.

FORCEFIELD_LIST = [
    "amber14/DNA.OL15.xml",
    "amber14/DNA.bsc1.xml",
    "amber14/RNA.OL3.xml",
    "amber14/lipid17.xml",
    "amber14/protein.ff14SB.xml",
    "amber14/protein.ff15ipq.xml",
    "amber14/spce.xml",
    "amber14/tip3p.xml",
    "amber14/tip3pfb.xml",
    "amber14/tip4pew.xml",
    "amber14/tip4pfb.xml",
    "charmm36/spce.xml",
    "charmm36/tip3p-pme-b.xml",
    "charmm36/tip3p-pme-f.xml",
    "charmm36/tip4p2005.xml",
    "charmm36/tip4pew.xml",
    "charmm36/tip5p.xml",
    "charmm36/tip5pew.xml",
    "charmm36/water.xml",
    "absinth.xml",
    "amber03.xml",
    "amber03_obc.xml",
    "amber10.xml",
    "amber10_obc.xml",
    "amber14-all",
    "amber14-all.xml",
    "amber96.xml",
    "amber96_obc.xml",
    "amber99Test.xml",
    "amber99_obc.xml",
    "amber99sb.xml",
    "amber99sbildn.xml",
    "amber99sbnmr.xml",
    "amberfb15.xml",
    "amoeba2009.xml",
    "amoeba2009_gk.xml",
    "amoeba2013.xml",
    "amoeba2013_gk.xml",
    "charmm36.xml",
    "charmm_polar_2013.xml",
    "hydrogens.xml",
    "iamoeba.xml",
    "pdbNames.xml",
    "residues.xml",
    "spce.xml",
    "swm4ndp.xml",
    "tip3p.xml",
    "tip3pfb.xml",
    "tip4pew.xml",
    "tip4pfb.xml",
    "tip5p.xml",
]


class SimulationFunctions:
    def __init__(
        self,
        path_registry,
        temperature: float = 0.05,
        model_name: str = "gpt-4",
        request_timeout: int = 1000,
        max_tokens: int = 2000,
    ):
        self.path_registry = path_registry
        self.temperature = temperature
        self.model_name = model_name
        self.request_timeout = request_timeout
        self.max_tokens = max_tokens

        self.llm = langchain.chat_models.ChatOpenAI(
            temperature=self.temperature,
            model_name=self.model_name,
            request_timeout=self.request_timeout,
            max_tokens=self.request_timeout,
        )

    #######==================System Congifuration==================########
    # System Configuration initialization.

    def _create_system(
        pdb,
        forcefield,
        nonbondedMethod="NoCutoff",
        nonbondedCutoff=None,
        ewaldErrorTolerance=None,
        constraints="None",
        rigidWater=False,
        constraintTolerance=None,
        **kwargs,
    ):
        # Create a dictionary to hold system parameters
        system_params = {
            "nonbondedMethod": nonbondedMethod,
            "constraints": constraints,
            "rigidWater": rigidWater,
        }

        # Set nonbondedCutoff if applicable
        if (
            nonbondedMethod in ["PME", "CutoffNonPeriodic", "CutoffPeriodic"]
            and nonbondedCutoff is not None
        ):
            system_params["nonbondedCutoff"] = nonbondedCutoff

        # Set ewaldErrorTolerance if PME is used
        if nonbondedMethod == "PME" and ewaldErrorTolerance is not None:
            system_params["ewaldErrorTolerance"] = ewaldErrorTolerance

        # Set constraintTolerance if constraints are used
        if constraints in ["HBonds", " AllBonds"] and constraintTolerance is not None:
            system_params["constraintTolerance"] = constraintTolerance
        elif system_params["rigidWater"] and constraintTolerance is not None:
            system_params["constraintTolerance"] = constraintTolerance

        # Update system_params with any additional parameters provided
        system_params.update(kwargs)
        system = forcefield.createSystem(pdb.topology, **system_params)
        return system

    ########==================Integrator==================########
    # Integrator
    def _define_integrator(
        integrator_type="LangevinMiddle",
        temperature=300 * kelvin,
        friction=1.0 / picoseconds,
        timestep=0.004 * picoseconds,
        **kwargs,
    ):
        # Create a dictionary to hold integrator parameters
        integrator_params = {
            "temperature": temperature,
            "friction": friction,
            "timestep": timestep,
        }

        # Update integrator_params with any additional parameters provided
        integrator_params.update(kwargs)

        # Create the integrator
        if integrator_type == "LangevinMiddle":
            integrator = LangevinMiddleIntegrator(**integrator_params)
        elif integrator_type == "Verlet":
            integrator = VerletIntegrator(**integrator_params)
        elif integrator_type == "Brownian":
            integrator = BrownianIntegrator(**integrator_params)
        else:
            raise Exception("Integrator type not recognized")

        return integrator

    def _prompt_summary(self, query: str):
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
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)

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

        # add filename to registry
        file_description = "Simulation Parameters"
        self.path_registry.map_path(filename, filename, file_description)

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
        print("Setting up forcefields :", Forcefield, Water_model)
        st.markdown("Setting up forcefields", unsafe_allow_html=True)
        # check if forcefields end in .xml
        if Forcefield.endswith(".xml") and Water_model.endswith(".xml"):
            forcefield = ForceField(Forcefield, Water_model)
        # adding forcefield to registry

        # Load the PDB file
        CleaningTools(self.path_registry)
        pdbfile = self.path_registry.get_mapped_path(params["File Path"])
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
            st.markdown("Setting up Langevin integrator", unsafe_allow_html=True)
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
            st.markdown("Setting up Verlet integrator", unsafe_allow_html=True)
            integrator = VerletIntegrator(float(_timestep) * picoseconds)

        simulation = Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)
        simulation.minimizeEnergy()
        # save initial positions to registry
        file_name = "initial_positions.pdb"
        with open(file_name, "w") as f:
            PDBFile.writeFile(
                simulation.topology,
                simulation.context.getState(getPositions=True).getPositions(),
                f,
            )
        print("Initial Positions saved to initial_positions.pdb")
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
        self.path_registry.map_path(file_name1, f"{name}.pdb", file_description1)
        file_name2 = "simulation_data.csv"
        file_description2 = (
            "Simulation Data, containing step, potential energy, and temperature"
        )
        self.path_registry.map_path(file_name2, f"{name}.csv", file_description2)

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
    description = """This tool will set up the simulation objects
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
            sim_fxns = SimulationFunctions(path_registry=self.path_registry)
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
            sim_fxns._setup_and_run_simulation(parameters)
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


#######==================System Configuration==================########
# System Configuration
class SetUpandRunFunctionInput(BaseModel):
    pdb_id: str
    forcefield_files: List[str]
    save: bool = Field(
        True,
        description=(
            (
                "Set to 'True' (default) to save the log files and trajectories "
                "of the simulation. "
                "If set to 'False', "
                "the simulation is considered as being in a testing "
                "or preliminary scripting stage, utilizing default parameters and "
                "results are not saved. "
                "This second setting is ideal for initial experimentation or "
                "basic script development before customizing the "
                "script for final use."
            )
        ),
    )

    system_params: Dict[str, Any] = Field(
        {
            "nonbondedMethod": "NoCutoff",
            "nonbondedCutoff": "1 * nanometers",
            "ewaldErrorTolerance": None,
            "constraints": "None",
            "rigidWater": False,
            "constraintTolerance": None,
            "solvate": False,
        },
        description=(
            "Parameters for the openmm system. "
            "For nonbondedMethod, you can choose from the following:\n"
            "NoCutoff, CutoffNonPeriodic, CutoffPeriodic, Ewald, PME. "
            "If anything but NoCutoff is chosen,"
            "you have to include a nonbondedCutoff"
            "and a constrainTolerance.\n"
            "If PME is chosen,"
            "you have to include an ewaldErrorTolerance too."
            "For constraints, you can choose from the following:\n"
            "None, HBonds, AllBonds or OnlyWater."
            "For rigidWater, you can choose from the following:\n"
            "True, False.\n"
            "Finally, if you want to solvate the system, before the simulation,"
            "you can set solvate to True.\n"
            "Example1:\n"
            "{'nonbondedMethod': 'NoCutoff',\n"
            "'constraints': 'None',\n"
            "'rigidWater': False}\n"
            "Example2:\n"
            "{'nonbondedMethod': 'CutoffPeriodic',\n"
            "'nonbondedCutoff': 1.0,\n"
            "'constraints': 'HBonds',\n"
            "'rigidWater': True,\n"
            "'constraintTolerance': 0.00001,\n"
            "'solvate': True} "
        ),
    )
    integrator_params: Dict[str, Any] = Field(
        {
            "integrator_type": "LangevinMiddle",
            "Temperature": "300 * kelvin",
            "Friction": "1.0 / picoseconds",
            "Timestep": "0.002 * picoseconds",
            "Pressure": "1.0 * bar",
        },
        description="""Parameters for the openmm integrator.""",
    )
    simulation_params: Dict[str, Any] = Field(
        {
            "Ensemble": "NVT",
            "Number of Steps": 5000,
            "record_interval_steps": 100,
            "record_params": ["step", "potentialEnergy", "temperature"],
        },
        description="""Parameters for the openmm simulation.
        The ensemble can be NPT, NVT or NVE.
        The number of steps is the number of steps the simulation will run for.
        record_interval_steps is the number of steps between each record:
        hould be the number of steps divided by 100.
        The record_params is a list of parameters that will
        be recorded during the simulation The options are:
        [Step,Time,Speed,Progress,RemainingTime,ElapsedTime,
        PotentialEnergy,KineticEnergy,TotalEnergy,
        Temperature,Volume,Density]""",
    )


#########===================================================================############


class OpenMMSimulation:
    def __init__(
        self,
        input_params: SetUpandRunFunctionInput,
        path_registry: PathRegistry,
        save: bool,
        sim_id: str,
        pdb_id: str,
    ):
        self.params = input_params
        self.save = save
        self.sim_id = sim_id
        self.pdb_id = pdb_id
        self.int_params = (
            self.params.integrator_params
            if self.params.integrator_params is not None
            else {
                "integrator_type": "LangevinMiddle",
                "Temperature": 300 * kelvin,
                "Friction": 1.0 / picoseconds,
                "Timestep": 0.002 * picoseconds,
                "Pressure": 1.0 * bar,
            }
        )

        self.sys_params = (
            self.params.system_params
            if self.params.system_params is not None
            else {
                "nonbondedMethod": NoCutoff,
                "nonbondedCutoff": 1 * nanometers,
                "ewaldErrorTolerance": None,
                "constraints": AllBonds,
                "rigidWater": True,
                "constraintTolerance": 0.000001,
                "solvate": False,
            }
        )

        self.sim_params = (
            self.params.simulation_params
            if self.params.simulation_params is not None
            else {
                "Ensemble": "NVT",
                "Number of Steps": 5000,
                "record_interval_steps": 100,
                "record_params": ["step", "potentialEnergy", "temperature"],
            }
        )

        self.path_registry = path_registry

    def setup_system(self):
        print("Building system...")
        st.markdown("Building system", unsafe_allow_html=True)
        self.pdb_id = self.params.pdb_id
        self.pdb_path = self.path_registry.get_mapped_path(self.pdb_id)
        self.pdb = PDBFile(self.pdb_path)
        self.forcefield = ForceField(*self.params.forcefield_files)
        self.system = self._create_system(self.pdb, self.forcefield, **self.sys_params)

        if self.sys_params.get("nonbondedMethod", None) in [
            CutoffPeriodic,
            PME,
        ]:
            if self.sim_params["Ensemble"] == "NPT":
                self.system.addForce(
                    MonteCarloBarostat(
                        self.int_params["Pressure"],
                        self.int_params["Temperature"],
                        self.sim_params.get("barostatInterval", 25),
                    )
                )

    def setup_integrator(self):
        print("Setting up integrator...")
        st.markdown("Setting up integrator", unsafe_allow_html=True)
        int_params = self.int_params
        integrator_type = int_params.get("integrator_type", "LangevinMiddle")

        if integrator_type == "LangevinMiddle":
            self.integrator = LangevinMiddleIntegrator(
                int_params["Temperature"],
                int_params["Friction"],
                int_params["Timestep"],
            )
        elif integrator_type == "LangevinIntegrator":
            self.integrator = LangevinIntegrator(
                int_params["Temperature"],
                int_params["Friction"],
                int_params["Timestep"],
            )
        else:
            raise ValueError("Invalid integrator type")

        self.integrator.setConstraintTolerance(
            self.sys_params.get("constraintTolerance", 0.000001)
        )

    def create_simulation(self):
        print("Creating simulation...")
        st.markdown("Creating simulation", unsafe_allow_html=True)
        self.simulation = Simulation(
            self.modeller.topology,
            self.system,
            self.integrator,
            Platform.getPlatformByName("CPU"),
        )
        self.simulation.context.setPositions(self.modeller.positions)

        # TEMPORARY FILE MANAGEMENT OR PATH REGISTRY MAPPING
        if self.save:
            trajectory_name = self.path_registry.write_file_name(
                type=FileType.RECORD,
                record_type="TRAJ",
                protein_file_id=self.pdb_id,
                Sim_id=self.sim_id,
                term="dcd",
            )
            topology_name = self.path_registry.write_file_name(
                type=FileType.RECORD,
                record_type="TOP",
                protein_file_id=self.pdb_id,
                Sim_id=self.sim_id,
                term="pdb",
            )

            log_name = self.path_registry.write_file_name(
                type=FileType.RECORD,
                record_type="LOG",
                protein_file_id=self.pdb_id,
                Sim_id=self.sim_id,
                term="txt",
            )

            traj_desc = (
                f"Simulation trajectory for protein {self.pdb_id}"
                f" and simulation {self.sim_id}"
            )
            top_desc = (
                f"Simulation topology for protein"
                f"{self.pdb_id} and simulation {self.sim_id}"
            )
            log_desc = (
                f"Simulation state log for protein {self.pdb_id} "
                f"and simulation {self.sim_id}"
            )

            self.simulation.reporters.append(
                DCDReporter(
                    f"{trajectory_name}",
                    self.sim_params["record_interval_steps"],
                )
            )
            self.simulation.reporters.append(
                PDBReporter(
                    f"{topology_name}",
                    self.sim_params["record_interval_steps"],
                )
            )
            self.simulation.reporters.append(
                StateDataReporter(
                    f"{log_name}",
                    self.sim_params["record_interval_steps"],
                    step=True,
                    potentialEnergy=True,
                    temperature=True,
                    separator="\t",
                )
            )
            self.registry_records = [
                ("holder", f"files/records/{trajectory_name}", traj_desc),
                ("holder", f"files/records/{log_name}", log_desc),
                ("holder", f"files/records/{topology_name}", top_desc),
            ]

            # TODO add checkpoint too?

        else:
            self.simulation.reporters.append(
                DCDReporter(
                    "temp_trajectory.dcd",
                    self.sim_params["record_interval_steps"],
                )
            )
            self.simulation.reporters.append(
                PDBReporter(
                    "temp_topology.pdb",
                    self.sim_params["record_interval_steps"],
                )
            )
            self.simulation.reporters.append(
                StateDataReporter(
                    "temp_log.txt",
                    self.sim_params["record_interval_steps"],
                    step=True,
                    potentialEnergy=True,
                    temperature=True,
                    separator="\t",
                )
            )

    def _create_system(
        self,
        pdb,
        forcefield,
        nonbondedMethod="NoCutoff",
        nonbondedCutoff=None,
        ewaldErrorTolerance=None,
        constraints="None",
        rigidWater=False,
        constraintTolerance=None,
        solvate=False,
        **kwargs,
    ):
        # Create a dictionary to hold system parameters
        system_params = {
            "nonbondedMethod": nonbondedMethod,
            "constraints": constraints,
            "rigidWater": rigidWater,
        }

        # Set nonbondedCutoff if applicable Had to double if pre-commit
        if nonbondedMethod in ["PME", "CutoffNonPeriodic", "CutoffPeriodic"]:
            if nonbondedCutoff is not None:
                system_params["nonbondedCutoff"] = nonbondedCutoff

        # Set ewaldErrorTolerance if PME is used
        if nonbondedMethod == "PME" and ewaldErrorTolerance is not None:
            system_params["ewaldErrorTolerance"] = ewaldErrorTolerance

        # Set constraintTolerance if constraints are used
        if constraints in ["HBonds", "AllBonds"] and constraintTolerance is not None:
            pass
        elif system_params["rigidWater"] and constraintTolerance is not None:
            pass

        # Update system_params with any additional parameters provided
        system_params.update(kwargs)

        # if use_constraint_tolerance:
        #    constraintTolerance = system_params.pop('constraintTolerance')
        self.modeller = Modeller(pdb.topology, pdb.positions)
        if solvate:
            try:
                self.modeller.addSolvent(forcefield)
            except ValueError as e:
                print("Error adding solvent", type(e).__name__, "–", e)
                if "No Template for" in str(e):
                    raise ValueError(str(e))
            except AttributeError as e:
                print("Error adding solvent: ", type(e).__name__, "–", e)
                print("Trying to add solvent with 1 nm padding")
                if "NoneType" and "value_in_unit" in str(e):
                    try:
                        self.modeller.addSolvent(forcefield, padding=1 * nanometers)
                    except Exception as e:
                        print("Error adding solvent", type(e).__name__, "–", e)
                        raise (e)
            system = forcefield.createSystem(self.modeller.topology, **system_params)
        else:
            system = forcefield.createSystem(self.modeller.topology, **system_params)

        return system

    def unit_to_string(self, unit):
        """Needed to convert units to strings for the script
        Otherwise internal __str()__ method makes the script
        not runnable"""
        return f"{unit.value_in_unit(unit.unit)}*{unit.unit.get_name()}"

    def _construct_script_content(
        self,
        pdb_path,
        forcefield_files,
        nonbonded_method,
        constraints,
        rigid_water,
        constraint_tolerance,
        nonbonded_cutoff,
        ewald_error_tolerance,
        hydrogen_mass,
        time_step,
        temperature,
        friction,
        ensemble,
        pressure,
        record_interval_steps,
        solvate,
        integrator_type,
    ):
        script_content = f"""
        # This script was generated by MDagent-Setup.

        from openmm import *
        from openmm.app import *
        from openmm.unit import *

        # Input Files
        pdb = PDBFile('{pdb_path}')
        forcefield = ForceField({forcefield_files})

        # System Configuration
        nonbondedMethod = {nonbonded_method}
        constraints = {constraints}
        rigidWater = {rigid_water}
        """
        if rigid_water and constraint_tolerance is not None:
            script_content += f"constraintTolerance = {constraint_tolerance}\n"

        # Conditionally add nonbondedCutoff

        if nonbonded_method != NoCutoff:
            script_content += f"nonbondedCutoff = {nonbonded_cutoff}\n"
        if nonbonded_method == PME:
            script_content += f"ewaldErrorTolerance = {ewald_error_tolerance}\n"
        if hydrogen_mass:
            script_content += f"hydrogenMass = {hydrogen_mass}\n"

        # ... other configurations ...
        script_content += f"""
        # Integration Options
        dt = {time_step}
        temperature = {temperature}
        friction = {friction}
        """
        if ensemble == "NPT":
            script_content += f"""
            pressure = {pressure}
            barostatInterval = {self.sim_params.get("barostatInterval", 25)}
            """

        # ... other integration options ...
        script_content += f"""
        # Simulation Options
        steps = {self.sim_params.get("Number of Steps", record_interval_steps)}
        equilibrationSteps = 1000
        platform = Platform.getPlatformByName('CPU')
        dcdReporter = DCDReporter('trajectory.dcd', 1000)
        pdbReporter = PDBReporter('trajectory.pdb', 1000)
        dataReporter = StateDataReporter('log.txt', {record_interval_steps},
            totalSteps=steps,
            step=True, speed=True, progress=True, elapsedTime=True, remainingTime=True,
            potentialEnergy=True, temperature=True, volume=True, density=True,
            separator='\t')
        checkpointReporter = CheckpointReporter('checkpoint.chk', 5000)

        # Minimize and Equilibrate
        # ... code for minimization and equilibration ...

        # Simulate

        print('Building system...')
        modeller = Modeller(pdb.topology, pdb.positions)
        """
        if solvate:
            script_content += (
                """modeller.addSolvent(forcefield, padding=1*nanometers)"""
            )

        if nonbonded_method == NoCutoff:
            if hydrogen_mass:
                script_content += """
            system = forcefield.createSystem(modeller.topology,
            nonbondedMethod=nonbondedMethod, constraints=constraints,
            rigidWater=rigidWater, hydrogenMass=hydrogenMass)
            """
            else:
                script_content += """
            system = forcefield.createSystem(modeller.topology,
            nonbondedMethod=nonbondedMethod, constraints=constraints,
            rigidWater=rigidWater)
            """
        if nonbonded_method == CutoffNonPeriodic or nonbonded_method == CutoffPeriodic:
            if hydrogen_mass:
                script_content += """
                system = forcefield.createSystem(modeller.topology,
                nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff,
                constraints=constraints, rigidWater=rigidWater,
                hydrogenMass=hydrogenMass)
            """
            else:
                script_content += """
                system = forcefield.createSystem(modeller.topology,
                nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff,
                constraints=constraints, rigidWater=rigidWater)
            """
        if nonbonded_method == PME:
            if hydrogen_mass:
                script_content += """
            system = forcefield.createSystem(modeller.topology,
            nonbondedMethod=nonbondedMethod,
            nonbondedCutoff=nonbondedCutoff, ewaldErrorTolerance=ewaldErrorTolerance,
            constraints=constraints, rigidWater=rigidWater, hydrogenMass=hydrogenMass)
            """
            else:
                script_content += """
            system = forcefield.createSystem(modeller.topology,
            nonbondedMethod=nonbondedMethod,
            nonbondedCutoff=nonbondedCutoff, ewaldErrorTolerance=ewaldErrorTolerance,
            constraints=constraints, rigidWater=rigidWater)
            """
        if ensemble == "NPT":
            script_content += """
            system.addForce(MonteCarloBarostat(pressure, temperature, barostatInterval))
            """

        if integrator_type == "LangevinMiddle" and constraints != "None":
            script_content += """
        integrator = LangevinMiddleIntegrator(temperature, friction, dt)
        integrator.setConstraintTolerance(constraintTolerance)
        simulation = Simulation(modeller.topology, system, integrator, platform)
        simulation.context.setPositions(modeller.positions)
        """
        if integrator_type == "LangevinMiddle" and constraints == "None":
            script_content += """
            integrator = LangevinMiddleIntegrator(temperature, friction, dt)
            simulation = Simulation(modeller.topology, system, integrator, platform)
            simulation.context.setPositions(modeller.positions)
        """

        script_content += """
        # Minimize and Equilibrate

        print('Performing energy minimization...')
        simulation.minimizeEnergy()
        print('Equilibrating...')
        simulation.context.setVelocitiesToTemperature(temperature)
        simulation.step(equilibrationSteps)

        # Simulate

        print('Simulating...')
        simulation.reporters.append(dcdReporter)
        simulation.reporters.append(pdbReporter)
        simulation.reporters.append(dataReporter)
        simulation.reporters.append(checkpointReporter)
        simulation.currentStep = 0
        simulation.step(steps)
"""
        return script_content

    def write_standalone_script(self, filename="reproduce_simulation.py"):
        """Extracting parameters from the class instance
        Inspired by the code snippet provided from openmm-setup
        https://github.com/openmm/openmm-setup
        """

        pdb_path = self.pdb_path
        forcefield_files = ", ".join(
            f"'{file}'" for file in self.params["forcefield_files"]
        )
        nonbonded_method = self.sys_params.get("nonbondedMethod", NoCutoff)
        nbCo = self.sys_params.get("nonbondedCutoff", 1 * nanometers)
        nonbonded_cutoff = self.unit_to_string(nbCo)
        constraints = self.sys_params.get("constraints", "None")
        rigid_water = self.sys_params.get("rigidWater", False)
        ewald_error_tolerance = self.sys_params.get("ewaldErrorTolerance", 0.0005)
        constraint_tolerance = self.sys_params.get("constraintTolerance", None)
        hydrogen_mass = self.sys_params.get("hydrogenMass", None)
        solvate = self.sys_params.get("solvate", False)

        integrator_type = self.int_params.get("integrator_type", "LangevinMiddle")
        friction = self.int_params.get("Friction", 1.0 / picoseconds)
        friction = f"{friction.value_in_unit(friction.unit)}{friction.unit.get_name()}"
        _temp = self.int_params.get("Temperature", 300 * kelvin)
        temperature = self.unit_to_string(_temp)

        t_step = self.int_params.get("Timestep", 0.004 * picoseconds)
        time_step = self.unit_to_string(t_step)
        press = self.int_params.get("Pressure", 1.0 * bar)
        pressure = self.unit_to_string(press)
        ensemble = self.sim_params.get("Ensemble", "NVT")
        self.sim_params.get("Number of Steps", 10000)
        record_interval_steps = self.sim_params.get("record_interval_steps", 1000)

        script_content = self._construct_script_content(
            pdb_path,
            forcefield_files,
            nonbonded_method,
            constraints,
            rigid_water,
            constraint_tolerance,
            nonbonded_cutoff,
            ewald_error_tolerance,
            hydrogen_mass,
            time_step,
            temperature,
            friction,
            ensemble,
            pressure,
            record_interval_steps,
            solvate,
            integrator_type,
        )

        # Remove leading spaces for proper formatting
        def remove_leading_spaces(text):
            lines = text.split("\n")
            stripped_lines = [line.lstrip() for line in lines]
            return "\n".join(stripped_lines)

        script_content = remove_leading_spaces(script_content)
        script_content = textwrap.dedent(script_content).strip()

        # Write to file
        directory = "files/simulations"
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(f"{directory}/{filename}", "w") as file:
            file.write(script_content)

        print(f"Standalone simulation script written to {directory}/{filename}")
        st.markdown("Standalone simulation script written", unsafe_allow_html=True)

    def run(self):
        # Minimize and Equilibrate
        print("Performing energy minimization...")
        st.markdown("Performing energy minimization", unsafe_allow_html=True)

        self.simulation.minimizeEnergy()
        print("Minimization complete!")
        top_name = f"files/pdb/{self.sim_id}_initial_positions.pdb"
        top_description = f"Initial positions for simulation {self.sim_id}"
        with open(top_name, "w") as f:
            PDBFile.writeFile(
                self.simulation.topology,
                self.simulation.context.getState(getPositions=True).getPositions(),
                f,
            )
        self.path_registry.map_path(f"top_{self.sim_id}", top_name, top_description)
        print("Initial Positions saved to initial_positions.pdb")
        st.markdown("Minimization complete! Equilibrating...", unsafe_allow_html=True)
        print("Equilibrating...")
        _temp = self.int_params["Temperature"]
        self.simulation.context.setVelocitiesToTemperature(_temp)
        _eq_steps = self.sim_params.get("equilibrationSteps", 1000)
        self.simulation.step(_eq_steps)
        # Simulate
        print("Simulating...")
        st.markdown("Simulating...", unsafe_allow_html=True)
        self.simulation.currentStep = 0
        self.simulation.step(self.sim_params["Number of Steps"])
        print("Done!")
        st.markdown("Done!", unsafe_allow_html=True)
        if not self.save:
            if os.path.exists("temp_trajectory.dcd"):
                os.remove("temp_trajectory.dcd")
            if os.path.exists("temp_log.txt"):
                os.remove("temp_log.txt")
            if os.path.exists("temp_checkpoint.chk"):
                os.remove("temp_checkpoint.chk")

        return "Simulation done!"


class SetUpandRunFunction(BaseTool):
    name: str = "SetUpandRunFunction"
    description: str = (
        "This tool will set up and run a short simulation of a protein. "
        "Then will write a standalone script that can be used "
        "to reproduce the simulation or change accordingly for "
        "a more elaborate simulation. It only runs short simulations because, "
        "if there are errors, you can try again changing the input"
    )

    args_schema: Type[BaseModel] = SetUpandRunFunctionInput

    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, **input_args):
        if self.path_registry is None:
            return "Path registry not initialized"
        input = self.check_system_params(input_args)
        error = input.get("error", None)
        if error:
            print(f"error found: {error}")
            return error

        try:
            pdb_id = input["pdb_id"]
            # check if pdb_id is in the registry or as 1XYZ_112233 format
            if pdb_id not in self.path_registry.list_path_names():
                return "No pdb_id found in input, use the file id not the file name"
        except KeyError:
            return "No pdb_id found in input"
        try:
            save = input["save"]  # either this simulation
            # to save or not the output files from this simulation
        except KeyError:
            save = True
            print(
                "No 'save' key found in input, setting to True. "
                "Record files will be deleted after script is written."
            )
        try:
            file_name = self.path_registry.write_file_name(
                type=FileType.SIMULATION,
                type_of_sim=input["simulation_params"]["Ensemble"],
                protein_file_id=pdb_id,
            )

            sim_id = self.path_registry.get_fileid(file_name, FileType.SIMULATION)
        except Exception as e:
            print(f"An exception was found: {str(e)}.")
            return f"An exception was found trying to write the filenames: {str(e)}."
        try:
            openmmsim = OpenMMSimulation(
                input, self.path_registry, save, sim_id, pdb_id
            )
            openmmsim.setup_system()
            openmmsim.setup_integrator()
            openmmsim.create_simulation()

            print("simulation set!")
            st.markdown("simulation set!", unsafe_allow_html=True)
        except ValueError as e:
            msg = str(e) + f"This were the inputs {input_args}"
            if "No template for" in msg:
                msg += (
                    "This error is likely due to non standard residues "
                    "in the protein, if you havent done it yet, try "
                    "cleaning the pdb file using the cleaning tool"
                )
            return msg
        except FileNotFoundError:
            return f"File not found, check File id. This were the inputs {input_args}"
        except OpenMMException as e:
            return f"OpenMM Exception: {str(e)}. This were the inputs {input_args}"
        try:
            openmmsim.run()
        except Exception as e:
            return (
                f"An exception was found: {str(e)}. Not a problem, thats one "
                "purpose of this tool: to run a short simulation to check for correct "
                "initialization. "
                ""
                "Try a) with different parameters like "
                "nonbondedMethod, constraints, etc \n or\n"
                "b) clean file inputs depending on error "
            )
        try:
            openmmsim.write_standalone_script(filename=file_name)
            self.path_registry.map_path(
                sim_id,
                f"files/simulations/{file_name}",
                f"Basic Simulation of Protein {pdb_id}",
            )
            if save:
                records = openmmsim.registry_records
                # move record files to files/records/
                print(os.listdir("."))
                if not os.path.exists("files/records"):
                    os.makedirs("files/records")
                for record in records:
                    os.rename(record[1].split("/")[-1], f"{record[1]}")
                for record in records:
                    record_list = list(record)
                    record_list[0] = self.path_registry.get_fileid(
                        record_list[1].split("/")[-1], FileType.RECORD
                    )
                    record = tuple(record_list)
                    self.path_registry.map_path(*record)
            return (
                "Simulation done! \n Summary: \n"
                "Record files written to files/records/ with IDs and descriptions: "
                f"{[(record[0],record[2]) for record in records]}\n"
                "Standalone script written to files/simulations/ with ID: "
                f"{sim_id}.\n"
                f"The initial topology file ID is top_{sim_id} saved in files/pdb/"
            )
        except Exception as e:
            print(f"An exception was found: {str(e)}.")
            return f"An exception was found trying to write the filenames: {str(e)}."

    def _parse_cutoff(self, cutoff):
        # Check if cutoff is already an OpenMM Quantity (has a unit)
        possible_units = ["nm", "nanometer", "nanometers", "angstrom", "angstroms", "a"]

        if isinstance(cutoff, unit.Quantity):
            return cutoff

        # Convert to string in case it's not (e.g., int or float)
        cutoff = str(cutoff)
        if cutoff[-1] == "s":
            cutoff = cutoff[:-1]

        # Remove spaces and convert to lowercase for easier parsing
        cutoff = cutoff.replace(" ", "").lower()

        # Check for multiplication symbol and split if necessary
        if "*" in cutoff:
            # Split on the '*' and extract the numerical part and the unit part
            num_part, unit_part = cutoff.split("*")

            # Convert the numerical part to a float
            num_value = float(num_part)
        else:
            # If there is no '*', it's either a number or a string like "1nm"
            # Attempt to convert directly to float; if it fails,
            # it must have a unit like "nm" or "angstrom"
            try:
                num_value = float(cutoff)
                unit_part = "nm"
            except ValueError:
                for possible_unit in possible_units:
                    if possible_unit in cutoff:
                        num_value = float(cutoff.replace(possible_unit, ""))
                        unit_part = possible_unit
                        break
                else:
                    # Use regular expression to extract the
                    # numerical part and the unit part
                    match = re.match(r"([+-]?[0-9]*\.?[0-9]+)([a-zA-Z]*)", cutoff)
                    if match:
                        num_part, unit_part = match.groups()
                        raise ValueError(
                            f"""Unknown unit for nonbondedCutoff
                                         got {unit_part}. Try using nm or angstroms as
                                         value * unit."""
                        )

        # Now convert the unit part to an OpenMM unit
        if unit_part in ["nm", "nanometer", "nanometers"]:
            return num_value * unit.nanometers
        elif unit_part in ["angstrom", "angstroms", "a"]:
            return num_value * unit.angstroms

        else:
            # If the unit is not recognized, raise an error
            raise ValueError(
                f"""Unknown unit for nonbondedCutoff
                    got {unit_part}. Try using nm or angstroms as
                    value * unit."""
            )

    def _parse_parameter(self, parameter, default_unit, possible_units):
        """
        Parse a parameter and return it as an OpenMM Quantity with the correct unit.

        Args:
            parameter (float, str, or unit.Quantity): The input parameter value
            default_unit (unit.Unit): The default unit to use if none is provided
            possible_units (dict): A mapping of strings to their respective unit objects

        Returns:
            unit.Quantity: The parameter as an OpenMM Quantity with the correct unit.
        """
        error_msg = ""
        if isinstance(parameter, unit.Quantity):
            return parameter, error_msg

        # Convert to string in case it's not (e.g., int or float)
        parameter_str = str(parameter)

        # Remove spaces and convert to lowercase for easier parsing
        parameter_str = parameter_str.replace(" ", "").lower()

        # Check for multiplication symbol and split if necessary
        # e.g. "1*kelvin" or "1*ps^-1"
        if "*" in parameter_str:
            num_part, unit_part = parameter_str.split("*")
            num_value = float(num_part)
        elif "poundforce/inch^2" in parameter_str:
            num_value = float(parameter_str.replace("poundforce/inch^2", ""))
            unit_part = "poundforce/inch^2"
        # Check for division symbol and split if necessary
        # e.g. "1/ps" or "1/ps^-1"
        elif "/" in parameter_str:
            num_part, unit_part = parameter_str.split("/")
            num_value = float(num_part)
            unit_part = "/" + unit_part
        elif "^-1" in parameter_str:
            parameter_str = parameter_str.replace("^-1", "")
            match = re.match(r"^(\d+(?:\.\d+)?)([a-zA-Z]+)$", parameter_str)
            num_value = float(match.group(1))
            unit_part = "/" + match.group(2)
        else:
            # Attempt to convert directly to float; if it fails,
            # it must have a unit like "K", "ps", etc.
            try:
                num_value = float(parameter_str)
                unit_part = default_unit
            except ValueError:
                match = re.match(r"([+-]?[0-9]*\.?[0-9]+)([a-zA-Z]*)", parameter_str)
                if match:
                    num_part, unit_part = match.groups()
                    num_value = float(num_part)
                else:
                    error_msg += f"Invalid format for parameter: '{parameter_str}'."

        # Convert the unit part to an OpenMM unit
        if unit_part.lower() in possible_units:
            return num_value * possible_units[unit_part.lower()], error_msg
        else:
            # If the unit is not recognized, raise an error
            error_msg += f"""Unknown unit '{unit_part}' for parameter.
            Valid units include: {list(possible_units.keys())}."""

            return parameter, error_msg

    def parse_temperature(self, temperature):
        possible_units = {
            "k": unit.kelvin,
            "kelvin": unit.kelvin,
        }
        return self._parse_parameter(temperature, "k", possible_units)

    def parse_friction(self, friction):
        possible_units = {
            "/ps": (1 / unit.picoseconds),
            "/picosecond": (1 / unit.picoseconds),
            "/picoseconds": (1 / unit.picoseconds),
            "picosecond^-1": (1 / unit.picoseconds),
            "picoseconds^-1": (1 / unit.picoseconds),
            "/ps^-1": (1 / unit.picoseconds),
            "ps^-1": (1 / unit.picoseconds),
            "1*ps^-1": (1 / unit.picoseconds),
        }
        return self._parse_parameter(friction, "1/ps", possible_units)

    def parse_timestep(self, timestep):
        possible_units = {
            "ps": unit.picoseconds,
            "picosecond": unit.picoseconds,
            "picoseconds": unit.picoseconds,
            "fs": unit.femtoseconds,
            "femtosecond": unit.femtoseconds,
            "femtoseconds": unit.femtoseconds,
            "ns": unit.nanoseconds,
            "nanosecond": unit.nanoseconds,
            "nanoseconds": unit.nanoseconds,
        }
        return self._parse_parameter(timestep, "ps", possible_units)

    def parse_pressure(self, pressure):
        possible_units = {
            "bar": unit.bar,
            "atm": unit.atmospheres,
            "atmosphere": unit.atmospheres,
            "pascal": unit.pascals,
            "pascals": unit.pascals,
            "pa": unit.pascals,
            "poundforce/inch^2": unit.psi,
            "psi": unit.psi,
        }
        return self._parse_parameter(pressure, "bar", possible_units)

    def _process_parameters(self, user_params, param_type="system_params"):
        """
        Process user provided parameters,
        converting strings to openmm objects if necessary.
        """
        error_msg = ""
        processed_params = {}
        if param_type == "system_params":
            for key, value in user_params.items():
                if key == "nonbondedMethod" or key == "nonbondedmethod":
                    if value == "NoCutoff":
                        processed_params[key] = NoCutoff
                    elif value == "PME":
                        processed_params[key] = PME
                    elif value == "CutoffPeriodic":
                        processed_params[key] = CutoffPeriodic
                    elif value == "CutoffNonPeriodic":
                        processed_params[key] = CutoffNonPeriodic
                    elif value == "Ewald":
                        processed_params[key] = Ewald
                    else:
                        # Assume it's already an openmm object
                        processed_params[key] = value
                if key == "nonbondedCutoff" or key == "nonbondedcutoff":
                    try:
                        processed_params[key] = self._parse_cutoff(value)
                    except ValueError as e:
                        error_msg += f"Invalid nonbondedCutoff: {e}. \n"
                if key == "ewaldErrorTolerance" or key == "ewalderrortolerance":
                    try:
                        processed_params[key] = float(value)
                    except TypeError as e:
                        error_msg += (
                            f"Invalid ewaldErrorTolerance: {e}. "
                            "If you are using null or None, "
                            "just dont include it "
                            "as part of the parameters.\n"
                        )
                if key == "constraints":
                    try:
                        if type(value) == str:
                            if value == "None":
                                processed_params[key] = None
                            elif value == "HBonds":
                                processed_params[key] = HBonds
                            elif value == "AllBonds":
                                processed_params[key] = AllBonds
                            elif value == "HAngles":
                                processed_params[key] = HAngles
                            else:
                                error_msg += (
                                    f"Invalid constraints: Got {value}. "
                                    "Try using None, HBonds, AllBonds or "
                                    "HAngles\n"
                                )
                        else:
                            processed_params[key] = value
                    except TypeError as e:
                        error_msg += (
                            f"Invalid constraints: {e}. If you are using "
                            "null or None, just dont include as "
                            "part of the parameters.\n"
                        )
                if key == "rigidWater" or key == "rigidwater":
                    if type(value) == bool:
                        processed_params[key] = value
                    elif value == "True":
                        processed_params[key] = True
                    elif value == "False":
                        processed_params[key] = False
                    else:
                        error_msg += (
                            f"Invalid rigidWater: got {value}. "
                            "Try using True or False.\n"
                        )
                if key == "constraintTolerance" or key == "constrainttolerance":
                    try:
                        processed_params[key] = float(value)
                    except ValueError as e:
                        error_msg += f"Invalid constraintTolerance: {e}."
                    except TypeError as e:
                        error_msg += (
                            f"Invalid constraintTolerance: {e}. If "
                            "constraintTolerance is null or None, "
                            "just dont include as part of "
                            "the parameters.\n"
                        )
                if key == "solvate":
                    try:
                        if type(value) == bool:
                            processed_params[key] = value
                        elif value == "True":
                            processed_params[key] = True
                        elif value == "False":
                            processed_params[key] = False
                        else:
                            error_msg += (
                                f"Invalid solvate: got {value}. "
                                "Use either True or False.\n"
                            )
                    except TypeError as e:
                        error_msg += (
                            f"Invalid solvate: {e}. If solvate is null or "
                            "None, just dont include as part of "
                            "the parameters.\n"
                        )

            return processed_params, error_msg
        if param_type == "integrator_params":
            for key, value in user_params.items():
                if key == "integrator_type" or key == "integratortype":
                    if value == "LangevinMiddle" or value == LangevinMiddleIntegrator:
                        processed_params[key] = "LangevinMiddle"
                    elif value == "Langevin" or value == LangevinIntegrator:
                        processed_params[key] = "Langevin"
                    elif value == "Verlet" or value == VerletIntegrator:
                        processed_params[key] = "Verlet"
                    elif value == "Brownian" or value == BrownianIntegrator:
                        processed_params[key] = "Brownian"
                    else:
                        error_msg += (
                            f"Invalid integrator_type: got {value}. "
                            "Try using LangevinMiddle, Langevin, "
                            "Verlet, or Brownian.\n"
                        )
                if key == "Temperature" or key == "temperature":
                    temperature, msg = self.parse_temperature(value)
                    processed_params[key] = temperature
                    error_msg += msg
                if key == "Friction" or key == "friction":
                    friction, msg = self.parse_friction(value)
                    processed_params[key] = friction
                    error_msg += msg
                if key == "Timestep" or key == "timestep":
                    timestep, msg = self.parse_timestep(value)
                    processed_params[key] = timestep
                    error_msg += msg
                if key == "Pressure" or key == "pressure":
                    pressure, msg = self.parse_pressure(value)
                    processed_params[key] = pressure
                    error_msg += msg

            return processed_params, error_msg
        if param_type == "simulation_params":
            for key, value in user_params.items():
                if key == "Ensemble" or key == "ensemble":
                    if value == "NPT":
                        processed_params[key] = "NPT"
                    elif value == "NVT":
                        processed_params[key] = "NVT"
                    elif value == "NVE":
                        processed_params[key] = "NVE"
                    else:
                        error_msg += (
                            f"Invalid Ensemble. got {value}. "
                            "Try using NPT, NVT, or NVE.\n"
                        )

                if key == "Number of Steps" or key == "number of steps":
                    processed_params[key] = int(value)
                if key == "record_interval_steps" or key == "record interval steps":
                    processed_params[key] = int(value)
                if key == "record_params" or key == "record params":
                    processed_params[key] = value
            return processed_params, error_msg

    def check_system_params(cls, values):
        """Check that the system parameters are valid."""
        # lowercase all keys in the dictionary
        error_msg = ""
        values = {k.lower(): v for k, v in values.items()}

        system_params = values.get("system_params")
        if system_params:
            system_params, msg = cls._process_parameters(
                system_params, param_type="system_params"
            )
            if msg != "":
                error_msg += msg
        else:
            system_params = {
                "nonbondedMethod": NoCutoff,
                "nonbondedCutoff": 1 * nanometers,
                "ewaldErrorTolerance": None,
                "constraints": AllBonds,
                "rigidWater": True,
                "constraintTolerance": 0.00001,
                "solvate": False,
            }
        integrator_params = values.get("integrator_params")
        if integrator_params:
            integrator_params, msg = cls._process_parameters(
                integrator_params, param_type="integrator_params"
            )
            if msg != "":
                error_msg += msg
        else:
            integrator_params = {
                "integrator_type": "LangevinMiddle",
                "Temperature": 300 * kelvin,
                "Friction": 1.0 / picoseconds,
                "Timestep": 0.004 * picoseconds,
                "Pressure": 1.0 * bar,
            }
        simulation_params = values.get("simulation_params")
        if simulation_params is None:
            simulation_params = {
                "Ensemble": "NVT",
                "Number of Steps": 10000,
                "record_interval_steps": 100,
                "record_params": ["step", "potentialEnergy", "temperature"],
            }
        # lowercase all keys in the dictionary

        # system_params = {k.lower(): v for k, v in system_params.items()}
        # integrator_params = {k.lower(): v for k, v in integrator_params.items()}
        # simulation_params = {k.lower(): v for k, v in simulation_params.items()}

        nonbondedMethod = system_params.get("nonbondedMethod")
        nonbondedCutoff = system_params.get("nonbondedCutoff")
        ewaldErrorTolerance = system_params.get("ewaldErrorTolerance")
        constraints = system_params.get("constraints")
        rigidWater = system_params.get("rigidWater")
        constraintTolerance = system_params.get("constraintTolerance")
        methods_with_cutoff = {
            "PME",
            "CutoffNonPeriodic",
            "CutoffPeriodic",
            "Ewald",
            PME,
            CutoffNonPeriodic,
            CutoffPeriodic,
            Ewald,
        }
        constraints_with_tolerance = {
            "HBonds",
            "AllBonds",
            "OnlyWater",
            HBonds,
            AllBonds,
        }

        if nonbondedMethod in methods_with_cutoff and nonbondedCutoff is None:
            error_msg += """nonbondedCutoff must be specified if
                        nonbondedMethod is not NoCutoff\n"""
        if nonbondedMethod in {"PME", PME} and ewaldErrorTolerance is None:
            error_msg += """ewaldErrorTolerance must be specified when
            nonbondedMethod is PME\n"""
        if constraints in constraints_with_tolerance and constraintTolerance is None:
            error_msg += """constraintTolerance must be specified when
                         constraints is HBonds or AllBonds"""
        if rigidWater and constraintTolerance is None:
            error_msg = "constraintTolerance must be specified if rigidWater is True"

        """Checking if the file is in the path"""
        pdb_id = values.get("pdb_id")
        if not pdb_id:
            error_msg += "The pdb id is not present in the inputs"

        """Validating the forcefield files and Integrator"""

        integrator_type = integrator_params.get("integrator_type")
        if integrator_type not in ["LangevinMiddle", "Verlet", "Brownian"]:
            error_msg += """integrator_type must be one of the following:
                             LangevinMiddle, Verlet, Brownian\n"""
        if integrator_type == "LangevinMiddle":
            friction = integrator_params.get("Friction")
            if friction is None:
                error_msg += """friction must be specified when
                            integrator_type is LangevinMiddle\n"""
            timestep = integrator_params.get("Timestep")
            if timestep is None:
                error_msg += """timestep must be specified when
                            integrator_type is LangevinMiddle\n"""
            temp = integrator_params.get("Temperature")
            if temp is None:
                error_msg += """temperature must be specified when
                integrator_type is LangevinMiddle\n"""

        if integrator_type == "Verlet":
            timestep = integrator_params.get("Timestep")
            if timestep is None:
                error_msg += """timestep must be specified when
                            integrator_type is Verlet\n"""
        if integrator_type == "Brownian":
            temperature = integrator_params.get("Temperature")
            if temperature is None:
                error_msg += """temperature must be specified when
                    integrator_type is Brownian\n"""

        # forcefield
        forcefield_files = values.get("forcefield_files")
        if forcefield_files is None or forcefield_files is []:
            print("Setting default forcefields")
            st.markdown("Setting default forcefields", unsafe_allow_html=True)
            forcefield_files = ["amber14-all.xml", "amber14/tip3pfb.xml"]
        elif len(forcefield_files) == 0:
            print("Setting default forcefields v2")
            st.markdown("Setting default forcefields", unsafe_allow_html=True)
            forcefield_files = ["amber14-all.xml", "amber14/tip3pfb.xml"]
        else:
            for file in forcefield_files:
                if file not in FORCEFIELD_LIST:
                    error_msg += "The forcefield file is not present"

        save = values.get("save", True)
        if type(save) != bool:
            error_msg += "save must be a boolean value"

        if error_msg != "":
            return {
                "error": error_msg
                + "\n Correct this and try again. \n Everthing else is fine"
            }
        values = {
            "pdb_id": pdb_id,
            "forcefield_files": forcefield_files,
            "save": save,
            "system_params": system_params,
            "integrator_params": integrator_params,
            "simulation_params": simulation_params,
        }
        # if no error, return the values
        return values

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


def create_simulation_input(pdb_path, forcefield_files):
    """
    This function takes a PDB file path and a list of forcefield files.
    It creates and returns a PDBFile and ForceField object.
    The forcefield_files list can contain one or more files.
    If only one file is provided, it assumes that the file includes
    both the forcefield and the water model if needed.

    Parameters:
    pdb_path (str): The file path to the PDB file.
    forcefield_files (list of str): A list of file paths to the forcefield XML files.

    Returns:
    tuple: A tuple containing the PDBFile and ForceField objects.
    """

    # Load the PDB file

    pdb_path.split(".")[0]
    end = pdb_path.split(".")[1]
    if end == "pdb":
        pdb = PDBFile(pdb_path)
    elif end == "cif":
        pdb = PDBxFile(pdb_path)

    # Clean up forcefield files list and remove any empty strings
    forcefield_files = (
        forcefield_files.replace("(default)", "").replace(" and ", ",").strip()
    )
    Forcefield_files = [file.strip() for file in forcefield_files.split(",")]
    Forcefield = Forcefield_files[0]
    Water_model = Forcefield_files[1]
    # check if they are part of the list
    if Forcefield not in FORCEFIELD_LIST:
        raise Exception("Forcefield not recognized")
    if Water_model not in FORCEFIELD_LIST:
        raise Exception("Water model not recognized")

    forcefield = ForceField(Forcefield, Water_model)
    # TODO  Not all forcefields require water model

    return pdb, forcefield
