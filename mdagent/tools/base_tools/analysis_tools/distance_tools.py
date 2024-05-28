from typing import Optional

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from mdagent.utils import PathRegistry

from .descriptions import (
    CUTOFF_DESC,
    DISPLACEMENT_TOOL_DESC,
    DISTANCE_TOOL_DESC,
    SELECTION_DESC,
    TOPOLOGY_FILEID_DESC,
    TRAJECTORY_FILEID_DESC,
)


class distanceUtils:
    pass


class distanceSchema(BaseModel):
    trajectory_fileid: str = Field(description=TRAJECTORY_FILEID_DESC)
    topology_fileid: str = Field(description=TOPOLOGY_FILEID_DESC)
    selection1: str = Field(description="First" + SELECTION_DESC)
    selection2: str = Field(description="Second" + SELECTION_DESC)


class displacementSchema(BaseModel):
    trajectory_fileid: str = Field(description=TRAJECTORY_FILEID_DESC)
    topology_fileid: str = Field(description=TOPOLOGY_FILEID_DESC)
    selection1: str = Field(description="First" + SELECTION_DESC)
    selection2: str = Field(description="Second" + SELECTION_DESC)


class neighborsSchema(BaseModel):
    trajectory_fileid: str = Field(description=TRAJECTORY_FILEID_DESC)
    topology_fileid: str = Field(description=TOPOLOGY_FILEID_DESC)
    selection1: str = Field(description="First" + SELECTION_DESC)
    selection2: str = Field(description="Second" + SELECTION_DESC)
    cutoff: float = Field(10.0, description=CUTOFF_DESC)


class contactSchema(BaseModel):
    trajectory_fileid: str = Field(description=TRAJECTORY_FILEID_DESC)
    topology_fileid: str = Field(description=TOPOLOGY_FILEID_DESC)
    selection1: str = Field(description="First" + SELECTION_DESC)
    selection2: str = Field(description="Second" + SELECTION_DESC)
    cutoff: float = Field(6.0, description=CUTOFF_DESC)


class distanceTool(BaseTool):
    name = "distanceTool"
    description = DISTANCE_TOOL_DESC
    input_schema = distanceSchema
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry] = None):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, **input):
        input = self.validate_input(input)
        pass

    def _arun(self):
        pass

    def validate_inputs(self, input):
        input = input.get("action_input", input)
        input = input.get("input", input)
        trajectory_id = input.get("trajectory_fileid", None)
        topology_id = input.get("topology_fileid", None)
        if not trajectory_id:
            raise ValueError("Incorrect Inputs: trajectory_fileid is required")
        if not topology_id:
            raise ValueError("Incorrect Inputs: topology_fileid is required")
        fileids = self.path_registry.list_path_names()
        error = ""
        if trajectory_id not in fileids:
            error += " Trajectory File ID not in path registry"
        if topology_id not in fileids:
            error += " Topology File ID not in path registry"


class displacementTool(BaseTool):
    name = "displacementTool"
    description = DISPLACEMENT_TOOL_DESC
    input_schema = displacementSchema
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry] = None):
        super().__init__()
        self.path_registry = path_registry

    def _run(self):
        pass

    def _arun(self):
        pass

    def validate_inputs(self, input):
        input = input.get("action_input", input)
        input = input.get("input", input)
        trajectory_id = input.get("trajectory_fileid", None)
        topology_id = input.get("topology_fileid", None)
        if not trajectory_id:
            raise ValueError("Incorrect Inputs: trajectory_fileid is required")
        if not topology_id:
            raise ValueError("Incorrect Inputs: topology_fileid is required")
        fileids = self.path_registry.list_path_names()
        error = ""
        if trajectory_id not in fileids:
            error += " Trajectory File ID not in path registry"
        if topology_id not in fileids:
            error += " Topology File ID not in path registry"


class neighborsTool(BaseTool):
    name = "neighborsTool"
    description = (
        "Tool for calculating number of neighbors"
        " on an atom selection in a trajectory."
    )
    input_schema = neighborsSchema
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry] = None):
        super().__init__()
        self.path_registry = path_registry

    def _run(self):
        pass

    def _arun(self):
        pass

    def validate_inputs(self, input):
        input = input.get("action_input", input)
        input = input.get("input", input)
        trajectory_id = input.get("trajectory_fileid", None)
        topology_id = input.get("topology_fileid", None)
        if not trajectory_id:
            raise ValueError("Incorrect Inputs: trajectory_fileid is required")
        if not topology_id:
            raise ValueError("Incorrect Inputs: topology_fileid is required")


class contactsTool(BaseTool):
    name = "NeighborsTool"
    description = (
        "Tool for calculating number of contacts"
        " between pairs of residues in a trajectory."
    )
    input_schema = neighborsSchema
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry] = None):
        super().__init__()
        self.path_registry = path_registry

    def _run(self):
        pass

    def _arun(self):
        pass

    def validate_inputs(self, input):
        input = input.get("action_input", input)
        input = input.get("input", input)
        trajectory_id = input.get("trajectory_fileid", None)
        topology_id = input.get("topology_fileid", None)
        if not trajectory_id:
            raise ValueError("Incorrect Inputs: trajectory_fileid is required")
        if not topology_id:
            raise ValueError("Incorrect Inputs: topology_fileid is required")
        fileids = self.path_registry.list_path_names()
        error = ""
        if trajectory_id not in fileids:
            error += " Trajectory File ID not in path registry"
        if topology_id not in fileids:
            error += " Topology File ID not in path registry"
