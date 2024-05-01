from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain.vectorstores import FAISS

examples = [
    {
        "input": "Calculate rdf of the protein with water given the simulation",
        "output": (
            "\nfrom MDAnalysis.analysis.rdf import InterRDF\n\n"
            "def calculate_rdf(topology_file,"
            "trajectory_file, protein_id, selection='protein and resname HOH'):\n"
            "   #The topology file is the one saved just before the simulation\n"
            "   #The trajectory file is the one saved during the simulation\n"
            "   #The protein_id is the unique identifier of the protein analyzed\n"
            "   #The selection is the selection string to select the atoms"
            " of interest\n\n"
            "   traj_path = path_registry.get_path(trajectory_file)\n"
            "   top_path = path_registry.get_path(topology_file)\n"
            "   u = mda.Universe(top_path, traj_path)\n"
            "   selection1 = u.select_atoms(selection.split('and')[0])\n"
            "   selection2 = u.select_atoms(selection.split('and')[1])\n"
            "   rdf = InterRDF(selection1, selection2, nbins=75,"
            "range=(0.0, 15.0))\n"
            "   rdf.run()\n"
            "   # Check if there are water molecules in the trajectory\n"
            "   if len(selection2) == 0:\n"
            "    raise ValueError(f'No {{selection.split('and')[1]}}"
            "    found in the trajectory. Cannot calculate RDF.')]\n"
            "   # Save the RDF data to a CSV file\n"
            "   rdf_data = pd.DataFrame({{'Distance (Angstrom)': rdf.results.bins, "
            "RDF': rdf.results.rdf}})\n"
            "   ...continue to save rdf_data..."
        ),
    },
    {
        "input": "Calculate rdf of the protein with water given the simulation",
        "output": (
            "\nimport mdtraj as md\n\n"
            "def calculate_rdf(topology_file, trajectory_file, protein_id,"
            " selection='protein and resname HOH'):\n"
            "    #The topology file is the one saved just before the simulation\n"
            "    #The trajectory file is the one saved during the simulation\n"
            "    #The protein_id is the unique identifier of the protein analyzed\n"
            "    #The selection is the selection string to select the atoms"
            "of interest\n\n"
            "    traj_path = path_registry.get_path(trajectory_file)\n"
            "    top_path = path_registry.get_path(topology_file)\n"
            "    traj = md.load(traj_path, top=top_path)\n"
            "    protein = traj.top.select(selection.split('and')[0])\n"
            "    water = traj.top.select(selection.split('and')[1])\n"
            "    r, g_r = = md.compute_rdf(protein, water,"
            " nbins=75, range=(0.0, 15.0))\n"
            "    # Check if there are water molecules in the trajectory\n"
            "    if len(water) == 0:\n"
            "        raise ValueError(f'No {{selection.split('and')[1]}}"
            "found in the trajectory. Cannot calculate RDF.')\n"
            "   rdf_data = '\\n'.join(f'{{distance}} {{value}}' for distance,"
            "value in zip(r, g_r))\n"
            "   ...continue to save rdf_data..."
        ),
    },
    {
        "input": (
            "Calculate radius of gyration of the protein with "
            "water given the simulation"
        ),
        "output": (
            ""
            "from MDAnalysis.analysis.rms import RMSD\n"
            "import pandas as pd\n"
            "from mdagent.utils import path_registry \n\n"
            "def calculate_radius_of_gyration(topology_file_id, trajectory_file_"
            "id, protein_id, selection='protein'):\n"
            "\t```\n"
            "\tCalculate the radius of gyration for a "
            "given protein over a trajectory.\n"
            ""
            "\tParameters:"
            "\ttopology_file : str\n"
            "\t    The file containing the molecular topology.\n"
            "\ttrajectory_file : str\n"
            "\t    The file containing the molecular dynamics trajectory.\n"
            "\tprotein_id : str\n"
            "\t    The unique identifier of the protein.\n"
            "\tselection : str, optional\n"
            "\t    The selection string to select the atoms of"
            " interest (default is 'protein').\n"
            "\tReturns:\n"
            "\trgyr_data : DataFrame\n"
            "\t    DataFrame containing the time series of the radius of gyration.\n"
            "\t```\n"
            "\ttop_path = path_registry.get_path(topology_file)\n"
            "\ttraj_path = path_registry.get_path(trajectory_file)\n"
            "\t# Load the trajectory and topology files\n"
            "\tu = mda.Universe(topology_file, trajectory_file)\n"
            "\t# Select the atoms based on the provided selection string\n"
            "\tselected_atoms = u.select_atoms(selection)\n"
            "\t# Instantiate the Rgyr class with the selected atoms\n"
            "\trgyr = Rgyr(selected_atoms)\n"
            "\trgyr.run()\n"
            "\trgyr_data = pd.DataFrame({\n"
            "\t    'Time (ps)': u.trajectory.times,\n"
            "\t    'Radius of Gyration (Å)': rgyr.results.rg\n"
            "\t})\n"
            "\t...continue to save rgyr_data...\n"
        ),
    },
]
to_vectorize = [" ".join(example.values()) for example in examples]
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(to_vectorize, embedding=embeddings, metadatas=examples)

example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vector_store,
    k=2,
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    # The input variables select the values to pass to the example_selector
    input_variables=["input"],
    example_selector=example_selector,
    # Define how each example will be formatted.
    # In this case, each example will become 2 messages:
    # 1 human, and 1 AI
    example_prompt=ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    ),
)
