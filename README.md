MD-Agent is a LLM-agent based toolset for Molecular Dynamics.
It's built using Langchain and uses a collection of tools to set up and execute molecular dynamics simulations, particularly in OpenMM.


## Environment Setup
To use the OpenMM features in the agent, please set up a conda environment, following these steps.
- Create conda environment: `conda env create -n mdagent -f environment.yaml`
- Activate your environment: `conda activate mdagent`

If you already have a conda environment, you can install dependencies with the following step.
- Install the necessary conda dependencies: `conda install -c conda-forge openmm pdbfixer mdtraj`


## Installation
```
pip install git+https://github.com/ur-whitelab/md-agent.git
```


## Usage
The first step is to set up your API keys in your environment. An OpenAI key is necessary for this project.
Other tools require API keys, such as paper-qa for literature searches. We recommend setting up the keys in a .env file. You can use the provided .env.example file as a template.
1. Copy the `.env.example` file and rename it to `.env`: `cp .env.example .env`
2. Replace the placeholder values in `.env` with your actual keys

## Using Streamlit Interface
If you'd like to use MDAgent via the streamlit app, make sure you have completed the steps above. Then, in your terminal, run `streamlit run st_app.py` in the project root directory.

From there you may upload files to use during the run. Note: the app is currently limited to uploading .pdb and .cif files, and the max size is defaulted at 200MB.
- To upload larger files, instead run `streamlit run st_app.py --server.maxUploadSize=some_large_number`
- To add different file types, you can add your desired file type to the list in the [streamlit app file](https://github.com/ur-whitelab/md-agent/blob/main/st_app.py).


## Contributing

We welcome contributions to MD-Agent! If you're interested in contributing to the project, please check out our [Contributor's Guide](CONTRIBUTING.md) for detailed instructions on getting started, feature development, and the pull request process.

We value and appreciate all contributions to MD-Agent.
