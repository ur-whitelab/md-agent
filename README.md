MD-Agent is a LLM-agent based toolset for Molecular Dynamics.
It's built using Langchain and uses a collection of tools to set up and execute molecular dynamics simulations.


## Installation
```
pip install git+https://github.com/ur-whitelab/md-agent.git
```


## Usage
The first step is to set up your API keys in your environment. An OpenAI key is necessary for this project.
Other tools require API keys, such as paper-qa for literature searches. We recommend setting up the keys in a .env file. You can use the provided .env.example file as a template.
1. Copy the `.env.example` file and rename it to `.env`: `cp .env.example .env`
2. Replace the placeholder values in `.env` with your actual keys


## Developing
To contribute to MD-Agent's development, follow these steps to ensure the pre-commit checks pass:
1. Clone the repository
2. Install the development dependencies: `pip install -r dev-requirements.txt`
3. Configure pre-commit: `pre-commit install`
4. Prior to committing, ensure that pre-commit checks pass: `pre-commit run --all`

Note: If you have already committed and encounter a pre-commit error during a pull request, complete step 4 above.
