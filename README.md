MD-Agent is a LLM-agent based toolset for Molecular Dynamics.
It's built using Langchain and uses a collection of tools to set up and execute molecular dynamics simulations.


## Installation
```
conda create -n mdagent python=3.10
conda activate mdagent
pip install -e .
```


# Usage
The first step is to set up your API keys in your environment. An OpenAI key is necessary for this project.
Other tools require API keys, such as:
- SERP (for web searches)
- Semantic Scholar and paper-qa (for literature searches)


## Developing
To contribute to MD-Agent's development, follow these steps to ensure the pre-commit checks pass:
1. Clone the repository
2. Install the development dependencies: `pip install -r dev-requirements.txt`
3. Configure pre-commit: `pre-commit install`
4. Prior to committing, ensure that pre-commit checks pass: `pre-commit run --all`

Note: If you have already committed and encounter a pre-commit error during a pull request, complete step 4 above.
