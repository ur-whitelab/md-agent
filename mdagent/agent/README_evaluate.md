# README for Evaluate.py

## Overview
`evaluate.py` is a Python script designed to facilitate the evaluation of MDAgent. It supports multiple use cases, including automated evaluation across different agent settings, loading previous evaluation results, and creating structured output tables. A list of generated or loaded evaluations can be accessed via `evaluator.evaluations`. Use `evaluator.reset()` to clear the evaluations list.

## Getting Started
To use `evaluate.py`, ensure that `mdagent` package is installed in your Python environment (see main README for installation instructions).

## Usage Examples

### Example 1: Evaluate Prompts with default MD-Agent Parameters
Evaluate specific prompts with default settings:
```python
from mdagent import Evaluator

evaluator = Evaluator()
prompts = [
    'Download and clean fibronectin.',
    'Simulate 1A3N in water for 100 ns.',
]
df = evaluator.automate(prompts)
df # this displays DataFrame table in Jupyter notebook
```
This will run MDAgent and evaluate the prompts using the default settings. The results will be
saved to json file in the "evaluation_results" directory and used to create pandas DataFrame.

### Example 2: Evaluate Prompts with specific MD-Agent Parameters
Evaluate specific prompts using single agent instance with specified parameters:
```python
from mdagent import Evaluator

evaluator = Evaluator()
agent_params = {
    "agent_type": "Structured",
    'model': 'gpt-3.5-turbo',
    'tools_model': 'gpt-3.5-turbo',
}
prompts = [
    'Download and clean fibronectin.',
    'Simulate 1A3N in water for 100 ns.',
]
df = evaluator.automate(prompts, agent_params=agent_params)
df_full = evaluator.create_table(simple=False) # to get a table with all details
```

### Example 3: Evaluate Prompts with Multiple Agent Parameters
Evaluate specific prompts using multiple agent settings with `automate_all` method:
```python
from mdagent import Evaluator

evaluator = Evaluator()
prompts = [
    'Download and clean fibronectin.',
    'Simulate 1A3N in water for 100 ns.',
]
agent_params_list = [
    {
        "agent_type": "OpenAIFunctionsAgent",
        "model": "gpt-4-1106-preview",
        "ckpt_dir": "ckpt_openaifxn_gpt4",
    },
    {
        "agent_type": "Structured",
        "model": "gpt-3.5-turbo",
        "ckpt_dir": "ckpt_structured_gpt3.5",
    },
]
df = evaluator.automate_all(prompts, agent_params_list=agent_params_list)
```

### Example 4: Load Previous Evaluation Results and Create a Table
Load previous evaluation results from a JSON file:
```python
from mdagent import Evaluator

evaluator = Evaluator()
evaluator.load('evaluation_results/mega_eval_20240422-181241.json')
df = evaluator.create_table()
df.to_latex('evaluation_results/eval_table.tex') # Optional: save table to a LaTeX file
```
You can load multiple evaluation files by calling `evaluator.load()` multiple times. The results will be appended to the `evaluator.evaluations` list.

### Example 5: Make Multi-Prompt Query with Agent Memory
Use agent memory to link multiple prompts:
```python
from mdagent import Evaluator

evaluator = Evaluator()
agent_params1 = {
    "use_memory": True
}
prompt_set1 = ['Simulate 1A3N in water for 100 ns.']
df1 = evaluator.automate(prompt_set1, agent_params1)
df1 # display the table containing run ID

agent_params2 = {
    "use_memory": True,
    "run_id": "U4831GA3", # <---- insert run_id from prompt 1 table
}
prompt_set2 = ['Calculate RMSD for 1A3N simulation over time.']
df2 = evaluator.automate(prompt_set2, agent_params2)
df2 # display results from both prompts
```
Another way to get run_id: you can access `evaluator.evaluations` list and pull the key
`run_id` from the dictionary that contains the results of the first prompt.

## Additional Information
- For whatever reason, instead of `evaluator.automate()`, you can manually call `evaluator.run_and_evaluate(prompts, agent_params=params)` once or several times, `evaluator.save()` to save all evaluations to a json file, then use `evaluator.create_table()` to get DataFrame object.
- `evaluate.py` is designed to be used in a Jupyter notebook environment.
