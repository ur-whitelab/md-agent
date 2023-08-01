from mdagent.agents import _make_llm
from mdagent.agents import ActionAgent, CodeCriticAgent, TaskCriticAgent, FirstActionAgent
import json
import sys
import re

class Iterator:
    def __init__(
        self,
        model="gpt-4",
        temp=0.1,
        max_iterations=120,
        api_key=None,
        verbose=True,
    ):
        self.llm = _make_llm(model, temp, max_iterations)
        self.code_history = []
        self.output_history = []
        self.files_history = []
        self.critique_history = []
        self.context = ""
        self.task = ""
        
        """
        This is the iterator, which is the main function the in mdagent. 
        This agent should interact with the following agents:
        - code critic (to critique the code and provide feedback)
        - action (to generate new code based on curriculum or critic feedback)
        - task critic (to critique the task and provide feedback)
        
        This agent should start with the input from curriculum and then cycle 
        through the following steps 5 times maximum:
        1. action (to write code)
        2. **run code** generated from action using python repl
        3. code critic (to critique and suggest changes) 
        
        The final step is to use task critique to assess if the task is completed.
        
        """
        
        #init agents
        self.action_agent = ActionAgent(
            model=model,
            temp=temp,
            max_iterations=max_iterations,
            api_key=api_key,
            verbose=verbose,
        )
        
        self.code_critic_agent = CodeCriticAgent(
            model=model,
            temp=temp,
            max_iterations=max_iterations,
            api_key=api_key,
            verbose=verbose,
        )
        
        self.task_critic_agent = TaskCriticAgent(
            model=model,
            temp=temp,
            max_iterations=max_iterations,
            api_key=api_key,
            verbose=verbose,
        )
        
    def _generate_output_dict(self):
        # Initialize the output dictionary
        output_dict = {}

        # Iterate through the lists and create sub-dictionaries for each entry
        for i in range(5):
            sub_dict = {
                "task": self.task,
                "context": self.context,
                "code": self.code_history[i],
                "output": self.output_history[i],
                "files": self.files_history[i],
                "critique": self.critique_history[i]
            }
            output_dict[f"iter{i+1}"] = sub_dict

        #todo: add human readable version (indent=4) of the output in run history
        return json.dumps(output_dict, indent=None)
    
    #function that runs the code
    def _run_code(self, code):
        #incoming code should be a json string
            # Load the JSON string and extract the Python code
        data = json.loads(code)
        python_code = data["code"]
        
        # Redirect stdout and stderr to capture the output
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = captured_stdout = sys.stderr = sys.StringIO()

        exc = False
        try:
            exec(python_code)
            output = captured_stdout.getvalue()
        except Exception as e:
            exc = True
            output = str(e)
        finally:
            # Restore stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
        return exc, output
    
    def _extract_code(self, output):
        match = re.search(r"Code:\n```\n(.+?)\n```\n", output, re.DOTALL)

        if match:
            # group(1) corresponds to the first group of parentheses in the regex.
            # In this case, it is the section that matches any character (.) any number of times (+),
            # as long as it is between the "Code:\n```\n" and "\n```\n".
            code = match.group(1)
            return code
        else:
            return None
        
    def _write2tool(self, code, name, description):
        code_template = f"""
        class {name}(BaseTool):
        name = "{name}"
        description = "{description}"

        def _run(self, query: str) -> str:
        \"\"\"Use the tool.\"\"\"
            result = self.use_the_tool(query)  # Replace this with the actual function call to use the tool
        return result


        async def _arun(self, query) -> str:
        \"\"\"Use the tool asynchronously.\"\"\"
        raise NotImplementedError("{name} does not support async")
        """

        return code_template
    
    def _run_loop(self, task, context, skills, code=None, code_output=None, files=None, critique=None, history=None):
        """
        this function just runs the loop 
        """
        #implement memory and get list of files
        #todo: implement memory
        
        #first run action
        if history == None: #first iteration
            action_output = self.first_action_agent._run(task, context, files, skills)
        else:
            action_output = self.action_agent._run(code, task, context, code_output, files, critique, history, skills)
        #extract code part:
        code = self._extract_code(action_output)
        #run code
        exc, output = self._run_code(code)
        if exc == True:
            print ("code failed, running code critic")
        else: #todo: give this to task critic, if it fails, then give it back to code and continue
            print ("code succeeded, running code critic")
        critique = self.code_critic_agent._run(code, code_output, task, context)
        return code, output, files, context, task, critique
        
        
    
    def _run(self, task, context, skills, code_output, files, critique):
        #task is from curriculum
        #context is from curriculum
        
        iter = 0
        task = "incomplete"
        history=None
        while iter < 5 and task != "complete":
            if iter > 0:
                history = self._generate_output_dict(self)
            #run loop
            code, output, files, context, task, critique = self._run_loop(task, context, skills, code_output, files, critique, history)
            iter += 1
            #save to history
            self.code_history += code
            self.output_history += output
            self.files_history += files
            self.context = context
            self.task = task
            self.critique_history += critique
            #run task critic to see if task is complete
            
            #todo --> task critic is wonky
            critic = self.task_critic_agent._run(files, task, context)
            if critic == True:
                task = "complete"
                successful_code = code
                
                #add successful code to tool manager 
                final_history = self._generate_output_dict(self)
                #todo: save final_history to file
                
                #and write successful_code to file
                return "task complete, final history saved to file"
            else: 
                continue
        