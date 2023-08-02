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
        
        
        
        #init agents
        self.action_agent = ActionAgent(
            model=model,
            temp=temp,
            max_iterations=max_iterations,
            api_key=api_key,
            verbose=verbose,
        )
        
        self.first_action_agent = FirstActionAgent(
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
        
    def _add_to_history(self, existing_history, iter, task, context, code_history, output_history, files_history, critique_history, task_critique_history):
        # Initialize the output dictionary
        if existing_history is None:
            existing_history = []

        # Initialize the output dictionary
        output_dict = {
            "iteration": iter,
            "task": task,
            "context": context,
            "code": code_history,
            "output": output_history,
            "files": files_history,
            "code critique": critique_history,
            "task_critique": task_critique_history
        }
        # Append to the existing history
        output_json_string = json.dumps(output_dict, indent=4)
        existing_history.append(output_json_string)
        return existing_history
    
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
        
    def _save_failures(self, history, msg):
        if msg == None:
            #save to file
            with open("failed_history.json", "a") as f:
                f.write("\n", history, "\n")
            return "failed history saved to file"
        else: 
            #save to file
            with open("failed_history.json", "a") as f:
                f.write("\n", msg, "\n")
            return None
    
    #this function doesnt work but should be in skill library probably
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
    
    def _run_task_critic(self, files, code, code_output, task, context, msg=None): 
        
        #this helper function helps parse the output from task critic
        def _parse_critic_output(critic_output):
            parsed_output = json.loads(critic_output)
            # Extract the success boolean
            success = parsed_output.get("success", None)
                # Check if it's a boolean value first
            if isinstance(success, bool):
                return success, parsed_output.get("critique", None)

            # If not a boolean, make it boolean
            if isinstance(success, str):
                success_lower = success.lower()
                if "true" in success_lower:
                    success = True
                elif "false" in success_lower:
                    success = False
                else:
                    raise ValueError(f"Invalid success value: {success}")
                return success, parsed_output.get("critique", None) 
            
        #running task critic, doing this in a loop to handle an error one time
        for _ in range(2):  # Two attempts
            try:
                full_out = self.task_critic_agent._run(files, code, code_output, task, context, msg)
                success, task_critique = _parse_critic_output(full_out)
                break # Break out of the loop if successful
            except ValueError: # Raised if the success value is invalid
                msg = "Please ensure that your output matches the formatting requirements." #try again with additional message
        else:  # Executed if the loop completes without a 'break', i.e. if both attempts failed
            success = False
            task_critique = None
        return success, task_critique
        
    
    def _run_action(self, recent_history, full_history, skills, task, context):
        if recent_history is None:
            #get files --> memory, this should be easy
            files = None
            action_output = self.first_action_agent._run(files, task, context, skills)
        else:
            action_output = self.action_agent._run(recent_history, full_history, skills)
        return action_output
    
    def _run_loop(self, task, context, skills, recent_history, full_history):
        """
        this function just runs the iteration 1 time
        """
        #implement memory and get list of files
        #todo: implement memory
        
        #get files
        files=None
        
        action_output = self._run_action(recent_history, full_history, skills, task, context)
        #extract code part
        code = self._extract_code(action_output)
        #run code
        failed, output = self._run_code(code)
        if failed == True:
            task_critique = None
            success = False
        else: 
            print ("code succeeded, running task critic")
            #run task critic
            success, task_critique = self._run_task_critic(files, code, output, task, context, msg=None)
            
            #check if task is complete
            if success == True:
                print ("task complete")
                return success, code, output, files, context, task, critique
            #otherwise, run code critic
        critique = self.code_critic_agent._run(code, output, task, context)
        return success, code, output, files, context, task, critique, task_critique
    
    
    def _run_iteration(self, task, context, skills, files):
        
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
        
        #task is from curriculum
        #context is from curriculum
        
        iter = 0
        success = False
        skill = False
        while iter < 5 and success == False:
            if iter == 0:
                success, code, output, files, context, task, critique, task_critique = self._run_loop(task, context, skills, None, None)
            if iter > 0:
                full_history = self._add_to_history(recent_history, iter, task, context, code, output, files, critique, task_critique)
                recent_history = full_history[-1]
                success, code, output, files, context, task, critique, task_critique = self._run_loop(task, context, skills, recent_history, full_history)
            iter += 1
            #save to history
            if success:
                #update variables and save to file
                self._save_failures(full_history, None)
                successful_code = code
                skill = True
                #give successful code to tool manager
                return skill
        #if max iterations reached without success, save failures to file
        print ("max iterations reached, saving failed history to file")
        full_failed = self._add_to_history(full_history, iter, task, context, code, output, files, critique, task_critique)
        self._save_failures(full_failed, None)
        return skill, full_failed #give to curriculum
    
    def main_run(self):

        N = 0 #goal number of learned skills 
        M = 0 #max iterations
        while N < 10 and M < 20:
            #run curriculum to get task and context
            task = None
            context = None
            files = None
            skills = None
            #get files --> memory, this should be easy
            #get skills
            self._save_failures(None, f"Run {M}")
            skill, output = self._run_iteration(self, task=task, context=context, skills=skills, files=files)
            if not skill:
                msg_curr = "max iterations reached, please adjust task or context and try again"
        #return number of skills successfully learned and tasks addressed        
        return None
        