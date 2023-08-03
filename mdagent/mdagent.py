import agents
from . import PathRegistry
from typing import Optional
    
class MDAgent:    
    def __init__(
        self,
        path_registry: Optional[PathRegistry],
        model="gpt-4",
        temp=0.1,
        max_iterations=120,
        api_key=None,
        verbose=True,
    ):
        self.path_registry = path_registry
        
        #init agents
        self.iterator = agents.Iterator(
            path_registry=path_registry,
            model=model,
            temp=temp,
            max_iterations=max_iterations,
            api_key=api_key,
            verbose=verbose,
        )
    
    def main_run(self):
        N = 0 #goal number of learned skills 
        M = 0 #max iterations
        while N < 10 and M < 20:
            #run curriculum to get task and context
            task = None
            context = None
            skills = None
            #get files --> memory, this should be easy
            #get skills
            self._save_failures(None, f"Run {M}")
            skill, output = self.iterator._run_iteration(self, task=task, context=context, skills=skills)
            if not skill:
                msg_curr = "max iterations reached, please adjust task or context and try again"
        #return number of skills successfully learned and tasks addressed    
        
        #add failed iteration file to path registry with description
        self.path_registry.map_path("failed_iterations", "failed_history.json", "Entries from all failed iterations")    
        return None