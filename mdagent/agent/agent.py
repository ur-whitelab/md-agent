class MDAgent:
    def __init__(
        self,
        tools,
        model="text-davinci-003",
        temp=0.1,
        max_iterations=40,
        api_key=None,
    ):
        import langchain
        from rmrkl import ChatZeroShotAgent, RetryAgentExecutor
        if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
            self.llm = langchain.chat_models.ChatOpenAI(
                temperature=temp,
                model_name=model,
                request_timeout=1000,
                max_tokens=2000,
            )
        elif model.startswith("text-"):
            self.llm = langchain.OpenAI(temperature=temp, model_name=model)
        elif model.startswith("claude"):
            self.llm = langchain.llms.Anthropic(
                temperature=temp,
                anthropic_api_key=api_key,
                model=model,
            )

        # Initialize agent
        self.agent_executor = RetryAgentExecutor.from_agent_and_tools(
            tools=tools,
            agent=ChatZeroShotAgent.from_llm_and_tools(self.llm, tools),
            verbose=True,
            max_iterations=max_iterations,
            return_intermediate_steps=True,
        )

    def run(self, prompt):
        outputs = self.agent_executor({"input": prompt})
        # Parse long output (with intermediate steps)
        intermed = outputs["intermediate_steps"]

        final = ""
        for step in intermed:
            final += f"Thought: {step[0].log}\n" f"Observation: {step[1]}\n"
        final += f"Final Answer: {outputs['output']}"

        return final
