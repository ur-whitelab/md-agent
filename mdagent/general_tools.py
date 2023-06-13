from typing import Any


def dummy_function() -> int:
    return 46


def partial(func: Any, *args: Any, **kwargs: Any) -> Any:
    """
    This function is a workaround for the partial
    function error in newer langchain versions.
    This can be removed if not needed.
    """

    def wrapped(*args_wrapped: Any, **kwargs_wrapped: Any) -> Any:
        final_args = args + args_wrapped
        final_kwargs = {**kwargs, **kwargs_wrapped}
        return func(*final_args, **final_kwargs)

    return wrapped


class Scholar2ResultLLM:
    def __init__(self, pqa_key: str):
        self.pqa_key = pqa_key

    def query(self, question: str) -> str:
        import pqapi

        response = pqapi.agent_query("default", question)
        return response.answer
