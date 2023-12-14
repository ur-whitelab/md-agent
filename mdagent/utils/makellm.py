import langchain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def _make_llm(model, temp, verbose):
    if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        llm = langchain.chat_models.ChatOpenAI(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=True if verbose else False,
            callbacks=[StreamingStdOutCallbackHandler()] if verbose else None,
        )
    elif model.startswith("text-"):
        llm = langchain.OpenAI(
            temperature=temp,
            model_name=model,
            streaming=True if verbose else False,
            callbacks=[StreamingStdOutCallbackHandler()] if verbose else None,
        )
    else:
        raise ValueError(f"Invalid model name: {model}")
    return llm
