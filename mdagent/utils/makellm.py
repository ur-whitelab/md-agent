from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether


def _make_llm(model, temp, verbose):
    if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        llm = ChatOpenAI(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=True if verbose else False,
            callbacks=[StreamingStdOutCallbackHandler()] if verbose else None,
        )
    elif model.startswith("Meta-Llama"):
        llm = ChatTogether(
            temperature=temp,
            model=f"meta-llama/{model}",
            request_timeout=1000,
            streaming=True if verbose else False,
            callbacks=[StreamingStdOutCallbackHandler()] if verbose else None,
        )
    else:
        raise ValueError(f"Invalid or Unsupported model name: {model}")
    return llm
