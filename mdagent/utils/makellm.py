from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def _make_llm(model, temp, verbose):
    from langchain_openai import ChatOpenAI

    if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        llm = ChatOpenAI(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=True if verbose else False,
            callbacks=[StreamingStdOutCallbackHandler()] if verbose else None,
        )
    elif model.startswith("accounts/fireworks"):
        from langchain_fireworks import ChatFireworks

        llm = ChatFireworks(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=True if verbose else False,
            callbacks=[StreamingStdOutCallbackHandler()] if verbose else None,
        )
    elif model.startswith("together/"):
        # user needs to add 'together/' prefix to use TogetherAI provider
        from langchain_together import ChatTogether

        llm = ChatTogether(
            temperature=temp,
            model=model.replace("together/", ""),
            request_timeout=1000,
            streaming=True if verbose else False,
            callbacks=[StreamingStdOutCallbackHandler()] if verbose else None,
        )
    else:
        raise ValueError(f"Unrecognized or Unsupported model name: {model}")
    return llm
