"""Framework adapters. None of these are imported here -- each pulls in an
optional third-party dependency (langchain-core, llama-index-core,
nemoguardrails) that most users of the core library don't have installed.
Import the one you need directly:

    from fireguard.adapters.langchain import FireguardRetriever
    from fireguard.adapters.llamaindex import FireguardRetriever
    from fireguard.adapters import nemo
"""
