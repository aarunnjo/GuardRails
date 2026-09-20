"""LangChain adapter -- wraps any BaseRetriever so every retrieval passes
through Firewall.scan() before reaching the prompt.

Optional dependency: langchain-core is imported lazily, inside
FireguardRetriever.__init__, not at module import time -- `import fireguard`
never requires it. Install with `pip install fireguard[langchain]`.
"""
from ..firewall import Firewall
from ..types import Chunk, RetrievalSet


def _tier_for(metadata: dict) -> str:
    """LangChain Documents carry a free-form metadata dict. fireguard reads
    metadata['tier'] if the caller set it, else falls back to 'open_web' --
    the safer default for content of unknown provenance."""
    return metadata.get("tier", "open_web")


def _source_for(metadata: dict, fallback: str) -> str:
    return metadata.get("source", fallback)


class FireguardRetriever:
    """Drop-in wrapper: same `.invoke()` surface as the retriever it wraps,
    but the returned documents have already passed fw.scan(). Flagged/
    blocked documents are silently dropped, not raised -- inspect
    `.last_scan` if the caller needs the verdict and reasons."""

    def __init__(self, retriever, firewall: Firewall | None = None):
        try:
            import langchain_core.documents  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "FireguardRetriever needs langchain-core: pip install fireguard[langchain]"
            ) from e
        self._retriever = retriever
        self._fw = firewall or Firewall()
        self.last_scan = None

    def _scan(self, query: str, documents: list) -> list:
        chunks = [
            Chunk(
                text=d.page_content,
                source_uri=_source_for(d.metadata, f"doc-{i}"),
                tier=_tier_for(d.metadata),
            )
            for i, d in enumerate(documents)
        ]
        result = self._fw.scan(RetrievalSet(query=query, chunks=chunks))
        self.last_scan = result

        approved_uris = [c.source_uri for c in result.approved_chunks]
        by_uri: dict[str, list] = {}
        for i, d in enumerate(documents):
            by_uri.setdefault(_source_for(d.metadata, f"doc-{i}"), []).append(d)

        approved_docs = []
        for uri in approved_uris:
            if by_uri.get(uri):
                approved_docs.append(by_uri[uri].pop(0))
        return approved_docs

    def invoke(self, query: str, **kwargs) -> list:
        documents = self._retriever.invoke(query, **kwargs)
        return self._scan(query, documents)

    def get_relevant_documents(self, query: str, **kwargs) -> list:
        documents = self._retriever.get_relevant_documents(query, **kwargs)
        return self._scan(query, documents)
