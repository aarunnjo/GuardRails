"""LlamaIndex adapter -- wraps any BaseRetriever so every retrieval passes
through Firewall.scan() before reaching the prompt.

Optional dependency: llama-index-core is imported lazily, inside
FireguardRetriever.__init__. Install with `pip install fireguard[llamaindex]`.
"""
from ..firewall import Firewall
from ..types import Chunk, RetrievalSet


class FireguardRetriever:
    """Drop-in wrapper around a llama_index BaseRetriever. `.retrieve()`
    returns the same NodeWithScore list, minus anything fw.scan() flagged.
    Inspect `.last_scan` for the verdict and reasons."""

    def __init__(self, retriever, firewall: Firewall | None = None):
        try:
            import llama_index.core.schema  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "FireguardRetriever needs llama-index-core: pip install fireguard[llamaindex]"
            ) from e
        self._retriever = retriever
        self._fw = firewall or Firewall()
        self.last_scan = None

    def retrieve(self, query) -> list:
        query_str = query.query_str if hasattr(query, "query_str") else str(query)
        nodes = self._retriever.retrieve(query)

        chunks = [
            Chunk(
                text=n.node.get_content(),
                source_uri=n.node.node_id,
                tier=n.node.metadata.get("tier", "open_web"),
            )
            for n in nodes
        ]
        result = self._fw.scan(RetrievalSet(query=query_str, chunks=chunks))
        self.last_scan = result

        approved_ids = {c.source_uri for c in result.approved_chunks}
        return [n for n in nodes if n.node.node_id in approved_ids]
