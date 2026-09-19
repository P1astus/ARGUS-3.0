"""Corpus retrieval: chunk, index, search.

The local model summarises what retrieval finds. It does not recall -- §9 of the handoff
is explicit that CPT at 103M tokens will not install reliable factual recall, so anything
the model asserts about a filing has to have come from a passage put in front of it.
That makes this package, not the model, the thing that determines what a briefing can
say.
"""

from argus.engine.retrieval.chunk import Chunk, ChunkSpec, chunk_document
from argus.engine.retrieval.search import Hit, SearchConfig, Searcher
from argus.engine.retrieval.store import ChunkRef, ChunkStore

__all__ = ["Chunk", "ChunkSpec", "chunk_document", "ChunkRef", "ChunkStore",
           "Hit", "SearchConfig", "Searcher"]
