"""Local extraction: retrieved passages -> Claims with verbatim quotes -> Briefing.

The compression step the architecture is built around (73-100k tokens of filings -> a
2-5k token briefing), arranged so that the local model's output is checkable without
re-reading the filings. See `quotes.py` for the mechanism that makes that true.
"""

from argus.engine.extract.claims import ExtractionStats, parse_claims
from argus.engine.extract.export import to_markdown
from argus.engine.extract.loop import ExtractConfig, ExtractionLoop, ExtractionResult
from argus.engine.extract.quotes import QuoteStatus, resolve

__all__ = ["ExtractionStats", "parse_claims", "to_markdown", "ExtractConfig",
           "ExtractionLoop", "ExtractionResult", "QuoteStatus", "resolve"]
