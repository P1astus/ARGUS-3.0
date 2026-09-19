"""Near-duplicate removal, partitioned by issuer.

THE SCALING PROBLEM AND ITS DISSOLUTION
Global MinHash/LSH over ~2-4M paragraphs is the genuinely expensive stage: signatures
alone run ~4GB at 128 permutations, before band buckets.

But near-duplication in filings is almost entirely WITHIN an issuer ACROSS TIME -- risk
factors copied near-verbatim year over year, quarterly MD&A reusing structure. Partitioning
by CIK turns one large problem into ~100 small independent ones, each trivially in-memory
and embarrassingly parallel. A single cheap global exact-hash pass then catches
cross-issuer boilerplate (shared law-firm language).

WHY OUR DEDUP RATE WILL EXCEED THE PUBLISHED ONE
The reference study removed only 1.9% of tokens, but its corpus was wide and shallow
(1,000 companies x 10 years). Ours is narrow and deep (~100 companies x 12-16 years), so
per-issuer repetition is roughly an order of magnitude higher. Expect 25-40%.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from dataclasses import dataclass

log = logging.getLogger(__name__)

_WORD = re.compile(r"\w+")


def normalize_for_hashing(text: str) -> str:
    """Aggressive normalisation so trivial edits do not defeat matching.

    Boilerplate is often re-filed with only a date or dollar figure changed; without
    collapsing digits, each year's risk factors would look distinct.
    """
    t = text.lower()
    t = re.sub(r"\d+", "0", t)          # dates and amounts are the usual only change
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def shingles(text: str, k: int = 9) -> set[str]:
    """k-word shingles, hashed to 64-bit ints for compactness."""
    words = _WORD.findall(normalize_for_hashing(text))
    if len(words) < k:
        return set()
    return {
        hashlib.blake2b(" ".join(words[i:i + k]).encode(), digest_size=8).hexdigest()
        for i in range(len(words) - k + 1)
    }


class MinHasher:
    """MinHash signatures without external dependencies."""

    def __init__(self, num_perm: int = 128, seed: int = 0) -> None:
        self.num_perm = num_perm
        self.seed = seed

    def signature(self, sh: set[str]) -> tuple[int, ...]:
        if not sh:
            return tuple()
        ints = [int(s, 16) for s in sh]
        sig = []
        for i in range(self.num_perm):
            # Cheap universal hashing: mixing the permutation index into each value is
            # sufficient for near-duplicate detection and avoids materialising 128
            # independent hash functions.
            salt = (i * 0x9E3779B97F4A7C15) ^ self.seed
            sig.append(min(((v ^ salt) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
                           for v in ints))
        return tuple(sig)

    @staticmethod
    def similarity(a: tuple[int, ...], b: tuple[int, ...]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        return sum(1 for x, y in zip(a, b) if x == y) / len(a)


@dataclass
class DedupResult:
    keep: list[str]
    drop: dict[str, str]          # doc_id -> representative it duplicates
    clusters: dict[str, str]      # doc_id -> cluster id
    n_input: int

    @property
    def drop_rate(self) -> float:
        return len(self.drop) / max(self.n_input, 1)


def dedupe_partition(
    docs: dict[str, str],
    threshold: float = 0.8,
    num_perm: int = 128,
    shingle_k: int = 9,
) -> DedupResult:
    """Deduplicate one issuer's documents.

    Documents are processed oldest-first by id order so the retained representative is
    stable across runs -- important for resumability, since a re-run must not shuffle
    which copy survives.
    """
    hasher = MinHasher(num_perm=num_perm)
    signatures = {doc_id: hasher.signature(shingles(text, shingle_k))
                  for doc_id, text in docs.items()}

    # Banding: candidates share at least one identical band, which avoids O(n^2)
    # comparisons within the partition.
    bands = 16
    rows = max(num_perm // bands, 1)
    buckets: dict[tuple, list[str]] = defaultdict(list)
    for doc_id, sig in signatures.items():
        if not sig:
            continue
        for b in range(bands):
            key = (b,) + sig[b * rows:(b + 1) * rows]
            buckets[key].append(doc_id)

    keep: list[str] = []
    drop: dict[str, str] = {}
    clusters: dict[str, str] = {}

    for doc_id in sorted(docs):
        if doc_id in drop:
            continue
        sig = signatures.get(doc_id)
        if not sig:
            keep.append(doc_id)
            clusters[doc_id] = doc_id
            continue

        candidates: set[str] = set()
        for b in range(bands):
            candidates.update(buckets.get((b,) + sig[b * rows:(b + 1) * rows], []))
        candidates.discard(doc_id)

        keep.append(doc_id)
        clusters[doc_id] = doc_id
        for other in candidates:
            if other in drop or other in keep:
                continue
            if MinHasher.similarity(sig, signatures[other]) >= threshold:
                drop[other] = doc_id
                clusters[other] = doc_id

    return DedupResult(keep=keep, drop=drop, clusters=clusters, n_input=len(docs))


def exact_hash_pass(docs: dict[str, str]) -> dict[str, str]:
    """Global exact-duplicate pass across issuers.

    Cheap (one hash per document) and catches shared boilerplate that per-issuer
    partitioning cannot see, such as identical legal language filed by multiple companies
    using the same counsel.
    """
    seen: dict[str, str] = {}
    drop: dict[str, str] = {}
    for doc_id in sorted(docs):
        h = hashlib.sha256(normalize_for_hashing(docs[doc_id]).encode()).hexdigest()
        if h in seen:
            drop[doc_id] = seen[h]
        else:
            seen[h] = doc_id
    return drop
