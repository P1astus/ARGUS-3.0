"""Store and search behaviour, on a small synthetic index.

`test_as_of_excludes_future_documents` is the one that must never regress. `as_of` is a
leakage boundary: if a passage published after the decision date can be scored at all,
every downstream measurement in this project is contaminated in a way that looks like
skill.
"""

from __future__ import annotations

from datetime import date

import pytest

from argus.engine.retrieval.chunk import ChunkSpec, chunk_document
from argus.engine.retrieval.search import SearchConfig, Searcher, fts_query
from argus.engine.retrieval.store import ChunkStore

SPEC = ChunkSpec(target=400, overlap=80, min_chars=50, snap_window=60)

DOCS = [
    # doc_id, ticker, date, source_type, section, text
    ("d_amat_10q", "AMAT", "2019-08-15", "10-Q", "mdna",
     "Semiconductor Systems segment net sales decreased 22 percent in the quarter.\n"
     "Backlog for the segment was $2.61 billion at the end of the quarter.\n"
     "Book-to-bill for the period was 0.92, below parity for the second quarter running.\n"
     "Management noted softer foundry and memory customer investment.\n"),
    ("d_mu_10q", "MU", "2023-01-05", "10-Q", "mdna",
     "Revenue decreased 39% as compared to the prior quarter.\n"
     "Days of inventory increased to 214 days at the end of the period.\n"
     "Average selling prices per bit declined approximately 25%.\n"
     "Capital expenditure plans were reduced in response to the pricing environment.\n"),
    ("d_amat_fut", "AMAT", "2020-02-12", "8-K", "full",
     "Applied reported record bookings and a book-to-bill of 1.24 in the quarter.\n"
     "Backlog rose to $4.10 billion, an all time high for the segment.\n"),
]


@pytest.fixture
def store(tmp_path) -> ChunkStore:
    st = ChunkStore(tmp_path / "idx")
    next_id = 1
    for doc_id, ticker, d, stype, section, text in DOCS:
        path = tmp_path / f"{doc_id}.txt"
        path.write_text(text)
        rows, texts = [], []
        for ch in chunk_document(doc_id, text, SPEC):
            rows.append((next_id, doc_id, ch.ordinal, ch.start, ch.end, len(ch.text),
                         ticker, "0001", d, stype, section, str(path)))
            texts.append(ch.text)
            next_id += 1
        st.add_batch(rows, texts, rows[0][0])
    return st


def test_chunks_read_back_verbatim(store, tmp_path):
    for ref in store.get(store.candidate_ids()):
        raw = (tmp_path / f"{ref.doc_id}.txt").read_text()
        assert ref.text() == raw[ref.start:ref.end]


def test_as_of_excludes_future_documents(store):
    ids = store.candidate_ids(tickers=["AMAT"], as_of=date(2019, 12, 31))
    docs = {r.doc_id for r in store.get(ids)}
    assert docs == {"d_amat_10q"}, "a filing published after as_of must not be a candidate"

    later = store.candidate_ids(tickers=["AMAT"], as_of=date(2020, 6, 1))
    assert {r.doc_id for r in store.get(later)} == {"d_amat_10q", "d_amat_fut"}


def test_ticker_filter(store):
    ids = store.candidate_ids(tickers=["MU"])
    assert {r.ticker for r in store.get(ids)} == {"MU"}


def test_section_and_source_type_filters(store):
    ids = store.candidate_ids(source_types=["8-K"])
    assert {r.doc_id for r in store.get(ids)} == {"d_amat_fut"}
    ids = store.candidate_ids(sections=["mdna"])
    assert {r.doc_id for r in store.get(ids)} == {"d_amat_10q", "d_mu_10q"}


def test_bm25_finds_the_right_passage(store):
    s = Searcher(store, None, SearchConfig(mode="bm25", top_k=3))
    hits = s.search("days of inventory", tickers=["MU"])
    assert hits and "Days of inventory increased to 214 days" in hits[0].ref.text()


def test_search_respects_as_of(store):
    s = Searcher(store, None, SearchConfig(mode="bm25", top_k=5))
    hits = s.search("book-to-bill", tickers=["AMAT"], as_of=date(2019, 12, 31))
    assert hits
    assert all(h.ref.doc_id == "d_amat_10q" for h in hits)
    assert all("1.24" not in h.ref.text() for h in hits), "future figure leaked in"


def test_dense_mode_without_vectors_returns_nothing(store):
    s = Searcher(store, None, SearchConfig(mode="dense"))
    assert s.search("days of inventory", tickers=["MU"]) == []


def test_max_per_doc_caps_one_document(store):
    s = Searcher(store, None, SearchConfig(mode="bm25", top_k=10, max_per_doc=1))
    hits = s.search("backlog quarter segment sales", tickers=["AMAT"])
    assert len({h.ref.doc_id for h in hits}) == len(hits)


def test_passage_expansion_stays_verbatim(store, tmp_path):
    s = Searcher(store, None, SearchConfig(mode="bm25", top_k=1))
    h = s.search("days of inventory", tickers=["MU"])[0]
    lo, hi, text = h.passage()
    raw = (tmp_path / "d_mu_10q.txt").read_text()
    assert text == raw[lo:hi]
    assert h.ref.text() in text, "expansion must contain the chunk it came from"


def test_neighbours(store):
    ids = store.candidate_ids(tickers=["MU"])
    refs = store.get(ids)
    n = store.neighbours(refs[0], radius=1)
    assert refs[0].chunk_id in {r.chunk_id for r in n}


def test_empty_candidate_set_returns_no_hits(store):
    s = Searcher(store, None, SearchConfig(mode="bm25"))
    assert s.search("backlog", tickers=["ZZZZ"]) == []


class TestFtsQuery:
    def test_quotes_hyphenated_terms(self):
        # Unquoted, FTS5 reads `-` as syntax and raises on entirely ordinary input.
        q = fts_query("book-to-bill ratio")
        assert '"book-to-bill"' in q and " OR " in q

    def test_drops_stopwords(self):
        assert "the" not in fts_query("the backlog of the quarter").lower().split()

    def test_empty_query(self):
        assert fts_query("the of and") == ""

    def test_does_not_raise_on_punctuation(self, store):
        s = Searcher(store, None, SearchConfig(mode="bm25"))
        for q in ["R&D spend", "10-K risk factors", '"quoted" (parens)', "***", "a/b"]:
            s.search(q, tickers=["MU"])   # must not raise
