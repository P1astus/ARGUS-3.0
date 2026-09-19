"""ARGUS command line."""

from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="ARGUS -- semiconductor swing-trade decision support")
quant_app = typer.Typer(help="Stage 1: quantitative ranking core")
app.add_typer(quant_app, name="quant")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@quant_app.command("build-dataset")
def build_dataset(
    universe: Path = typer.Option("configs/universe/semis_v1.yaml"),
    end: str = typer.Option(None, help="ISO end date; defaults to today"),
    horizon: int = typer.Option(10, help="label horizon in trading days"),
    rebalance: str = typer.Option("W-FRI"),
    out: Path = typer.Option("data/interim/panel.parquet"),
    refresh: bool = typer.Option(False, help="bypass the price cache"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Fetch prices, build features and labels, write the modelling panel."""
    _setup_logging(verbose)
    from argus.data.prices import PriceStore
    from argus.data.providers.yfinance_provider import YFinanceProvider
    from argus.quant.dataset import build_panel
    from argus.quant.labels import LabelSpec
    from argus.quant.universe import load_universe

    spec = load_universe(universe)
    store = PriceStore(YFinanceProvider())
    end_date = date.fromisoformat(end) if end else date.today()

    panel = build_panel(
        store, spec, end_date,
        label_spec=LabelSpec(horizon_days=horizon),
        rebalance=rebalance,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    panel.features.join(panel.labels).to_parquet(out)
    panel.close.to_parquet(out.with_name(out.stem + "_close.parquet"))
    panel.coverage.to_parquet(out.with_name(out.stem + "_coverage.parquet"))

    from argus.quant.universe import summarize_coverage
    cov = summarize_coverage(panel.coverage)
    typer.echo(f"\npanel: {len(panel.features):,} rows x {len(panel.feature_names)} features")
    typer.echo(f"dates: {len(panel.dates)}  "
               f"({panel.dates[0].date()} .. {panel.dates[-1].date()})")
    typer.echo(f"universe coverage: mean {cov['mean_coverage']:.1%}, "
               f"worst year {cov['worst_year']} at {cov['worst_year_coverage']:.1%}")
    typer.echo(f"wrote {out}")


@quant_app.command("validate")
def validate(
    universe: Path = typer.Option("configs/universe/semis_v1.yaml"),
    end: str = typer.Option(None),
    horizon: int = typer.Option(10),
    folds: int = typer.Option(5),
    rebalance: str = typer.Option("W-FRI"),
    features: str = typer.Option("v2", help="feature set: v1, v2, or core"),
    model: str = typer.Option("gbdt", help="model: gbdt or ridge"),
    cost_bps: float = typer.Option(10.0),
    min_train_dates: int = typer.Option(150, help="rebalance periods, not trading days"),
    report_out: Path = typer.Option("artifacts/phase1_validation.txt"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Run purged walk-forward validation. This output IS the Phase 1 gate."""
    _setup_logging(verbose)
    from argus.data.prices import PriceStore
    from argus.data.providers.yfinance_provider import YFinanceProvider
    from argus.quant.dataset import build_panel
    from argus.quant.labels import LabelSpec
    from argus.quant.report import evaluate, render, run_walk_forward
    from argus.quant.universe import load_universe
    from argus.quant.validation.splitter import PurgedWalkForward

    spec = load_universe(universe)
    store = PriceStore(YFinanceProvider())
    end_date = date.fromisoformat(end) if end else date.today()
    label_spec = LabelSpec(horizon_days=horizon)

    from argus.quant.features import DEFAULT_FEATURES, FEATURES_V2
    from argus.quant.model import FEATURES_CORE, LinearRanker
    fset = {"v1": DEFAULT_FEATURES, "v2": FEATURES_V2, "core": FEATURES_CORE}[features]
    model_factory = (lambda: LinearRanker(alpha=10.0)) if model == "ridge" else None
    panel = build_panel(store, spec, end_date, label_spec=label_spec,
                        rebalance=rebalance, feature_names=fset)

    # Horizon is expressed in trading days but the panel is sampled weekly, so the
    # splitter's purge/embargo must be converted to rebalance periods or it would
    # over-purge by a factor of ~5.
    periods_per_horizon = max(1, round(horizon / 5)) if rebalance.startswith("W") else horizon
    cv = PurgedWalkForward(
        n_splits=folds,
        horizon=periods_per_horizon,
        min_train_dates=min_train_dates,
    )
    cv.assert_no_leakage(panel.dates)

    results = evaluate(panel, run_walk_forward(panel, cv, model_factory=model_factory),
                       cost_bps=cost_bps)
    text = render(results, panel)

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(text)
    typer.echo(text)
    typer.echo(f"\nwrote {report_out}")

    raise typer.Exit(0 if results["gate"].passed else 1)


engine_app = typer.Typer(help="Stage 5: the trained engine")
app.add_typer(engine_app, name="engine")


@engine_app.command("corpus-build")
def corpus_build(
    universe: Path = typer.Option("configs/universe/semis_v1.yaml"),
    cutoff: str = typer.Option("2025-12-31", help="hard corpus date cutoff"),
    start: str = typer.Option("2010-01-01"),
    forms: str = typer.Option("10-K,10-Q,8-K",
        help="comma-separated SEC form types; add 20-F for foreign private issuers "
             "(TSM/UMC/ASML/STM/... never file 10-K -- see handoff §11)"),
    user_agent: str = typer.Option(os.environ.get("ARGUS_SEC_UA", "ARGUS Research contact@example.com")),
    limit: int = typer.Option(None, help="cap documents fetched (for a trial run)"),
    skip_fetch: bool = typer.Option(False, help="process already-fetched docs only"),
    skip_discovery: bool = typer.Option(False, help="reuse the existing catalog; saves ~1h"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Build the CPT corpus from EDGAR. Resumable; safe to re-run."""
    _setup_logging(verbose)
    from datetime import date as _date
    from argus.engine.corpus.build import CorpusConfig, build

    cfg = CorpusConfig(universe=str(universe), start=_date.fromisoformat(start),
                       cutoff=_date.fromisoformat(cutoff),
                       forms=tuple(f.strip() for f in forms.split(",") if f.strip()))
    manifest = build(cfg, user_agent=user_agent, skip_fetch=skip_fetch,
                     skip_discovery=skip_discovery, limit=limit)

    typer.echo("")
    typer.echo(f"documents:  {manifest['total_documents']:,}")
    typer.echo(f"tokens:     {manifest['total_tokens']:,}")
    typer.echo(f"dedup rate: {manifest['dedup']['rate']:.1%}")
    typer.echo(f"date range: {manifest['date_range']['min']} .. {manifest['date_range']['max']}")
    typer.echo(f"cutoff:     {manifest['cutoff']}")
    typer.echo(f"manifest sha256: {manifest['sha256'][:16]}")


@engine_app.command("corpus-status")
def corpus_status(root: Path = typer.Option("data/corpus")) -> None:
    """Show catalog progress without touching the network."""
    from argus.engine.corpus.catalog import Catalog
    m = Catalog(root / "catalog.sqlite").manifest()
    typer.echo(f"total documents: {m['total_documents']:,}")
    for status, n in sorted(m["by_status"].items()):
        typer.echo(f"  {status:14s} {n:>8,}")
    typer.echo(f"tokens (deduped+packed): {m['total_tokens']:,}")


retrieve_app = typer.Typer(help="Corpus retrieval: index, search, evaluate")
app.add_typer(retrieve_app, name="retrieve")


def _open_index(index_root: Path, dense: bool = True):
    """Open the chunk store, attaching the embedder only if vectors actually exist.

    Falling back to lexical-only rather than failing is deliberate: the lexical index
    builds in 11 seconds and the dense pass takes ~35 minutes, so search has to be usable
    in between -- otherwise the only way to sanity-check a fresh index is to wait.
    """
    from argus.engine.retrieval.store import ChunkStore
    store = ChunkStore(index_root)
    embedder = None
    if dense and store.vectors is not None:
        from argus.engine.retrieval.embed import Embedder
        embedder = Embedder()
    elif dense:
        typer.echo("note: no vector index found; running lexical-only", err=True)
    return store, embedder


@retrieve_app.command("index")
def retrieve_index(
    corpus: Path = typer.Option("data/corpus"),
    index_root: Path = typer.Option("data/index"),
    lexical: bool = typer.Option(True, help="chunk + BM25 pass (fast)"),
    dense: bool = typer.Option(True, help="embedding pass (~35 min for the full corpus)"),
    resume: bool = typer.Option(True, help="continue an interrupted dense pass"),
    budget_seconds: float = typer.Option(None, help="stop the dense pass cleanly after N s"),
    limit: int = typer.Option(None, help="cap documents (for a trial run)"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Build the retrieval index over the corpus. Both passes are resumable."""
    _setup_logging(verbose)
    from argus.engine.retrieval.index import IndexConfig, build_dense, build_lexical

    cfg = IndexConfig(corpus_root=corpus, index_root=index_root)
    if lexical:
        st = build_lexical(cfg, limit=limit, reset=True)
        typer.echo(f"lexical: {st['chunks']:,} chunks from {st['documents']:,} docs "
                   f"in {st['seconds']}s ({st['skipped']} skipped)")
    if dense:
        st = build_dense(cfg, resume=resume, budget_seconds=budget_seconds)
        typer.echo(f"dense:   {st['chunks']:,}/{st['total']:,} vectors in {st['seconds']}s"
                   + ("" if st["complete"] else "  [INCOMPLETE -- rerun to resume]"))


@retrieve_app.command("search")
def retrieve_search(
    query: str = typer.Argument(...),
    ticker: str = typer.Option(None, help="comma-separated tickers"),
    as_of: str = typer.Option(None, help="exclude documents published after this date"),
    k: int = typer.Option(8),
    mode: str = typer.Option("hybrid", help="hybrid | bm25 | dense"),
    index_root: Path = typer.Option("data/index"),
    chars: int = typer.Option(300, help="preview characters per hit"),
) -> None:
    """Search the corpus. Useful for checking what the model would actually be shown."""
    from argus.engine.retrieval.search import SearchConfig, Searcher

    store, embedder = _open_index(index_root, dense=mode in ("hybrid", "dense"))
    searcher = Searcher(store, embedder)
    hits = searcher.search(
        query,
        tickers=[t.strip().upper() for t in ticker.split(",")] if ticker else None,
        as_of=date.fromisoformat(as_of) if as_of else None,
        config=SearchConfig(top_k=k, mode=mode))

    if not hits:
        typer.echo("no hits")
        raise typer.Exit(1)
    for i, h in enumerate(hits, start=1):
        r = h.ref
        typer.echo(f"\n{i:2d}. {r.ticker} {r.source_type}/{r.section} {r.doc_date}  "
                   f"rrf={h.score:.4f} bm25={h.rank_bm25} dense={h.rank_dense}")
        typer.echo("    " + r.text()[:chars].replace("\n", " "))


@retrieve_app.command("evaluate")
def retrieve_evaluate(
    index_root: Path = typer.Option("data/index"),
    k: int = typer.Option(20),
    out: Path = typer.Option("artifacts/retrieval_eval.json"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Score bm25 vs dense vs hybrid on the rule-defined probes."""
    _setup_logging(verbose)
    import json

    from argus.engine.retrieval import evaluate as ev

    store, embedder = _open_index(index_root)
    report = ev.run(store, embedder, k=k)
    text = ev.render(report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    out.with_suffix(".txt").write_text(text)
    typer.echo(text)
    typer.echo(f"\nwrote {out}")


extract_app = typer.Typer(help="Local extraction: passages -> Claims -> Briefing")
app.add_typer(extract_app, name="extract")


@extract_app.command("brief")
def extract_brief(
    ticker: str = typer.Argument(...),
    as_of: str = typer.Option(None, help="decision date; nothing published after it is visible"),
    universe: Path = typer.Option("configs/universe/semis_v1.yaml"),
    model: str = typer.Option("base", help="base | instruct | a full MLX model id"),
    index_root: Path = typer.Option("data/index"),
    provenance: Path = typer.Option("data/provenance"),
    out: Path = typer.Option("artifacts/briefings"),
    live: bool = typer.Option(False, help="fetch the most recent filings directly from "
                              "SEC instead of the pre-built index -- use this for dates "
                              "past the corpus cutoff (the index cannot see filings it "
                              "was never built with, regardless of --as-of)"),
    live_limit: int = typer.Option(8, help="--live only: how many recent filings to fetch"),
    price: bool = typer.Option(True, help="attach a price/volume snapshot (yfinance)"),
    insider: bool = typer.Option(True, help="attach Form 3/4/5 insider activity (EDGAR)"),
    macro: bool = typer.Option(True, help="attach macro series context (FRED, no key needed)"),
    analyst: bool = typer.Option(True, help="attach analyst rating/target consensus, "
                                 "reconstructed as-of (yfinance, no key needed)"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Build one briefing and write the markdown hand-off."""
    _setup_logging(verbose)
    from argus.engine.extract.export import filename, to_markdown
    from argus.engine.extract.loop import ExtractionLoop
    from argus.engine.extract.prompts import STOP_SEQUENCES
    from argus.engine.inference.local import (BASE_MODEL, INSTRUCT_MODEL, GenConfig,
                                              MLXGenerator)
    from argus.engine.retrieval.search import Searcher
    from argus.quant.universe import load_universe
    from argus.tools.provenance import ProvenanceStore, audit_briefing

    ticker = ticker.upper()
    spec = load_universe(universe)
    if ticker not in spec.tickers:
        typer.echo(f"{ticker} is not in {universe}", err=True)
        raise typer.Exit(2)

    model_id = {"base": BASE_MODEL, "instruct": INSTRUCT_MODEL}.get(model, model)
    gen = MLXGenerator(GenConfig(model=model_id, stop=STOP_SEQUENCES))
    when = date.fromisoformat(as_of) if as_of else date.today()

    if live:
        from argus.engine.extract.live import run_live
        from argus.tools.retrievers.live_edgar import LiveEdgarRetriever
        res = run_live(ticker, spec.sub_segment(ticker), when, LiveEdgarRetriever(),
                       gen, ProvenanceStore(provenance), limit=live_limit)
    else:
        store, embedder = _open_index(index_root)
        loop = ExtractionLoop(Searcher(store, embedder), gen, ProvenanceStore(provenance))
        res = loop.run(ticker, spec.sub_segment(ticker), when)

    # Structured appendices -- displayed directly, no claim extraction involved (see
    # contracts/context.py). Each is independently best-effort: a failed fetch drops that
    # one section rather than failing the whole briefing, since none of the three is part
    # of the Phase 4 provenance gate `audit_briefing` checks below.
    updates = {}
    if price:
        from argus.data.prices import PriceStore
        from argus.data.providers.yfinance_provider import YFinanceProvider
        from argus.tools.price_context import build_price_snapshot
        snap = build_price_snapshot(PriceStore(YFinanceProvider()), ticker, when)
        if snap is not None:
            updates["price"] = snap
    if insider:
        from argus.tools.insider import build_insider_activity
        activity = build_insider_activity(ticker, when)
        if activity is not None:
            updates["insider_activity"] = activity
    if macro:
        from argus.tools.macro import build_macro_context
        updates["macro"] = build_macro_context(when)
    if analyst:
        from argus.tools.analyst import build_analyst_consensus
        consensus = build_analyst_consensus(ticker, when)
        if consensus is not None:
            updates["analyst"] = consensus
    if updates:
        res.briefing = res.briefing.model_copy(update=updates)

    audit = audit_briefing(res.briefing, ProvenanceStore(provenance))
    md = to_markdown(res.briefing, {**res.audit, "model": model_id})
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename(res.briefing)
    path.write_text(md)

    typer.echo(f"{len(res.briefing.claims)} claims "
               f"({res.briefing.sourced_fraction:.0%} sourced) from "
               f"{len(res.briefing.sources)} passages in {res.seconds:.0f}s")
    typer.echo(f"exact quotes {res.stats.exact_quote_rate:.0%}, "
               f"repaired {res.stats.quote_whitespace}, "
               f"fabricated (dropped) {res.stats.quote_fabricated}")
    typer.echo(f"quote audit against retained bytes: "
               f"{'PASS' if audit['passed'] else 'FAIL'}")
    typer.echo(f"wrote {path}")


@extract_app.command("baseline")
def extract_baseline(
    index_root: Path = typer.Option("data/index"),
    provenance: Path = typer.Option("data/provenance"),
    out: Path = typer.Option("artifacts/baseline"),
    arms: str = typer.Option("base,instruct", help="comma-separated: base, instruct"),
    cases: str = typer.Option("semis", help="case set: semis | tech"),
    limit: int = typer.Option(None, help="use only the first N cases"),
    budget_seconds: float = typer.Option(
        None, help="stop after this long; rerun to resume (a case costs ~150-310s)"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Measure extraction quality on the UNTRAINED model.

    This is the denominator for any later CPT comparison -- run it before training, not
    after.

    Resumable: completed cases are recorded in <out>/<arm>/cases.jsonl and skipped on a
    rerun. Run it in the FOREGROUND -- a case costs ~150-310s interactive and dramatically
    more detached (see handoff §6) -- using --budget-seconds to fit each run inside
    whatever wall-clock window you have.
    """
    _setup_logging(verbose)
    from argus.engine.extract import baseline as bl
    from argus.engine.inference.local import BASE_MODEL, INSTRUCT_MODEL
    from argus.engine.retrieval.search import Searcher
    from argus.tools.provenance import ProvenanceStore

    lookup = {"base": BASE_MODEL, "instruct": INSTRUCT_MODEL}
    arm_ids = tuple(lookup.get(a.strip(), a.strip()) for a in arms.split(",") if a.strip())
    case_sets = {"semis": bl.DEFAULT_CASES, "tech": bl.TECH_CASES}
    cases = case_sets[cases]
    cases = cases[:limit] if limit else cases

    store, embedder = _open_index(index_root)
    report = bl.run(Searcher(store, embedder), ProvenanceStore(provenance),
                    out, arms=arm_ids, cases=cases, budget_seconds=budget_seconds)
    text = bl.render(report)
    (out / "baseline.txt").write_text(text)
    typer.echo("\n" + text)

    incomplete = [m for m, a in report["arms"].items()
                  if not a["summary"].get("complete")]
    if incomplete:
        typer.echo(f"\nINCOMPLETE ({len(cases)} cases expected) -- rerun to resume: "
                   + ", ".join(m.split('/')[-1] for m in incomplete))
    typer.echo(f"\nwrote {out}/baseline.json and {out}/baseline.txt")


@quant_app.command("features")
def list_features() -> None:
    """List registered features."""
    from argus.quant.features import registry
    for name in registry.available():
        marker = "*" if name in registry.DEFAULT_FEATURES else " "
        typer.echo(f" {marker} {name}")
    typer.echo("\n* = in DEFAULT_FEATURES")


journal_app = typer.Typer(help="Stage 6: the trade journal (forward-recorded evidence)")
app.add_typer(journal_app, name="journal")


def _journal(dsn: str):
    from argus.journal.repository import Journal
    return Journal(dsn=dsn)


@journal_app.command("record")
def journal_record(
    ticker: str = typer.Argument(...),
    segment: str = typer.Option(..., help="SubSegment value, e.g. fabless, tech_cloud_saas"),
    direction: str = typer.Option(..., help="long | short | flat"),
    thesis: str = typer.Option(..., help="the base-case thesis, one or two sentences"),
    entry: float = typer.Option(None, help="required unless direction=flat"),
    target: float = typer.Option(None, help="required unless direction=flat"),
    invalidation: float = typer.Option(None, help="required unless direction=flat"),
    conviction: float = typer.Option(..., min=0.0, max=1.0),
    holding_days: int = typer.Option(10, help="target holding period, trading days"),
    bull_target: float = typer.Option(None, help="defaults to entry + 2x the base reward"),
    bull_thesis: str = typer.Option("Upside case.", help="one sentence"),
    bull_prob: float = typer.Option(0.25),
    base_prob: float = typer.Option(0.50),
    bear_thesis: str = typer.Option("Downside case.", help="one sentence"),
    bear_prob: float = typer.Option(0.25),
    quant_score: float = typer.Option(None),
    quant_percentile: float = typer.Option(None),
    arm: str = typer.Option("human", help="human | quant_only | base_prompt | cpt_sft"),
    as_of: str = typer.Option(None, help="ISO date; defaults to today"),
    entry_price: float = typer.Option(None,
        help="actual fill; defaults to --entry (the planned price) if omitted"),
    paper: bool = typer.Option(True, "--paper/--live",
        help="--live records a real position; get this right, it does not ask twice"),
    dsn: str = typer.Option("data/journal.sqlite"),
    corpus_manifest: Path = typer.Option("data/corpus/manifest.json",
        help="read for corpus_manifest_sha256/cutoff provenance; skipped if missing"),
) -> None:
    """Record a recommendation, and open a paper (or live) trade if it is not FLAT.

    This is Stage 6 in isolation -- it does not run retrieval or extraction. Use
    `extract brief` first to gather sourced evidence, do the actual analysis with the
    frontier model in conversation, then record the decision here. A CLI command cannot
    and should not manufacture the judgment call itself; recording it is all this does.

    Non-trades matter as much as trades: a model (or a person) measured only on the
    setups it chose to take looks more skilled than it is, because declining bad setups is
    most of the skill. Record a `--direction flat` recommendation with a real thesis when
    the honest answer is "no trade," not only when there is one.
    """
    import json as _json

    from argus.contracts.provenance import Arm, RunProvenance
    from argus.contracts.quant import SubSegment
    from argus.contracts.recommendation import (Direction, Levels, Recommendation,
                                                Scenario, ScenarioKind)

    seg = SubSegment(segment)
    dirn = Direction(direction)
    when = date.fromisoformat(as_of) if as_of else date.today()

    scenarios = []
    if dirn == Direction.FLAT:
        if any(v is not None for v in (entry, target, invalidation)):
            typer.echo("note: entry/target/invalidation are ignored for a FLAT "
                      "recommendation", err=True)
        # A FLAT rec still needs 3 scenarios to satisfy the schema; they carry no levels
        # and are not tradeable -- this is bookkeeping, not a hidden trade plan.
        scenarios = [
            Scenario(kind=ScenarioKind.BULL, thesis=bull_thesis, probability=bull_prob),
            Scenario(kind=ScenarioKind.BASE, thesis=thesis, probability=base_prob),
            Scenario(kind=ScenarioKind.BEAR, thesis=bear_thesis, probability=bear_prob),
        ]
        chosen = ScenarioKind.BASE
    else:
        if entry is None or target is None or invalidation is None:
            typer.echo("--entry, --target and --invalidation are required unless "
                      "--direction flat", err=True)
            raise typer.Exit(2)
        bt = bull_target if bull_target is not None else entry + 2 * abs(target - entry)
        scenarios = [
            Scenario(kind=ScenarioKind.BULL, thesis=bull_thesis, probability=bull_prob,
                     levels=Levels(entry=entry, target=bt, invalidation=invalidation)),
            Scenario(kind=ScenarioKind.BASE, thesis=thesis, probability=base_prob,
                     levels=Levels(entry=entry, target=target, invalidation=invalidation)),
            Scenario(kind=ScenarioKind.BEAR, thesis=bear_thesis, probability=bear_prob,
                     levels=Levels(entry=entry, target=invalidation,
                                  invalidation=invalidation)),
        ]
        chosen = ScenarioKind.BASE

    rec = Recommendation(
        ticker=ticker.upper(), sub_segment=seg, as_of=when, direction=dirn,
        conviction=conviction, scenarios=scenarios, chosen=chosen,
        target_holding_days=holding_days, quant_score=quant_score,
        quant_percentile=quant_percentile, summary=thesis,
    )

    manifest_sha, cutoff = None, None
    if corpus_manifest.exists():
        m = _json.loads(corpus_manifest.read_text())
        manifest_sha, cutoff = m.get("sha256"), m.get("cutoff")

    prov = RunProvenance(arm=Arm(arm), corpus_manifest_sha256=manifest_sha,
                        corpus_cutoff=cutoff)

    j = _journal(dsn)
    rec_id = j.record_recommendation(rec, prov)
    typer.echo(f"recommendation #{rec_id} recorded ({dirn}, conviction {conviction:.2f})")

    if dirn != Direction.FLAT:
        fill = entry_price if entry_price is not None else entry
        label = "LIVE" if not paper else "paper"
        trade_id = j.open_trade(rec, prov, entry_price=fill, is_paper=paper)
        typer.echo(f"{label} trade #{trade_id} OPENED: {ticker.upper()} {dirn} @ {fill:.2f}  "
                   f"target {target:.2f}  invalidation {invalidation:.2f}  "
                   f"holding {holding_days}d")
        if not paper:
            typer.echo("*** LIVE trade recorded -- this is not a drill ***")


@journal_app.command("close")
def journal_close(
    trade_id: int = typer.Argument(...),
    exit_price: float = typer.Option(...),
    reason: str = typer.Option(..., help="target | invalidation | time | discretionary"),
    exit_date_: str = typer.Option(None, "--exit-date", help="ISO date; defaults to today"),
    benchmark_return: float = typer.Option(None,
        help="equal-weight sector return over the same window; omit if unknown"),
    dsn: str = typer.Option("data/journal.sqlite"),
) -> None:
    """Close an open trade and compute its realized (and relative, if given) return."""
    when = date.fromisoformat(exit_date_) if exit_date_ else date.today()
    j = _journal(dsn)
    j.close_trade(trade_id, when, exit_price, reason, benchmark_return)
    typer.echo(f"trade #{trade_id} closed @ {exit_price:.2f} ({reason})")
    if benchmark_return is None:
        typer.echo("note: no benchmark_return given -- realized_rel_return is unset, "
                  "so this trade won't count in performance_by_arm until it is backfilled")


@journal_app.command("status")
def journal_status(dsn: str = typer.Option("data/journal.sqlite")) -> None:
    """Open positions and performance-by-arm -- the prospective Phase 3 gate, so far."""
    j = _journal(dsn)
    open_trades = j.open_trades()
    typer.echo(f"OPEN ({len(open_trades)})")
    for t in open_trades:
        typer.echo(f"  #{t.id:<4d} {t.ticker:<6s} entered {t.entry_date} @ "
                   f"{t.entry_price:.2f}  arm={t.arm}"
                   + (f"  conviction={t.conviction:.2f}" if t.conviction else ""))

    perf = j.performance_by_arm()
    typer.echo(f"\nCLOSED, by arm ({sum(p['n_trades'] for p in perf.values())} total)")
    if not perf:
        typer.echo("  none closed yet")
    for arm, p in perf.items():
        typer.echo(f"  {arm:<12s} n={p['n_trades']:<3d} "
                   f"mean_rel_return={p['mean_rel_return']:+.2%}  "
                   f"hit_rate={p['hit_rate']:.0%}  t={p['t_stat']:.2f}")
    if perf and max(p["n_trades"] for p in perf.values()) < 20:
        typer.echo("\n  (n is small -- t-stats here are not yet meaningful; reported so "
                  "you can see that rather than infer significance from a mean alone)")


if __name__ == "__main__":
    app()
