"""Trade journal persistence.

Postgres is the target (per the project spec), but the layer is written against the DB-API
so the same code runs on SQLite. That is not a hedge -- it means the journal works and is
testable today, before a Postgres install exists, and switching is a connection-string
change.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from argus.contracts.provenance import Arm, RunProvenance
from argus.contracts.recommendation import Direction, Recommendation

log = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


@dataclass
class TradeRow:
    id: int
    ticker: str
    entry_date: date
    entry_price: float
    exit_date: date | None
    realized_rel_return: float | None
    arm: str
    conviction: float | None

    @property
    def is_open(self) -> bool:
        return self.exit_date is None


class Journal:
    """Trade journal.

    Args:
        dsn: a Postgres URL (``postgresql://...``) or a filesystem path for SQLite.
    """

    def __init__(self, dsn: str = "data/journal.sqlite") -> None:
        self.dsn = dsn
        self.is_postgres = dsn.startswith(("postgres://", "postgresql://"))
        if not self.is_postgres:
            Path(dsn).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        if self.is_postgres:
            import psycopg  # imported lazily so SQLite use needs no driver
            conn = psycopg.connect(self.dsn)
        else:
            conn = sqlite3.connect(self.dsn)
            conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ph(self) -> str:
        """Parameter placeholder -- psycopg uses %s, sqlite3 uses ?."""
        return "%s" if self.is_postgres else "?"

    def _init_schema(self) -> None:
        sql = SCHEMA_PATH.read_text()
        if self.is_postgres:
            # SQLite's INTEGER PRIMARY KEY autoincrements; Postgres needs an explicit
            # identity column.
            sql = sql.replace("id                    INTEGER PRIMARY KEY",
                              "id                    SERIAL PRIMARY KEY")
        with self._conn() as c:
            cur = c.cursor()
            for stmt in filter(None, (s.strip() for s in sql.split(";"))):
                cur.execute(stmt)

    # ------------------------------------------------------------------ writes

    def record_recommendation(
        self,
        rec: Recommendation,
        prov: RunProvenance,
        briefing_json: str | None = None,
        sourced_fraction: float | None = None,
    ) -> int:
        """Store a recommendation, traded or not.

        Non-trades are recorded deliberately: measuring a model only on the trades it
        chose to take flatters it, because declining bad setups is most of the skill and
        calibration needs the denominator.
        """
        p = self._ph()
        with self._conn() as c:
            cur = c.cursor()
            cur.execute(
                f"""INSERT INTO recommendations
                    (ticker, sub_segment, as_of, direction, conviction, payload_json,
                     briefing_json, sourced_fraction, arm, provenance_key, created_at)
                    VALUES ({p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p})""",
                (rec.ticker, str(rec.sub_segment), rec.as_of, str(rec.direction),
                 rec.conviction, rec.model_dump_json(), briefing_json, sourced_fraction,
                 str(prov.arm), prov.key(), datetime.now(timezone.utc)),
            )
            return self._last_id(cur)

    def open_trade(self, rec: Recommendation, prov: RunProvenance,
                   entry_price: float, is_paper: bool = True) -> int:
        """Open a position from a recommendation."""
        if rec.direction == Direction.FLAT:
            raise ValueError("cannot open a trade from a FLAT recommendation")

        s = rec.chosen_scenario
        p = self._ph()
        now = datetime.now(timezone.utc)
        with self._conn() as c:
            cur = c.cursor()
            cur.execute(
                f"""INSERT INTO trades
                    (ticker, sub_segment, entry_date, entry_price, direction,
                     quant_score, quant_percentile, thesis, scenario_chosen,
                     scenario_probability, conviction, take_profit, invalidation,
                     target_holding_days, arm, base_model, adapter_sha256,
                     corpus_manifest_sha256, corpus_cutoff, config_sha256, code_git_sha,
                     quant_model_version, is_paper, created_at, updated_at)
                    VALUES ({','.join([p]*25)})""",
                (rec.ticker, str(rec.sub_segment), rec.as_of, entry_price,
                 str(rec.direction), rec.quant_score, rec.quant_percentile, s.thesis,
                 str(rec.chosen), s.probability, rec.conviction,
                 s.levels.target if s.levels else None,
                 s.levels.invalidation if s.levels else None,
                 rec.target_holding_days, str(prov.arm), prov.base_model,
                 prov.adapter_sha256, prov.corpus_manifest_sha256, prov.corpus_cutoff,
                 prov.config_sha256, prov.code_git_sha, prov.quant_model_version,
                 1 if is_paper else 0, now, now),
            )
            return self._last_id(cur)

    def close_trade(self, trade_id: int, exit_date: date, exit_price: float,
                    exit_reason: str, benchmark_return: float | None = None) -> None:
        """Close a position and compute its relative outcome.

        Relative return is stored because it is what the labels measure -- an absolute
        gain during a sector-wide rally is not evidence of skill.
        """
        p = self._ph()
        with self._conn() as c:
            cur = c.cursor()
            cur.execute(f"SELECT entry_price, direction FROM trades WHERE id = {p}", (trade_id,))
            row = cur.fetchone()
            if row is None:
                raise KeyError(f"no trade with id {trade_id}")
            entry_price, direction = row[0], row[1]

            raw = exit_price / entry_price - 1.0
            if direction == "short":
                raw = -raw
            rel = None if benchmark_return is None else raw - benchmark_return

            cur.execute(
                f"""UPDATE trades SET exit_date={p}, exit_price={p}, exit_reason={p},
                    realized_return={p}, benchmark_return={p}, realized_rel_return={p},
                    updated_at={p} WHERE id={p}""",
                (exit_date, exit_price, exit_reason, raw, benchmark_return, rel,
                 datetime.now(timezone.utc), trade_id),
            )

    def _last_id(self, cur) -> int:
        if self.is_postgres:
            cur.execute("SELECT lastval()")
            return int(cur.fetchone()[0])
        return int(cur.lastrowid)

    # ------------------------------------------------------------------- reads

    def open_trades(self) -> list[TradeRow]:
        return self._select("SELECT * FROM trades WHERE exit_date IS NULL ORDER BY entry_date")

    def closed_trades(self, arm: str | None = None) -> list[TradeRow]:
        q = "SELECT * FROM trades WHERE exit_date IS NOT NULL"
        params = ()
        if arm:
            q += f" AND arm = {self._ph()}"
            params = (arm,)
        return self._select(q + " ORDER BY entry_date", params)

    def _select(self, query: str, params: tuple = ()) -> list[TradeRow]:
        with self._conn() as c:
            cur = c.cursor()
            cur.execute(query, params)
            cols = [d[0] for d in cur.description]
            rows = []
            for r in cur.fetchall():
                d = dict(zip(cols, r))
                rows.append(TradeRow(
                    id=d["id"], ticker=d["ticker"], entry_date=d["entry_date"],
                    entry_price=d["entry_price"], exit_date=d["exit_date"],
                    realized_rel_return=d["realized_rel_return"], arm=d["arm"],
                    conviction=d["conviction"],
                ))
            return rows

    def performance_by_arm(self) -> dict[str, dict]:
        """Outcome summary per arm -- the prospective form of the Phase 3 gate.

        Retrospective evaluation is contaminated by the base model's own pretraining.
        These numbers are not: they come from decisions recorded before the outcome was
        known. Slow to accumulate, and the only genuinely uncontaminated evidence
        available.
        """
        out: dict[str, dict] = {}
        for arm in Arm:
            trades = self.closed_trades(arm=str(arm))
            rels = [t.realized_rel_return for t in trades if t.realized_rel_return is not None]
            if not rels:
                continue
            n = len(rels)
            mean = sum(rels) / n
            var = sum((r - mean) ** 2 for r in rels) / (n - 1) if n > 1 else 0.0
            sd = var ** 0.5
            out[str(arm)] = {
                "n_trades": n,
                "mean_rel_return": mean,
                "hit_rate": sum(1 for r in rels if r > 0) / n,
                "std": sd,
                # With few trades this is noise; reported so the caller can see that
                # rather than infer significance from a mean alone.
                "t_stat": mean / (sd / n ** 0.5) if sd > 0 else float("nan"),
            }
        return out
