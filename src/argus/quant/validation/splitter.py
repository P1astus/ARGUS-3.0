"""Purged walk-forward cross-validation.

THE PROBLEM
A label at date t is computed from prices at t and t+h. A training sample at t-3 with
h=10 therefore *contains information about* the same future window as a test sample at
t+2. Plain time-series CV -- train on everything before the split, test on everything
after -- leaks through this overlap and produces optimistic ICs that evaporate live.

THE FIX (Lopez de Prado): two distinct operations, both required.

  PURGE   drop training samples whose LABEL WINDOW [t, t+h] overlaps the test window.
          Without this, training labels are partly computed from test-period prices.

  EMBARGO drop training samples in the [gap] immediately AFTER the test window. Serial
          correlation in features means a sample just after the test period still carries
          information about it. Purging alone does not remove this.

Both default to the label horizon. `from_label_spec` wires them directly from the
LabelSpec so a splitter cannot silently be configured with a shorter embargo than the
labels it is splitting -- the failure mode that makes leakage invisible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from argus.quant.labels import LabelSpec


@dataclass(frozen=True)
class Fold:
    """One walk-forward split, expressed as positional indices into the date axis."""

    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    n_purged: int
    n_embargoed: int

    def __repr__(self) -> str:
        return (f"Fold({self.fold_id}: train {len(self.train_idx)} "
                f"[{self.train_dates[0].date()}..{self.train_dates[-1].date()}], "
                f"test {len(self.test_idx)} "
                f"[{self.test_dates[0].date()}..{self.test_dates[-1].date()}], "
                f"purged {self.n_purged}, embargoed {self.n_embargoed})")


class PurgedWalkForward:
    """Expanding-window walk-forward CV with purge and embargo.

    Expanding rather than rolling: markets change slowly enough that older data still
    carries signal, and discarding the 2011 capex bust or the 2018 memory downturn would
    remove exactly the regimes the model most needs to have seen.
    """

    def __init__(
        self,
        n_splits: int = 5,
        horizon: int = 10,
        embargo: int | None = None,
        min_train_dates: int = 252,
    ) -> None:
        self.n_splits = n_splits
        self.horizon = horizon
        # Embargo defaults to the horizon. Shorter would leave serially-correlated
        # samples adjacent to the test window in training.
        self.embargo = horizon if embargo is None else embargo
        self.min_train_dates = min_train_dates

        if self.embargo < horizon:
            raise ValueError(
                f"embargo ({self.embargo}) < horizon ({horizon}): training samples "
                "immediately after the test window would still overlap it")

    @classmethod
    def from_label_spec(cls, spec: LabelSpec, n_splits: int = 5, **kw) -> PurgedWalkForward:
        """Construct with horizon and embargo tied to the labels being split."""
        return cls(n_splits=n_splits, horizon=spec.horizon_days, **kw)

    def split(self, dates: pd.DatetimeIndex) -> Iterator[Fold]:
        """Yield folds over a sorted, unique date index."""
        dates = pd.DatetimeIndex(dates).sort_values().unique()
        n = len(dates)
        if n < self.min_train_dates + self.n_splits * 2:
            raise ValueError(
                f"{n} dates is too few for {self.n_splits} folds with "
                f"min_train_dates={self.min_train_dates}")

        # Equal-sized contiguous test blocks over the tail of the sample.
        test_size = (n - self.min_train_dates) // self.n_splits

        for k in range(self.n_splits):
            test_start = self.min_train_dates + k * test_size
            test_end = test_start + test_size if k < self.n_splits - 1 else n
            test_idx = np.arange(test_start, test_end)

            # Candidate training set: everything outside the test block.
            candidate = np.concatenate([np.arange(0, test_start), np.arange(test_end, n)])

            # PURGE: a training date t leaks if its label window [t, t+horizon] reaches
            # into the test block. For dates before the block that means the last
            # `horizon` of them.
            purge_lo = max(test_start - self.horizon, 0)
            purged_mask = (candidate >= purge_lo) & (candidate < test_start)

            # EMBARGO: drop the `embargo` dates immediately after the test block.
            embargo_hi = min(test_end + self.embargo, n)
            embargo_mask = (candidate >= test_end) & (candidate < embargo_hi)

            keep = ~(purged_mask | embargo_mask)
            train_idx = candidate[keep]

            if len(train_idx) == 0:
                continue

            yield Fold(
                fold_id=k,
                train_idx=train_idx,
                test_idx=test_idx,
                train_dates=dates[train_idx],
                test_dates=dates[test_idx],
                n_purged=int(purged_mask.sum()),
                n_embargoed=int(embargo_mask.sum()),
            )

    def assert_no_leakage(self, dates: pd.DatetimeIndex) -> None:
        """Verify no training label window intersects its test window.

        This restates the guarantee independently of the construction above, so a future
        refactor of the index arithmetic cannot quietly reintroduce leakage. Called by the
        test suite and cheap enough to call before a real run.
        """
        dates = pd.DatetimeIndex(dates).sort_values().unique()
        for fold in self.split(dates):
            test_lo, test_hi = fold.test_idx.min(), fold.test_idx.max()
            for t in fold.train_idx:
                label_lo, label_hi = t, t + self.horizon
                if label_hi >= test_lo and label_lo <= test_hi:
                    raise AssertionError(
                        f"fold {fold.fold_id}: training date index {t} has label window "
                        f"[{label_lo},{label_hi}] overlapping test [{test_lo},{test_hi}]")
