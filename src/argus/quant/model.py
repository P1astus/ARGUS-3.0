"""LightGBM cross-sectional ranker.

Deliberately small and heavily regularised. The panel is modest (~100 names x ~15 years of
weekly rebalances is tens of thousands of rows, not millions) and the signal-to-noise ratio
in cross-sectional equity returns is low. A deep GBDT will memorise it perfectly and
generalise not at all -- depth 3-5 with strong regularisation is not timidity, it is the
only configuration that has a chance of surviving out of sample.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Hyperparameters. Defaults chosen for a small, noisy panel."""

    objective: str = "regression"       # "lambdarank" available via RankerModel
    num_leaves: int = 15                # ~depth 4
    max_depth: int = 4
    learning_rate: float = 0.03
    n_estimators: int = 300
    min_child_samples: int = 100        # large: prevents leaves fitting a handful of names
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.7
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = -1

    def to_lgb(self) -> dict:
        d = asdict(self)
        return d


class QuantRanker:
    """Wraps LightGBM with the cross-sectional conventions this project needs."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.model: lgb.LGBMRegressor | None = None
        self.feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> QuantRanker:
        """Fit on long-format (date, ticker) rows.

        The label is the forward relative return. We regress on it rather than on its
        within-date rank: the raw label already carries magnitude information that a rank
        would discard, and the evaluation metric (rank IC) re-imposes ordering anyway.
        """
        self.feature_names = list(X.columns)
        self.model = lgb.LGBMRegressor(**self.config.to_lgb())
        self.model.fit(X.to_numpy(), y.to_numpy(), feature_name=self.feature_names)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        preds = self.model.predict(X.to_numpy())
        return pd.Series(preds, index=X.index, name="score")

    def predict_wide(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predictions as a wide (date x ticker) frame, which the metrics expect."""
        return self.predict(X).unstack("ticker")

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        return pd.Series(
            self.model.feature_importances_, index=self.feature_names, name="importance"
        ).sort_values(ascending=False)


class LinearRanker:
    """Ridge regression on cross-sectionally normalised features.

    WHY A LINEAR MODEL
    Measured in-sample vs out-of-sample IC showed the GBDT fitting training data 10-40x
    better than it generalised (v1 gap 0.180, v2 gap 0.241). The cause is capacity: the
    panel looks like ~31k rows, but rows sharing a date are one correlated cross-section,
    so the effective sample is ~826 weekly observations. A 300-tree GBDT has thousands of
    parameters against that.

    Ridge with a handful of features has ~6. It cannot represent interactions or
    non-linearity -- which, at this sample size, is the point.
    """

    def __init__(self, alpha: float = 10.0) -> None:
        # Deliberately strong regularisation. With ~826 effective observations the prior
        # should dominate unless the data argues loudly otherwise.
        self.alpha = alpha
        self.model = None
        self.feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearRanker":
        from sklearn.linear_model import Ridge

        self.feature_names = list(X.columns)
        self.model = Ridge(alpha=self.alpha, fit_intercept=True)
        self.model.fit(X.to_numpy(), y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        return pd.Series(self.model.predict(X.to_numpy()), index=X.index, name="score")

    def predict_wide(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.predict(X).unstack("ticker")

    def feature_importance(self) -> pd.Series:
        """Absolute standardised coefficients.

        Features are rank-normalised to a common [0,1] scale before fitting, so
        coefficient magnitudes are directly comparable.
        """
        if self.model is None:
            raise RuntimeError("model is not fitted")
        return pd.Series(
            np.abs(self.model.coef_), index=self.feature_names, name="importance"
        ).sort_values(ascending=False)


# Deliberately small, low-collinearity feature set. Selected from the measured v1
# correlation matrix (pairwise |rho| <= 0.25 among the first four) plus the one genuinely
# new axis from v2 -- fixed BEFORE running, not chosen by trying combinations.
FEATURES_CORE: tuple[str, ...] = (
    "momentum_12_1",
    "momentum_20d",
    "volatility_20d",
    "volume_trend",
    "subsegment_rel_20d",
)


@dataclass
class FoldResult:
    fold_id: int
    predictions: pd.DataFrame
    labels: pd.DataFrame
    train_dates: tuple[str, str]
    test_dates: tuple[str, str]
    n_train: int
    n_test: int
    importance: pd.Series = field(repr=False, default_factory=pd.Series)
