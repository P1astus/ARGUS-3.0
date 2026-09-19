"""Feature library.

Importing this package registers every feature, including the v2 orthogonal set.
"""

from argus.quant.features import registry  # noqa: F401  (registers core features)
from argus.quant.features import orthogonal  # noqa: F401  (registers v2 additions)
from argus.quant.features.orthogonal import FEATURES_V2  # noqa: F401
from argus.quant.features.registry import DEFAULT_FEATURES, available, get, normalize  # noqa: F401
