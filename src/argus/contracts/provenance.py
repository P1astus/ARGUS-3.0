"""Run provenance -- what produced a given output.

WHY THIS IS NOT OPTIONAL
The Phase 3 gate asks whether the trained model's reasoning adds value. Answering that
later requires knowing, for every recorded recommendation, exactly which artefacts
produced it: which base model, which adapter, which corpus, which config, which arm.

None of it can be reconstructed after the fact. A journal full of results with no
provenance can show that something worked without being able to say what.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class Arm(StrEnum):
    """Which pipeline produced an output -- the unit of comparison in the Phase 3 gate."""

    QUANT_ONLY = "quant_only"      # Stage 1 score alone
    BASE_PROMPT = "base_prompt"    # base model + prompt + tools, NO continued pretraining
    CPT_SFT = "cpt_sft"            # the trained engine
    HUMAN = "human"                # discretionary, for comparison


class RunProvenance(BaseModel):
    """Immutable record of what produced an output."""

    arm: Arm
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    base_model: str | None = None
    adapter_path: str | None = None
    adapter_sha256: str | None = None

    corpus_manifest_sha256: str | None = None
    corpus_cutoff: str | None = Field(
        default=None,
        description="Corpus hard date cutoff. Eval setups must postdate this, so it is "
                    "recorded alongside every output rather than assumed.")

    config_sha256: str | None = None
    code_git_sha: str | None = None

    quant_model_version: str | None = None

    def key(self) -> str:
        """Short stable identifier for grouping outputs from the same configuration."""
        payload = json.dumps({
            "arm": str(self.arm),
            "base_model": self.base_model,
            "adapter": self.adapter_sha256,
            "corpus": self.corpus_manifest_sha256,
            "config": self.config_sha256,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_obj(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def git_sha(default: str | None = None) -> str | None:
    """Current commit, or None outside a repo.

    Best-effort: the project may not be under version control yet, and provenance should
    degrade rather than fail.
    """
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or default if out.returncode == 0 else default
    except Exception:
        return default
