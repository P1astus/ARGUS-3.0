"""Pre-flight assertions that gate every CPT/SFT run.

Rationale: a full CPT pass over the Path A corpus is a multi-day, unattended job. The
failure mode that costs the most is not a crash -- it is a run that completes having
adapted a different parameter set than intended, which is silent and only detectable
after the fact.

Two directions of that failure are live for Qwen3.6-35B-A3B:

  * Under-adaptation. ml-explore/mlx-lm#571 reported LoRA silently skipping MoE expert
    MLPs, leaving adapters on attention projections only.
  * Over-adaptation. mlx-lm's ``linear_to_lora_layers`` auto-discovers every Linear /
    SwitchLinear / Embedding when ``keys`` is unset. On a 40-layer, 256-expert MoE this
    attaches adapters to the full expert stack, which inflates both step time and memory
    far past what a 64GB machine can sustain.

``assert_trainable`` makes the actual, post-construction parameter set an explicit,
checkable contract instead of an assumption.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import mlx.nn as nn
from mlx.utils import tree_flatten


# Modules whose adaptation is intentional for attention-only CPT.
ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")

# Modules that indicate expert adaptation. Matching any of these when the config asked
# for attention-only means mlx-lm auto-discovery pulled in the MoE stack.
EXPERT_MARKERS = ("switch_mlp", "experts", "gate_proj", "up_proj", "down_proj")

# Router/gate weights. Adapting these destabilises expert load balance and is hard to
# diagnose from a loss curve, so it is opt-in only.
ROUTER_MARKERS = ("gate.weight", "router")


@dataclass
class TrainableReport:
    total_params: int
    trainable_params: int
    by_module: dict[str, int] = field(default_factory=dict)
    adapted_paths: list[str] = field(default_factory=list)

    @property
    def trainable_pct(self) -> float:
        """Trainable share of *stored elements*, not of logical parameters.

        Quantised layers keep their weights packed into uint32, so ``.size`` counts
        storage elements rather than parameters -- a 4-bit 35B model reports ~5.4B here.
        The ratio is still the right thing to bound (it is stable for a given
        quantisation and adapter config), but it must not be read as "% of 35B".
        """
        return 100.0 * self.trainable_params / max(self.total_params, 1)

    def render(self) -> str:
        lines = [
            "trainable parameter report",
            f"  stored elems: {self.total_params:>15,}  (packed; not logical param count)",
            f"  trainable   : {self.trainable_params:>15,}  ({self.trainable_pct:.4f}% of stored)",
            "  by module kind:",
        ]
        for kind, n in sorted(self.by_module.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {kind:<24s} {n:>15,}")
        return "\n".join(lines)


def _module_kind(path: str) -> str:
    """Collapse a parameter path to a comparable module kind.

    ``language_model.model.layers.31.self_attn.q_proj.lora_a`` -> ``self_attn.q_proj``
    """
    parts = [p for p in path.split(".") if not re.fullmatch(r"\d+", p)]
    parts = [p for p in parts if not p.startswith("lora")]
    return ".".join(parts[-2:]) if len(parts) >= 2 else path


def collect(model: nn.Module) -> TrainableReport:
    """Build a report of what is actually trainable on a constructed model."""
    total = sum(v.size for _, v in tree_flatten(model.parameters()))
    trainable = tree_flatten(model.trainable_parameters())

    by_module: dict[str, int] = defaultdict(int)
    paths: list[str] = []
    n_trainable = 0
    for path, value in trainable:
        n_trainable += value.size
        by_module[_module_kind(path)] += value.size
        paths.append(path)

    return TrainableReport(
        total_params=total,
        trainable_params=n_trainable,
        by_module=dict(by_module),
        adapted_paths=paths,
    )


def assert_trainable(
    model: nn.Module,
    *,
    expect_experts: bool,
    expect_router: bool = False,
    min_pct: float = 0.001,
    # Ceiling raised from 5% to 15%. "LoRA Learns Less and Forgets Less" (arXiv
    # 2405.09673) finds that standard low ranks substantially underperform full
    # finetuning for CONTINUED PRETRAINING, and that rank ~256 is needed to approach it
    # (at 20B tokens, r=256 scored 0.617 vs full-FT 0.545 on code CPT). Rank 256 on this
    # 14B measures 10.5% of stored elements, which is a legitimate domain-adaptation
    # configuration rather than a misconfiguration -- the old 5% ceiling would block it.
    max_pct: float = 15.0,
) -> TrainableReport:
    """Fail loudly when the adapted parameter set is not what the config asked for.

    Args:
        model: the model after ``linear_to_lora_layers`` has been applied.
        expect_experts: whether MoE expert MLPs are meant to carry adapters. Set False
            for the attention-only recipe; a violation then means auto-discovery pulled
            in the expert stack and the run would be far slower and larger than planned.
        expect_router: whether router/gate weights are meant to be trainable. Almost
            always False -- adapting the router destabilises expert balance.
        min_pct: floor on trainable share. Guards the mlx-lm#571 direction, where
            adapters silently fail to attach at all.
        max_pct: ceiling on trainable share.

    Returns:
        The report, so callers can log it alongside the run manifest.

    Raises:
        RuntimeError: on any violation. This is deliberately fatal -- these checks exist
            to stop a multi-day run before it starts, not to warn during it.
    """
    report = collect(model)
    problems: list[str] = []

    if report.trainable_params == 0:
        problems.append("no trainable parameters -- adapters did not attach (see mlx-lm#571)")

    if not (min_pct <= report.trainable_pct <= max_pct):
        problems.append(
            f"trainable share {report.trainable_pct:.4f}% outside "
            f"[{min_pct}%, {max_pct}%] -- adapter target is likely misconfigured"
        )

    has_expert = any(
        any(m in p for m in EXPERT_MARKERS) for p in report.adapted_paths
    )
    if has_expert and not expect_experts:
        offenders = sorted({_module_kind(p) for p in report.adapted_paths
                            if any(m in p for m in EXPERT_MARKERS)})
        problems.append(
            "MoE expert modules carry adapters but expect_experts=False -- mlx-lm "
            f"auto-discovery pulled in the expert stack: {offenders}. "
            "Set an explicit `keys` list in the LoRA config."
        )
    if expect_experts and not has_expert:
        problems.append(
            "expect_experts=True but no expert modules were adapted -- this is the "
            "mlx-lm#571 silent no-op"
        )

    has_router = any(
        any(m in p for m in ROUTER_MARKERS) for p in report.adapted_paths
    )
    if has_router and not expect_router:
        problems.append(
            "router/gate weights are trainable but expect_router=False -- adapting the "
            "router destabilises expert load balance"
        )

    has_attention = any(
        any(t in p for t in ATTENTION_TARGETS) for p in report.adapted_paths
    )
    if not has_attention:
        problems.append("no attention projections adapted -- unexpected for any recipe")

    if problems:
        raise RuntimeError(
            "pre-flight FAILED; refusing to start run\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n\n"
            + report.render()
        )

    return report
