"""The acceptance gate (spec §6.6, tightened in P7): keep a patch only if the dev gain beats the measured noise.

A case is *fixed* if it failed before and passes now, *regressed* if it passed before and fails now. The gate accepts
only if ``fixed - regressed`` is larger than the noise floor (how many dev cases flip pass/fail when the SAME agent is
re-run without any cache) AND at most one case regressed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateResult:
    accepted: bool
    fixed: list[str]
    regressed: list[str]
    still_failing: list[str]
    net_gain: int
    noise_floor: int
    reason: str


def acceptance_gate(before: dict[str, bool], after: dict[str, bool], noise_floor: int, max_regressions: int = 1) -> GateResult:
    """``before`` and ``after`` map case id to passed."""
    ids = sorted(before)
    fixed = [c for c in ids if not before[c] and after.get(c, False)]
    regressed = [c for c in ids if before[c] and not after.get(c, False)]
    still = [c for c in ids if not before[c] and not after.get(c, False)]
    net = len(fixed) - len(regressed)
    if len(regressed) > max_regressions:
        reason = f"{len(regressed)} cases regressed (at most {max_regressions} allowed)"
    elif net <= noise_floor:
        reason = f"net gain {net} does not exceed the noise floor {noise_floor}"
    else:
        reason = f"net gain {net} exceeds the noise floor {noise_floor} with {len(regressed)} regression(s)"
    return GateResult(net > noise_floor and len(regressed) <= max_regressions, fixed, regressed, still, net, noise_floor, reason)


def confirm_gate(first: GateResult, confirm_passed: dict[str, bool], noise_floor: int, max_regressions: int = 1) -> GateResult:
    """Re-judge a first-run verdict after the confirmation rerun (uncached) of ONLY the fixed and regressed cases.

    A fix counts only if the case passes on both runs. Regressions are kept as the first run found them (a regression
    that passes on the rerun is not forgiven: the stricter reading). The result says ``unconfirmed`` if the patch
    cleared the gate on the first run but not after confirmation."""
    held = [c for c in first.fixed if confirm_passed.get(c, False)]
    dropped = [c for c in first.fixed if c not in held]
    net = len(held) - len(first.regressed)
    ok = net > noise_floor and len(first.regressed) <= max_regressions
    if ok:
        reason = f"confirmed: {len(held)} of {len(first.fixed)} fixes held on the rerun; net gain {net} exceeds the noise floor {noise_floor}"
    else:
        reason = (f"rejected: unconfirmed ({len(held)} of {len(first.fixed)} fixes held on the uncached rerun; net gain {net} "
                  f"does not exceed the noise floor {noise_floor} with {len(first.regressed)} regression(s))")
    return GateResult(ok, held, first.regressed, sorted(first.still_failing + dropped), net, noise_floor, reason)


def noise_flips(first: dict[str, bool], second: dict[str, bool]) -> list[str]:
    """Cases whose verdict differs between two runs of the same agent: the noise floor is how many there are."""
    return sorted(c for c in first if c in second and first[c] != second[c])
