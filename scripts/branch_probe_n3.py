#!/usr/bin/env python3
"""
Is a failure band a region where NO pure profile is an equilibrium?

Take the pure equilibrium profile from just below a failure band and the one from
just above it, hold their STRATEGIES fixed, and re-verify each at deltas INSIDE
the band.  If both fail throughout, the band is not a search failure: no pure
branch survives there and the true equilibrium must mix.  If one verifies, VFI
simply missed a pure equilibrium that exists.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib.verify_cli import run_verification

ATOL = 1e-12

CASES = [
    ("eurrususa", 0.92, 0.99, [0.925, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985]),
    ("chnrususa", 0.92, 0.98, [0.94, 0.95, 0.96, 0.97]),
    ("chneurrus", 0.88, 0.999, [0.9, 0.92, 0.95, 0.97, 0.99]),
    ("chneurnde", 0.885, None, [0.9, 0.92, 0.95, 0.99]),
    ("nderususa", 0.985, None, [0.99, 0.995, 0.999]),
]


def find_profile(trio: str, delta: float) -> Path | None:
    for d in ("reports/delta_continuation/profiles",
              "reports/delta_homotopy/profiles",
              "reports/warm_start_experiments/profiles"):
        for pat in (f"{trio}_d{delta:g}.xlsx",
                    f"{trio}_d{delta}_atol1e-12_s42.xlsx",
                    f"{trio}_d{delta}_atol1e-12.xlsx"):
            f = REPO / d / pat
            if f.exists():
                return f
    return None


def main():
    print("Do the neighbouring PURE branches survive inside the failure band?")
    print("(strategies held fixed, only delta changed)\n")
    for trio, lo, hi, inside in CASES:
        print(trio)
        srcs = []
        for tag, d in (("below", lo), ("above", hi)):
            if d is None:
                continue
            f = find_profile(trio, d)
            if f is None:
                print(f"  ({tag} profile at delta={d} not found)")
            else:
                srcs.append((tag, d, f))
        for d in inside:
            marks = []
            for tag, src_d, f in srcs:
                try:
                    ok, _, _ = run_verification(f, atol=ATOL, quiet=True,
                                                discounting=d)
                except Exception as exc:
                    ok = f"ERR:{exc}"
                marks.append(f"{tag}({src_d:g})={'VERIFIES' if ok is True else 'no'}")
            print(f"  delta={d:<7} " + "   ".join(marks))
        print()


if __name__ == "__main__":
    main()
