"""Deterministic synthetic quote quantization, not market or trading evidence."""

import argparse
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from .inverse import implied_volatility, volatility_interval
from .model import call_price, price_bounds, vega

MATURITIES = (1 / 252, 5 / 252, 21 / 252, 0.5, 1.0, 2.0)
STRIKE_RATIOS = (0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3)


def run_audit(tick=0.01):
    if isinstance(tick, bool) or not np.isscalar(tick) or not np.isfinite(tick) or not 1e-6 <= tick <= 1:
        raise ValueError("this fixed-grid experiment supports tick sizes from 1e-6 to 1")
    forward, sigma = 100.0, 0.25
    rows = []
    for maturity in MATURITIES:
        discount = float(np.exp(-0.03 * maturity))
        for ratio in STRIKE_RATIOS:
            strike = forward * ratio
            price = call_price(forward, strike, maturity, sigma, discount)
            quote = float(np.rint(price / tick) * tick)
            lower, upper = price_bounds(forward, strike, discount)
            full_precision_iv = implied_volatility(price, forward, strike, maturity, discount)
            if lower <= quote <= upper:
                point = asdict(implied_volatility(quote, forward, strike, maturity, discount))
            else:
                point = {"status": "outside_bounds", "volatility": None}
            interval = volatility_interval(quote - tick / 2, quote + tick / 2,
                                            forward, strike, maturity, discount)
            if interval.volatility_lower > sigma + 1e-8 or (
                    interval.volatility_upper is not None and interval.volatility_upper < sigma - 1e-8):
                raise ArithmeticError("quote-consistent range excludes the generating volatility")
            if interval.volatility_upper is None:
                width = None
            else:
                width = interval.volatility_upper - interval.volatility_lower
            sensitivity = vega(forward, strike, maturity, sigma, discount)
            rows.append({"maturity_years": maturity, "strike_ratio": ratio,
                         "forward": forward, "strike": strike, "discount": discount,
                         "true_volatility": sigma, "unrounded_price": price, "rounded_quote": quote,
                         "intrinsic_price": lower, "price_upper_bound": upper,
                         "full_precision_inverse": asdict(full_precision_iv), "quote_point_inverse": point,
                         "quote_consistent_interval": asdict(interval), "interval_width": width,
                         "true_vega": sensitivity,
                         "local_linear_width": tick / sensitivity if sensitivity > 1e-300 else None})
    interior = [r for r in rows if r["quote_point_inverse"]["status"] == "interior"]
    exact_interior = [r for r in rows if r["full_precision_inverse"]["status"] == "interior"]
    return {
        "schema_version": 1,
        "data": "Deterministic synthetic forward-Black calls; no market data or trading results.",
        "configuration": {"forward": forward, "true_volatility": sigma, "discount_rate": 0.03,
                          "tick": tick, "quote_half_width": tick / 2,
                          "maturities_years": list(MATURITIES), "strike_ratios": list(STRIKE_RATIOS)},
        "software": {name: version(name) for name in ("numpy", "scipy", "matplotlib")},
        "checks": {"number_of_options": len(rows), "true_volatility_in_all_ranges": True,
                   "quote_point_status_counts": dict(Counter(r["quote_point_inverse"]["status"] for r in rows)),
                   "full_precision_status_counts": dict(Counter(r["full_precision_inverse"]["status"] for r in rows)),
                   "max_interior_quote_repricing_error": max((abs(r["quote_point_inverse"]["repricing_error"]) for r in interior), default=None),
                   "max_interior_quote_volatility_error": max((abs(r["quote_point_inverse"]["volatility"] - sigma) for r in interior), default=None),
                   "max_full_precision_interior_volatility_error": max(abs(r["full_precision_inverse"]["volatility"] - sigma) for r in exact_interior)},
        "options": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--tick", type=float, default=0.01)
    args = parser.parse_args()
    summary = run_audit(args.tick)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    from .plotting import plot_audit
    plot_audit(summary, args.output)
    print(json.dumps(summary["checks"], indent=2))


if __name__ == "__main__":
    main()
