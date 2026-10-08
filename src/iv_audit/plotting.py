"""Static scientific figures; no external images or data."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

from .model import call_price


def plot_audit(summary, output: Path):
    config = summary["configuration"]
    maturities, ratios = config["maturities_years"], config["strike_ratios"]
    widths = np.array([r["interval_width"] for r in summary["options"]], dtype=float).reshape(len(maturities), len(ratios)) * 100
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(9.5, 5.3), layout="constrained")
        finite = widths[np.isfinite(widths) & (widths > 0)]
        im = ax.imshow(np.ma.masked_invalid(widths), cmap="YlOrRd", aspect="auto",
                       norm=LogNorm(vmin=finite.min(), vmax=finite.max()))
        for i in range(len(maturities)):
            for j in range(len(ratios)):
                value = widths[i, j]
                text_color = "white" if np.isfinite(value) and im.norm(value) > 0.65 else "black"
                ax.text(j, i, f"{value:.2f}" if np.isfinite(value) else "unbounded",
                        ha="center", va="center", color=text_color, fontsize=10)
        ax.set_xticks(range(len(ratios)), [f"{r:.2f}" for r in ratios])
        ax.set_yticks(range(len(maturities)), ["1 trading day", "5 trading days", "21 trading days", "0.5 year", "1 year", "2 years"])
        ax.set_xlabel("Strike / forward")
        ax.set_ylabel("Time to maturity")
        ax.set_title(f"Synthetic quotes rounded to {config['tick']:g}: model-consistent IV range width\nVolatility percentage points (color scale is logarithmic)")
        fig.colorbar(im, ax=ax, label="Range width (vol percentage points)")
        fig.savefig(output / "quote-resolution-map.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
        for ax, ratio in zip(axes, (1.0, 1.15)):
            record = next(r for r in summary["options"] if r["maturity_years"] == maturities[0] and r["strike_ratio"] == ratio)
            interval = record["quote_consistent_interval"]
            low, high = interval["volatility_lower"], interval["volatility_upper"]
            # These two default-grid cases have finite upper ranges.
            high = high if high is not None else 1.0
            padding = max((high - low) * 0.3, 0.002)
            vol_grid = np.linspace(max(0, low - padding), high + padding, 250)
            prices = [call_price(record["forward"], record["strike"], record["maturity_years"], v, record["discount"]) for v in vol_grid]
            ax.plot(vol_grid * 100, prices, color="#007e87", label="Model call price")
            ax.axhspan(interval["feasible_price_lower"], interval["feasible_price_upper"],
                       color="#778293", alpha=0.25, label="Feasible price band")
            ax.axvspan(low * 100, high * 100, color="#d7a741", alpha=0.15, label="Consistent IV range")
            ax.axvline(record["true_volatility"] * 100, color="#a84432", linestyle="--", label="Generating 25% IV")
            point = record["quote_point_inverse"]["volatility"]
            if point is not None:
                ax.scatter(point * 100, record["rounded_quote"], color="#333333", zorder=4, s=25, label="Rounded-quote inverse")
            ax.set_xlabel("Annual volatility (%)")
            ax.set_ylabel("Call price per unit of underlying")
            ax.set_title(f"1 trading day • strike / forward = {ratio:g}")
            lo_price, hi_price = interval["feasible_price_lower"], interval["feasible_price_upper"]
            ax.set_ylim(max(0, lo_price - 0.5 * config["tick"]), hi_price + config["tick"])
            ax.grid(alpha=0.15)
        axes[0].legend(frameon=False, fontsize=8, loc="upper left")
        fig.suptitle("Price precision and volatility precision are different questions", fontsize=13)
        fig.savefig(output / "conditioning-examples.png", dpi=160)
        plt.close(fig)
