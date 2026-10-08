"""Bracketed inversion and monotone quote-interval propagation."""

from dataclasses import dataclass
import math

from scipy.optimize import brentq

from .model import call_price, finite_scalar, log_otm_value, price_bounds, vega


class InversionError(RuntimeError):
    """A root was not bracketed or validated within the numerical budget."""


@dataclass(frozen=True)
class IVResult:
    status: str
    volatility: float | None
    total_std: float | None
    repricing_error: float
    vega: float | None
    iterations: int


@dataclass(frozen=True)
class VolatilityInterval:
    quote_lower: float
    quote_upper: float
    feasible_price_lower: float
    feasible_price_upper: float
    volatility_lower: float
    volatility_upper: float | None
    upper_unbounded: bool


def implied_volatility(price, forward, strike, maturity, discount=1.0, *,
                       max_total_std=64.0, max_iterations=300):
    """Solve in total std = sigma sqrt(T), using a bounded log-price residual.

    Boundary prices are labelled: lower -> zero-volatility limit; upper ->
    infinite-volatility limit (None, never a made-up finite estimate).
    Invalid prices are rejected without tolerance-based silent clipping.
    """
    lower, upper = price_bounds(forward, strike, discount)
    maturity = finite_scalar(maturity, "maturity", positive=True)
    price = finite_scalar(price, "price", nonnegative=True)
    max_total_std = finite_scalar(max_total_std, "max_total_std", positive=True)
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if price < lower or price > upper:
        raise ValueError("call price is outside the individual no-arbitrage bounds")
    if price == lower:
        return IVResult("lower_boundary", 0.0, 0.0, 0.0,
                        vega(forward, strike, maturity, 0, discount), 0)
    if price == upper:
        return IVResult("upper_boundary", None, None, 0.0, None, 0)
    # Subtract discounted intrinsic first, avoiding price/discount rounding
    # a distinguishable price back to intrinsic. ITM information can still be lost
    # in the supplied total call price itself; no solver can recover those bits.
    target_log_value = math.log(price - lower) - math.log(discount)

    def residual(total_std):
        if total_std == 0:
            return -1.0
        log_ratio = log_otm_value(forward, strike, total_std) - target_log_value
        # Equal to (model time value - target)/(model time value + target).
        return math.tanh(0.5 * log_ratio)

    high = min(1.0, max_total_std)
    while residual(high) < 0 and high < max_total_std:
        high = min(2 * high, max_total_std)
    if residual(high) < 0:
        raise InversionError("root exceeds max_total_std; no finite estimate returned")
    if residual(high) == 0:
        # Avoid relying on iteration diagnostics for a root at the initial endpoint.
        total_std, iterations = high, 0
    else:
        try:
            total_std, info = brentq(residual, 0, high, xtol=1e-13, rtol=1e-13,
                                    maxiter=max_iterations, full_output=True, disp=False)
        except (ValueError, RuntimeError) as exc:
            raise InversionError("root solver failed") from exc
        if not info.converged:
            raise InversionError("root solver exhausted its iteration budget")
        iterations = int(info.iterations)
    volatility = total_std / math.sqrt(maturity)
    repricing_error = call_price(forward, strike, maturity, volatility, discount) - price
    if abs(repricing_error) > 1e-10 * max(1.0, upper):
        raise InversionError("repricing validation failed")
    return IVResult("interior", volatility, total_std, repricing_error,
                    vega(forward, strike, maturity, volatility, discount), iterations)


def volatility_interval(quote_lower, quote_upper, forward, strike, maturity, discount=1.0):
    """Map a closed price interval to the model-consistent volatility range.

    Intersect with individual call bounds explicitly; record both original and
    feasible price intervals. Not a confidence interval or cross-strike check.
    """
    lower, upper = price_bounds(forward, strike, discount)
    finite_scalar(maturity, "maturity", positive=True)
    quote_lower = finite_scalar(quote_lower, "quote_lower")
    quote_upper = finite_scalar(quote_upper, "quote_upper")
    if quote_lower > quote_upper:
        raise ValueError("quote interval endpoints are reversed")
    feasible_lower, feasible_upper = max(quote_lower, lower), min(quote_upper, upper)
    if feasible_lower > feasible_upper or feasible_lower == upper:
        raise ValueError("quote interval contains no finite-volatility call price")
    low_iv = implied_volatility(feasible_lower, forward, strike, maturity, discount)
    high_iv = implied_volatility(feasible_upper, forward, strike, maturity, discount)
    return VolatilityInterval(quote_lower, quote_upper, feasible_lower, feasible_upper,
                              low_iv.volatility, high_iv.volatility, high_iv.status == "upper_boundary")
