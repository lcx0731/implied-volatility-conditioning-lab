"""Forward-Black European call, with log-domain OTM time value.

Not a production pricing library. All prices are per one unit of underlying.
"""

import math

import numpy as np
from scipy.special import log_ndtr


def finite_scalar(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    if (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"invalid {name}")
    return float(value)


def price_bounds(forward, strike, discount=1.0):
    forward = finite_scalar(forward, "forward", positive=True)
    strike = finite_scalar(strike, "strike", positive=True)
    discount = finite_scalar(discount, "discount", positive=True)
    lower, upper = discount * max(forward - strike, 0), discount * forward
    if not np.isfinite(upper) or lower == upper:
        raise ValueError("price bounds are not distinct finite float64 values; rescale inputs")
    return lower, upper


def log_otm_value(forward, strike, total_std):
    """Undiscounted OTM call or put value; arguments validated by public APIs.

    Use the put when F>K and add intrinsic value later. log_ndtr avoids CDF
    underflow; expm1 reduces subtraction loss between the two positive terms.
    If the log terms coincide at extreme scales, return -inf (unresolved tail),
    not an invented positive price. Tests validate the experiment's range.
    """
    if total_std == 0:
        return -math.inf
    m = math.log(forward) - math.log(strike)
    d1 = m / total_std + 0.5 * total_std
    d2 = m / total_std - 0.5 * total_std
    if forward <= strike:
        first = math.log(forward) + float(log_ndtr(d1))
        second = math.log(strike) + float(log_ndtr(d2))
    else:
        first = math.log(strike) + float(log_ndtr(-d2))
        second = math.log(forward) + float(log_ndtr(-d1))
    difference = second - first
    if difference >= 0:
        if difference > 32 * np.finfo(float).eps * max(1, abs(first)):
            raise ArithmeticError("OTM subtraction lost its positive ordering")
        return -math.inf
    return first + math.log(-math.expm1(difference))


def call_price(forward, strike, maturity, volatility, discount=1.0):
    lower, upper = price_bounds(forward, strike, discount)
    maturity = finite_scalar(maturity, "maturity", nonnegative=True)
    volatility = finite_scalar(volatility, "volatility", nonnegative=True)
    total_std = volatility * math.sqrt(maturity)
    if not np.isfinite(total_std):
        raise ValueError("total standard deviation overflows")
    if total_std == 0:
        return lower
    time_value = math.exp(log_otm_value(forward, strike, total_std))
    price = lower + discount * time_value
    if price > upper + 8 * np.finfo(float).eps * upper:
        raise ArithmeticError("computed price exceeds its upper bound")
    return min(price, upper)  # At most roundoff-sized upper-bound correction.


def vega(forward, strike, maturity, volatility, discount=1.0):
    """d price / d annual volatility (1.0 = 100 vol percentage points)."""
    price_bounds(forward, strike, discount)
    maturity = finite_scalar(maturity, "maturity", nonnegative=True)
    volatility = finite_scalar(volatility, "volatility", nonnegative=True)
    if maturity == 0:
        return 0.0
    if volatility == 0:
        return discount * forward * math.sqrt(maturity / (2 * math.pi)) if forward == strike else 0.0
    d1 = (math.log(forward) - math.log(strike)) / (volatility * math.sqrt(maturity)) + 0.5 * volatility * math.sqrt(maturity)
    return discount * forward * math.sqrt(maturity) * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
