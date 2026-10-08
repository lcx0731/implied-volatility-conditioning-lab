import math

import mpmath as mp
import numpy as np
import pytest
from scipy.special import ndtr

from iv_audit.model import call_price, log_otm_value, price_bounds, vega


@pytest.mark.parametrize("forward,strike,maturity,volatility,discount", [
    (100, 100, 1, 0.2, 0.95), (100, 130, 0.005, 0.25, 0.999),
    (100, 120, 1 / 252, 0.25, 1), (100, 85, 0.5, 0.4, 0.97),
    (0.01, 0.012, 0.1, 0.3, 0.99), (100, 100, 2, 3, 1),
])
def test_call_matches_independent_high_precision_formula(forward, strike, maturity, volatility, discount):
    with mp.workdps(100):
        f, k, t, sigma, d = map(mp.mpf, (forward, strike, maturity, volatility, discount))
        total = sigma * mp.sqrt(t)
        d1, d2 = mp.log(f / k) / total + total / 2, mp.log(f / k) / total - total / 2
        cdf = lambda x: mp.erfc(-x / mp.sqrt(2)) / 2
        reference = float(d * (f * cdf(d1) - k * cdf(d2)))
    actual = call_price(forward, strike, maturity, volatility, discount)
    assert actual == pytest.approx(reference, rel=2e-9, abs=0)


def test_at_the_money_closed_form():
    expected = 100 * (2 * ndtr(0.1) - 1)
    assert call_price(100, 100, 1, 0.2) == pytest.approx(expected, abs=1e-12)


def test_intrinsic_at_zero_volatility_and_expiry():
    assert call_price(100, 90, 1, 0, 0.95) == pytest.approx(9.5)
    assert call_price(100, 90, 0, 1, 0.95) == pytest.approx(9.5)
    assert call_price(100, 110, 0, 0.2) == 0
    assert vega(100, 100, 0, 0.2) == 0
    assert vega(100, 100, 1, 0) == pytest.approx(100 / math.sqrt(2 * math.pi))
    assert vega(100, 90, 1, 0) == 0


@pytest.mark.parametrize("strike,maturity", [(80, 1), (100, 1 / 252), (120, 0.5)])
def test_vega_against_central_difference(strike, maturity):
    sigma, step, discount = 0.25, 1e-6, 0.97
    difference = (call_price(100, strike, maturity, sigma + step, discount)
                  - call_price(100, strike, maturity, sigma - step, discount)) / (2 * step)
    assert vega(100, strike, maturity, sigma, discount) == pytest.approx(difference, rel=1e-6)


def test_price_is_monotone_in_volatility_and_bounded():
    values = [call_price(100, 105, 0.5, sigma, 0.98) for sigma in np.linspace(0.05, 2, 30)]
    low, high = price_bounds(100, 105, 0.98)
    assert np.all(np.diff(values) > 0)
    assert all(low <= p <= high for p in values)


def test_itm_total_price_can_lose_representable_otm_time_value():
    total_std = 0.25 * math.sqrt(1 / 252)
    log_value = log_otm_value(100, 70, total_std)
    assert math.isfinite(log_value) and math.exp(log_value) > 0
    assert call_price(100, 70, 1 / 252, 0.25) == price_bounds(100, 70)[0]


@pytest.mark.parametrize("args", [(0, 100, 1, 0.2), (100, -1, 1, 0.2),
    (100, 100, -1, 0.2), (100, 100, 1, -1), (100, 100, np.nan, 0.2),
    (100, 100, 1, np.inf), (True, 100, 1, 0.2)])
def test_invalid_pricing_inputs(args):
    with pytest.raises(ValueError):
        call_price(*args)


def test_invalid_discount_and_degenerate_bounds():
    with pytest.raises(ValueError):
        price_bounds(100, 100, 0)
    with pytest.raises(ValueError, match="distinct"):
        price_bounds(100, 1e-30)
