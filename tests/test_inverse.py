import numpy as np
import pytest

from iv_audit import InversionError, call_price, implied_volatility, price_bounds, vega, volatility_interval


@pytest.mark.parametrize("strike,maturity,sigma", [(80, 1, 0.3), (100, 1, 0.2),
    (115, 0.1, 0.4), (130, 1 / 252, 0.25), (100, 1, 2.0), (100, 1, 1.0)])
def test_round_trip_including_tiny_otm_price(strike, maturity, sigma):
    price = call_price(100, strike, maturity, sigma, 0.97)
    result = implied_volatility(price, 100, strike, maturity, 0.97)
    assert result.status == "interior"
    assert result.volatility == pytest.approx(sigma, abs=2e-10)
    assert result.iterations >= 0
    assert abs(result.repricing_error) < 1e-9


def test_explicit_boundary_statuses():
    low, high = price_bounds(100, 80, 0.95)
    lower = implied_volatility(low, 100, 80, 1, 0.95)
    upper = implied_volatility(high, 100, 80, 1, 0.95)
    assert lower.status == "lower_boundary" and lower.volatility == 0
    assert upper.status == "upper_boundary" and upper.volatility is None


def test_bound_violations_are_not_silently_clipped():
    for price in (np.nextafter(20.0, -np.inf), np.nextafter(100.0, np.inf)):
        with pytest.raises(ValueError, match="outside"):
            implied_volatility(price, 100, 80, 1)


def test_loss_of_time_value_is_not_claimed_as_recovered_volatility():
    price = call_price(100, 70, 1 / 252, 0.25)
    result = implied_volatility(price, 100, 70, 1 / 252)
    assert result.status == "lower_boundary"
    assert result.volatility != 0.25


@pytest.mark.parametrize("scale", [0.01, 1000.0])
def test_currency_scale_invariance(scale):
    price = call_price(100, 110, 0.4, 0.3, 0.98)
    result = implied_volatility(price * scale, 100 * scale, 110 * scale, 0.4, 0.98)
    assert result.volatility == pytest.approx(0.3, abs=1e-10)


def test_inverse_derivative_is_reciprocal_vega():
    price = call_price(100, 100, 0.5, 0.25)
    step = 1e-4
    up = implied_volatility(price + step, 100, 100, 0.5).volatility
    down = implied_volatility(price - step, 100, 100, 0.5).volatility
    assert (up - down) / (2 * step) == pytest.approx(1 / vega(100, 100, 0.5, 0.25), rel=1e-7)


def test_insufficient_bracket_or_iteration_budget_fails():
    price = call_price(100, 100, 1, 2)
    with pytest.raises(InversionError, match="max_total_std"):
        implied_volatility(price, 100, 100, 1, max_total_std=0.5)
    with pytest.raises(InversionError, match="budget"):
        implied_volatility(call_price(100, 110, 0.5, 0.35), 100, 110, 0.5, max_iterations=1)


def test_quote_interval_endpoints_reprice_and_contain_generating_sigma():
    price = call_price(100, 105, 0.5, 0.25, 0.98)
    result = volatility_interval(price - 0.005, price + 0.005, 100, 105, 0.5, 0.98)
    assert result.volatility_lower < 0.25 < result.volatility_upper
    assert call_price(100, 105, 0.5, result.volatility_lower, 0.98) == pytest.approx(price - 0.005, abs=1e-10)
    assert call_price(100, 105, 0.5, result.volatility_upper, 0.98) == pytest.approx(price + 0.005, abs=1e-10)


def test_quote_interval_intersects_bounds_explicitly():
    result = volatility_interval(-0.005, 0.005, 100, 115, 1 / 252)
    assert result.quote_lower == -0.005
    assert result.feasible_price_lower == 0
    assert result.volatility_lower == 0
    assert result.volatility_upper > 0.25


def test_unbounded_range_is_not_capped_at_solver_budget():
    result = volatility_interval(99.99, 100.01, 100, 100, 1)
    assert result.upper_unbounded
    assert result.volatility_upper is None
    assert result.volatility_lower > 0
    complete = volatility_interval(-1, 101, 100, 100, 1)
    assert complete.volatility_lower == 0 and complete.upper_unbounded


def test_local_linear_width_for_small_atm_band():
    price = call_price(100, 100, 1, 0.25)
    result = volatility_interval(price - 1e-5, price + 1e-5, 100, 100, 1)
    assert result.volatility_upper - result.volatility_lower == pytest.approx(
        2e-5 / vega(100, 100, 1, 0.25), rel=1e-6)


@pytest.mark.parametrize("low,high", [(2, 1), (-2, -1), (101, 102), (100, 100), (0, np.inf)])
def test_invalid_or_infeasible_quote_band(low, high):
    with pytest.raises(ValueError):
        volatility_interval(low, high, 100, 100, 1)


@pytest.mark.parametrize("options", [{"maturity": 0}, {"maturity": -1}, {"price": np.nan},
    {"max_iterations": 0}, {"max_iterations": True}, {"max_total_std": 0}])
def test_invalid_inverse_options(options):
    arguments = {"price": 10.0, "forward": 100, "strike": 100, "maturity": 1, **options}
    with pytest.raises(ValueError):
        implied_volatility(**arguments)
