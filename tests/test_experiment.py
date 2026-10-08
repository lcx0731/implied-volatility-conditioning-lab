import json
import subprocess
import sys

import numpy as np
import pytest

from iv_audit.experiment import run_audit


def test_default_audit_is_deterministic_and_quote_ranges_contain_truth():
    first, second = run_audit(), run_audit()
    assert first == second
    assert first["checks"]["number_of_options"] == 42
    assert first["checks"]["true_volatility_in_all_ranges"]
    for row in first["options"]:
        band = row["quote_consistent_interval"]
        assert band["volatility_lower"] <= row["true_volatility"] + 1e-8
        assert band["volatility_upper"] >= row["true_volatility"] - 1e-8
    # Underidentification is expected, not a successful 25% point recovery.
    assert first["checks"]["quote_point_status_counts"]["lower_boundary"] > 0
    assert first["checks"]["full_precision_status_counts"]["lower_boundary"] > 0


def test_finer_quote_resolution_runs_without_changing_generating_prices():
    coarse, fine = run_audit(0.01), run_audit(0.001)
    assert [r["unrounded_price"] for r in coarse["options"]] == [r["unrounded_price"] for r in fine["options"]]
    assert np.median([r["interval_width"] for r in fine["options"]]) < np.median([r["interval_width"] for r in coarse["options"]])


@pytest.mark.parametrize("tick", [0, -1, np.nan, np.inf, 1e-9, 2, True])
def test_invalid_tick(tick):
    with pytest.raises(ValueError):
        run_audit(tick)


def test_cli_creates_valid_json_and_figures(tmp_path):
    subprocess.run([sys.executable, "-m", "iv_audit.experiment", "--output", str(tmp_path)],
                    check=True, capture_output=True, text=True)
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["checks"]["number_of_options"] == 42
    for name in ("quote-resolution-map.png", "conditioning-examples.png"):
        data = (tmp_path / name).read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) > 10000
