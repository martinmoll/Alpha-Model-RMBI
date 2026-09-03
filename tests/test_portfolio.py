import numpy as np
import pandas as pd
from core.portfolio import construct_portfolio, build_portfolio_series


def _make_month_df(n=50):
    np.random.seed(42)
    return pd.DataFrame({
        "permno": range(10001, 10001 + n),
        "pred": np.random.randn(n) * 0.1,
        "y_raw": np.random.randn(n) * 0.08,
        "vol_12m_xs": np.abs(np.random.randn(n)) * 0.5 + 0.5,
        "sector": ["Tech", "Finance", "Health", "Energy", "Consumer"] * (n // 5),
    })


def test_equal_weight():
    df = _make_month_df()
    result = construct_portfolio(df, method="equal_weight", K=10,
                                strategy_type="long_only", K_short=10, vol_tilt=0.0)
    assert len(result) == 10
    assert "weight" in result.columns
    np.testing.assert_almost_equal(result["weight"].sum(), 1.0)


def test_score_weight():
    df = _make_month_df()
    result = construct_portfolio(df, method="score_weight", K=10,
                                strategy_type="long_only", K_short=10, vol_tilt=0.0)
    assert len(result) == 10
    np.testing.assert_almost_equal(result["weight"].sum(), 1.0, decimal=5)


def test_inverse_vol():
    df = _make_month_df()
    result = construct_portfolio(df, method="inverse_vol", K=10,
                                strategy_type="long_only", K_short=10, vol_tilt=0.0)
    assert len(result) == 10
    np.testing.assert_almost_equal(result["weight"].sum(), 1.0, decimal=5)


def test_erc():
    np.random.seed(42)
    df = _make_month_df(30)
    returns_hist = pd.DataFrame(
        np.random.randn(24, 30) * 0.05,
        columns=range(10001, 10031),
    )
    result = construct_portfolio(
        df, method="erc", K=10, strategy_type="long_only",
        K_short=10, vol_tilt=0.0, returns_history=returns_hist,
    )
    assert len(result) == 10
    np.testing.assert_almost_equal(result["weight"].sum(), 1.0, decimal=3)
    assert (result["weight"] > 0).all()


def test_mvo():
    np.random.seed(42)
    df = _make_month_df(30)
    returns_hist = pd.DataFrame(
        np.random.randn(24, 30) * 0.05,
        columns=range(10001, 10031),
    )
    result = construct_portfolio(
        df, method="mvo", K=10, strategy_type="long_only",
        K_short=10, vol_tilt=0.0, returns_history=returns_hist,
    )
    assert len(result) == 10
    np.testing.assert_almost_equal(result["weight"].sum(), 1.0, decimal=3)
    assert (result["weight"] >= -0.001).all()
    assert (result["weight"] <= 0.151).all()


def test_mvo_tc_aware():
    np.random.seed(42)
    df = _make_month_df(30)
    returns_hist = pd.DataFrame(
        np.random.randn(24, 30) * 0.05,
        columns=range(10001, 10031),
    )
    prev_w = np.ones(10) / 10
    result = construct_portfolio(
        df, method="mvo", K=10, strategy_type="long_only",
        K_short=10, vol_tilt=0.0, returns_history=returns_hist,
        prev_weights=prev_w, tc_bps=50.0,
    )
    assert len(result) == 10
    np.testing.assert_almost_equal(result["weight"].sum(), 1.0, decimal=3)


def test_long_short():
    df = _make_month_df()
    result = construct_portfolio(df, method="equal_weight", K=10,
                                strategy_type="long_short", K_short=5, vol_tilt=0.0)
    assert len(result) == 15
    assert "side" in result.columns
    assert (result[result["side"] == "long"]["weight"] > 0).all()
    assert (result[result["side"] == "short"]["weight"] < 0).all()


def test_build_portfolio_series(sample_predictions):
    result = build_portfolio_series(
        predictions=sample_predictions,
        method="equal_weight",
        K=10,
        strategy_type="long_only",
        K_short=10,
        vol_tilt=0.05,
        regime_lookback=0,
    )
    assert "monthly_returns" in result
    assert "holdings" in result
    assert "ic" in result
    assert "turnover" in result
    assert len(result["monthly_returns"]) > 0


# --- returns_history plumbing and point-in-time guard -----------------------
# The covariance matrix was always the identity, because no caller ever passed
# returns_history. These tests pin both the fix and the point-in-time rule that
# the fix must not break.

def _series_inputs(n_months=6, n_stocks=30):
    np.random.seed(7)
    months = [f"2015-{m:02d}" for m in range(1, n_months + 1)]
    permnos = list(range(10001, 10001 + n_stocks))
    preds = {}
    for m in months:
        preds[m] = pd.DataFrame({
            "permno": permnos,
            "pred": np.random.randn(n_stocks) * 0.1,
            "y_raw": np.random.randn(n_stocks) * 0.08,
            "vol_12m_xs": np.abs(np.random.randn(n_stocks)) * 0.5 + 0.5,
        })
    return months, permnos, preds


def test_build_returns_history_shape():
    from core.portfolio import build_returns_history
    panel = pd.DataFrame({
        "ym": ["2015-01", "2015-01", "2015-02", "2015-02"],
        "permno": [1, 2, 1, 2],
        "ret_1": [0.01, 0.02, 0.03, 0.04],
    })
    hist = build_returns_history(panel)
    assert list(hist.index) == ["2015-01", "2015-02"]
    assert set(hist.columns) == {1, 2}
    assert hist.loc["2015-02", 1] == 0.03


def test_erc_weights_use_the_covariance():
    """With a real covariance, ERC must not collapse to equal weight."""
    months, permnos, preds = _series_inputs()
    rng = np.random.default_rng(0)
    # Two blocks with very different volatility -> ERC must tilt away from 1/N.
    base = rng.normal(0, 0.01, (36, len(permnos)))
    base[:, :10] *= 8.0
    hist = pd.DataFrame(base, index=[f"2012-{i:02d}" for i in range(1, 13)]
                        + [f"2013-{i:02d}" for i in range(1, 13)]
                        + [f"2014-{i:02d}" for i in range(1, 13)],
                        columns=permnos)
    out = build_portfolio_series(
        predictions=preds, method="erc", K=10, strategy_type="long_only",
        K_short=10, vol_tilt=0.0, regime_lookback=0, returns_history=hist,
    )
    w = out["holdings"][months[0]]["weight"].values
    assert not np.allclose(w, np.ones(len(w)) / len(w), atol=1e-3), \
        "ERC collapsed to equal weight -- covariance was ignored"


def test_covariance_is_point_in_time():
    """Weights at month m must not change when data AFTER m is added."""
    months, permnos, preds = _series_inputs()
    rng = np.random.default_rng(1)
    past_idx = [f"2014-{i:02d}" for i in range(1, 13)]
    past = pd.DataFrame(rng.normal(0, 0.02, (12, len(permnos))),
                        index=past_idx, columns=permnos)

    # A future block with wildly different covariance structure. It must start
    # strictly AFTER the month under test: ret_1[2015-01] is the return over
    # January and is legitimately known at the January decision point, so only
    # 2015-02 onward counts as future information.
    future_idx = [f"2015-{i:02d}" for i in range(2, 8)]
    future = pd.DataFrame(rng.normal(0, 0.50, (6, len(permnos))),
                          index=future_idx, columns=permnos)
    with_future = pd.concat([past, future])

    out_past = build_portfolio_series(
        predictions=preds, method="erc", K=10, strategy_type="long_only",
        K_short=10, vol_tilt=0.0, regime_lookback=0, returns_history=past,
    )
    out_full = build_portfolio_series(
        predictions=preds, method="erc", K=10, strategy_type="long_only",
        K_short=10, vol_tilt=0.0, regime_lookback=0, returns_history=with_future,
    )
    w_past = out_past["holdings"][months[0]]["weight"].values
    w_full = out_full["holdings"][months[0]]["weight"].values
    np.testing.assert_allclose(
        w_past, w_full, atol=1e-6,
        err_msg="Future returns changed the weights -- covariance is not point-in-time",
    )


def test_costs_reduce_returns_and_gross_is_kept():
    months, permnos, preds = _series_inputs()
    free = build_portfolio_series(
        predictions=preds, method="equal_weight", K=10, strategy_type="long_only",
        K_short=10, vol_tilt=0.0, regime_lookback=0, cost_bps=0.0,
    )
    costed = build_portfolio_series(
        predictions=preds, method="equal_weight", K=10, strategy_type="long_only",
        K_short=10, vol_tilt=0.0, regime_lookback=0, cost_bps=10.0,
    )
    # Gross is identical; net is strictly worse in every month.
    pd.testing.assert_series_equal(
        free["monthly_returns_gross"], costed["monthly_returns_gross"])
    assert (costed["monthly_returns"] < costed["monthly_returns_gross"]).all()
    # A frictionless run leaves net == gross.
    pd.testing.assert_series_equal(
        free["monthly_returns"], free["monthly_returns_gross"], check_names=False)


def test_first_month_is_charged_a_full_build():
    months, permnos, preds = _series_inputs(n_months=1)
    out = build_portfolio_series(
        predictions=preds, method="equal_weight", K=10, strategy_type="long_only",
        K_short=10, vol_tilt=0.0, regime_lookback=0, cost_bps=10.0,
    )
    m0 = months[0]
    drag = out["monthly_returns_gross"][m0] - out["monthly_returns"][m0]
    np.testing.assert_almost_equal(drag, 1.0 * 10.0 / 10_000 * 2, decimal=10)
