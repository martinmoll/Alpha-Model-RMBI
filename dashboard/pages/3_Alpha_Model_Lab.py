# dashboard/pages/3_Alpha_Model_Lab.py
"""Page 3: Alpha Model Lab — configure, train, and launch backtests."""

import streamlit as st
import pandas as pd
from core.models import list_models, get_model, get_default_params, get_param_ranges, get_feature_tier
from core.backtest import run_walk_forward
from core.portfolio import build_portfolio_series
from core.diagnostics import compute_performance_metrics
from features import FEATURE_GROUPS, get_tier_defaults
import cache_manager as cache
from components.theory import theory_section
from components.workflow import render_workflow_status, render_next_steps, config_label
from components.theme import inject_theme, COLORS, FONT_SANS

st.set_page_config(page_title="Alpha Model Lab", layout="wide")
inject_theme()

C = COLORS
st.markdown(
    f'<h1 style="font-family:{FONT_SANS};font-size:28px;font-weight:700;'
    f'color:{C["text"]};margin:0;">Alpha Model Lab</h1>',
    unsafe_allow_html=True,
)
render_workflow_status("model")

df = st.session_state.get("df")
market_monthly = st.session_state.get("market_monthly")
if df is None:
    st.error("Dataset not loaded.")
    st.stop()

theory_section("Walk-Forward Validation", "walk_forward")
theory_section("Regularization", "regularization")

# --- Model Selection ---
st.header("Model Configuration")
model_col1, model_col2 = st.columns([1, 2])

with model_col1:
    model_name = st.selectbox("Model", list_models())
    tier = get_feature_tier(model_name)
    st.caption(f"Feature Tier: {tier} ({'~118 features' if tier == 2 else '~52 features'})")

with model_col2:
    param_ranges = get_param_ranges(model_name)
    model_params = {}
    if param_ranges:
        cols = st.columns(min(len(param_ranges), 3))
        for i, (param, spec) in enumerate(param_ranges.items()):
            with cols[i % len(cols)]:
                if isinstance(spec["default"], float):
                    model_params[param] = st.slider(
                        param, min_value=spec["min"], max_value=spec["max"],
                        value=spec["default"], step=spec.get("step", 0.01),
                    )
                else:
                    model_params[param] = st.slider(
                        param, min_value=spec["min"], max_value=spec["max"],
                        value=spec["default"], step=spec.get("step", 1),
                    )
    else:
        st.info("No tunable hyperparameters for this model.")
        model_params = get_default_params(model_name)

# --- Feature Selection ---
st.header("Feature Selection")

preset = st.selectbox("Preset", [
    "Tier default", "Momentum only", "Value only", "Quality only",
    "Kitchen sink (all 118)",
])

if preset == "Tier default":
    selected_features = get_tier_defaults(tier)
elif preset == "Momentum only":
    selected_features = FEATURE_GROUPS.get("momentum", [])
elif preset == "Value only":
    selected_features = FEATURE_GROUPS.get("value", [])
elif preset == "Quality only":
    selected_features = FEATURE_GROUPS.get("quality", [])
else:
    selected_features = get_tier_defaults(2)

with st.expander("Customize features", expanded=False):
    custom_features = []
    for group_name, group_cols in FEATURE_GROUPS.items():
        available = [c for c in group_cols if c in df.columns or c in selected_features]
        if available:
            selected_in_group = st.multiselect(
                group_name.capitalize(),
                options=available,
                default=[c for c in available if c in selected_features],
                key=f"feat_{group_name}",
            )
            custom_features.extend(selected_in_group)
    if custom_features:
        selected_features = custom_features

available_features = [f for f in selected_features if f in df.columns]
st.caption(f"Selected: {len(available_features)} features")

# --- Walk-Forward Configuration ---
st.header("Walk-Forward Configuration")
wf_col1, wf_col2, wf_col3, wf_col4 = st.columns(4)

all_months = sorted(df["ym"].unique())
oos_candidates = [m for m in all_months if m >= "2010-01"]

with wf_col1:
    # Default 2016-07: the panel jumps from 89 names to 468 that month. Before
    # it, the universe is only the handful of today's constituents with price
    # history that far back, so earlier start dates are severely survivorship-
    # biased even by this dataset's standards.
    _default_oos = "2016-07" if "2016-07" in oos_candidates else oos_candidates[0]
    oos_start = st.selectbox("OOS Start", oos_candidates,
                             index=oos_candidates.index(_default_oos))
    if oos_start < "2016-07":
        st.caption(
            ":warning: Before 2016-07 the universe is under 90 names — today's "
            "survivors only. Results from that window are not credible."
        )
with wf_col2:
    retrain_freq = st.selectbox("Retrain Every (months)", [6, 12, 24], index=1)
with wf_col3:
    window_type = st.selectbox("Window Type", ["expanding", "rolling"])
with wf_col4:
    rolling_window = None
    if window_type == "rolling":
        rolling_window = st.slider("Rolling Window (months)", 24, 120, 60)

auto_tune = st.checkbox("Auto-tune hyperparameters", value=False,
                         help="Inner time-series CV to select HPs at each retrain point")

# --- Portfolio Configuration ---
st.header("Portfolio Configuration")
port_col1, port_col2, port_col3, port_col4, port_col5 = st.columns(5)

with port_col1:
    strategy_type = st.selectbox("Strategy", ["Long Only", "Long-Short"])
    strategy_key = "long_short" if strategy_type == "Long-Short" else "long_only"
with port_col2:
    K = st.slider("K (stocks)", min_value=5, max_value=50, step=5, value=10)
with port_col3:
    K_short = K
    if strategy_key == "long_short":
        K_short = st.slider("K short", min_value=5, max_value=50, step=5, value=K)
with port_col4:
    vol_tilt = st.slider("Vol tilt", min_value=0.0, max_value=0.50, step=0.01, value=0.05)
with port_col5:
    regime_lookback = st.slider("Regime lookback", min_value=0, max_value=12, value=6)

construction_method = st.selectbox(
    "Construction Method",
    ["equal_weight", "score_weight", "inverse_vol", "erc", "mvo"],
)

cap_on = st.checkbox(
    "Cap idiosyncratic volatility (recommended)", value=True,
    help="Without a cap the model puts ~91% of the book in the top two "
         "volatility deciles, where the survivor-only universe is most "
         "distorted. Capping barely moves the Sharpe but roughly halves the "
         "return and the drawdown.",
)
max_ivol_xs = st.slider("Max ivol (cross-sectional z)", min_value=-1.0,
                        max_value=3.0, value=1.0, step=0.5) if cap_on else None

cost_bps = st.slider(
    "Transaction cost (bps, one way)", min_value=0, max_value=50, value=10, step=5,
    help="Charged against realized returns every month, on the notional traded. "
         "Set to 0 for a frictionless (unrealistic) run.",
)

# Separate knob: MVO can also penalize turnover *inside* the optimizer, which
# is a different thing from charging the realized cost above.
tc_bps = 0.0
if construction_method == "mvo":
    tc_bps = st.slider("MVO turnover penalty (bps)", min_value=0, max_value=50,
                       value=10, step=5)

# --- Action Buttons ---
btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    run_clicked = st.button("Run Backtest", type="primary")
with btn_col2:
    pin_clicked = st.button("Pin Config")

# --- Run Backtest ---
if run_clicked:
    pred_key = cache.prediction_key(
        model_name, model_params, retrain_freq,
        feature_cols=available_features, window_type=window_type,
        auto_tune=auto_tune, data_fingerprint=cache.dataset_fingerprint(df),
        oos_start=oos_start, rolling_window=rolling_window,
    )
    predictions = cache.get_predictions(pred_key)

    if predictions is None:
        model = get_model(model_name, model_params)
        progress = st.progress(0, text="Running walk-forward backtest...")
        result = run_walk_forward(
            data=df, model=model, feature_cols=available_features,
            oos_start=oos_start, retrain_freq=retrain_freq,
            window_type=window_type, rolling_window=rolling_window,
            auto_tune=auto_tune,
            progress_callback=lambda step, total, month: progress.progress(
                step / total, text=f"Training... {month} ({step}/{total})",
            ),
        )
        progress.empty()
        predictions = result.predictions
        cache.save_predictions(pred_key, predictions)
        st.session_state.backtest_feature_importance = result.feature_importance
        st.session_state.backtest_train_dates = result.train_dates
        st.session_state.backtest_tuned_params = result.tuned_params
    else:
        st.session_state.backtest_feature_importance = None
        st.session_state.backtest_train_dates = []

    port_key = cache.portfolio_key(
        pred_key, K, vol_tilt, regime_lookback,
        strategy_key, K_short, construction_method, tc_bps=tc_bps,
        cost_bps=cost_bps, max_ivol_xs=max_ivol_xs,
    )
    portfolio = cache.get_portfolio(port_key)

    if portfolio is None:
        portfolio = build_portfolio_series(
            predictions=predictions, method=construction_method,
            K=K, strategy_type=strategy_key, K_short=K_short,
            vol_tilt=vol_tilt, regime_lookback=regime_lookback,
            market_monthly=market_monthly, tc_bps=tc_bps,
            returns_history=st.session_state.get("returns_history"),
            cost_bps=cost_bps, max_ivol_xs=max_ivol_xs,
        )
        cache.save_portfolio(port_key, portfolio)

    st.session_state.backtest_result = portfolio
    st.session_state.backtest_predictions = predictions
    st.session_state.backtest_params = {
        "model_name": model_name, "model_params": model_params,
        "retrain_freq": retrain_freq, "K": K, "K_short": K_short,
        "vol_tilt": vol_tilt, "regime_lookback": regime_lookback,
        "strategy_type": strategy_key, "construction_method": construction_method,
        "features": available_features, "window_type": window_type,
        "oos_start": oos_start, "rolling_window": rolling_window,
        "cost_bps": cost_bps, "max_ivol_xs": max_ivol_xs,
    }
    st.success("Backtest complete!")
    render_next_steps("model")

# --- Pin Config ---
if pin_clicked:
    result = st.session_state.get("backtest_result")
    params = st.session_state.get("backtest_params")
    pinned = st.session_state.get("pinned_configs", [])
    if result is not None and len(pinned) < 4:
        label = config_label(params)
        pinned.append({
            "label": label, "result": result, "params": params,
            "predictions": st.session_state.get("backtest_predictions"),
        })
        st.session_state.pinned_configs = pinned
        st.success(f"Pinned: {label}")

# --- Show pinned configs ---
pinned = st.session_state.get("pinned_configs", [])
if pinned:
    st.markdown("**Pinned configs:**")
    chip_cols = st.columns(len(pinned))
    to_remove = None
    for i, p in enumerate(pinned):
        with chip_cols[i]:
            if st.button(f"X {p['label']}", key=f"rm_pin_{i}"):
                to_remove = i
    if to_remove is not None:
        pinned.pop(to_remove)
        st.session_state.pinned_configs = pinned
        st.rerun()
