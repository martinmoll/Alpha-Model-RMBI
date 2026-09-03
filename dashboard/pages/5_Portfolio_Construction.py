# dashboard/pages/5_Portfolio_Construction.py
"""Page 5: Portfolio Construction & Risk Analysis — tabbed layout."""

import streamlit as st
import pandas as pd
import numpy as np
from core.portfolio import build_portfolio_series
from core.data_loader import permno_ticker_map
from core.diagnostics import compute_performance_metrics
from core.risk import (
    risk_contribution, factor_exposure, factor_alpha, rolling_factor_exposure,
    transaction_cost_drag, capacity_curve,
)
from components.charts import (
    sector_allocation_chart, risk_pie_chart, bar_chart, STYLE,
)
from components.metrics import metric_card, metric_card_row, metric_row, banner, comparison_table
from components.theory import theory_section
from components.interpretations import (
    render_interpretation, interpret_factor_exposure, interpret_ff5_alpha,
    interpret_r_squared, interpret_turnover, interpret_sharpe,
)
from components.workflow import render_workflow_status, render_empty_state, render_next_steps
from components.theme import inject_theme, COLORS, FONT_MONO, FONT_SANS
import plotly.graph_objects as go

st.set_page_config(page_title="Portfolio Construction", layout="wide")
inject_theme()

if render_empty_state("portfolio"):
    st.stop()

result = st.session_state.get("backtest_result")
predictions = st.session_state.get("backtest_predictions")
params = st.session_state.get("backtest_params")
ff5 = st.session_state.get("ff5_factors")
market = st.session_state.get("market_monthly")

C = COLORS

# --- Header ---
head_left, head_right = st.columns([2, 1])
with head_left:
    st.markdown(
        f'<h1 style="font-family:{FONT_SANS};font-size:28px;font-weight:700;'
        f'color:{C["text"]};margin:0;">Portfolio Construction & Risk</h1>',
        unsafe_allow_html=True,
    )
    model_name = params.get("model_name", "—")
    strategy = params.get("strategy_type", "long_only").replace("_", "-")
    K = params.get("K", 10)
    method = params.get("construction_method", "equal_weight")
    subtitle = f"{model_name} &middot; {strategy} &middot; K={K} &middot; {method}"
    st.markdown(
        f'<p style="font-family:{FONT_SANS};font-size:13px;color:{C["text_secondary"]};'
        f'margin:4px 0 0;">{subtitle}</p>',
        unsafe_allow_html=True,
    )
with head_right:
    construction_method = st.selectbox(
        "Construction method",
        ["equal_weight", "score_weight", "inverse_vol", "erc", "mvo"],
        index=["equal_weight", "score_weight", "inverse_vol", "erc", "mvo"].index(method)
        if method in ["equal_weight", "score_weight", "inverse_vol", "erc", "mvo"] else 0,
        label_visibility="collapsed",
    )

render_workflow_status("portfolio")

theory_section("Portfolio Construction Methods", "portfolio_construction")

# --- Build method results for comparison ---
methods = ["equal_weight", "score_weight", "inverse_vol", "erc", "mvo"]
st.sidebar.header("Compare Methods")
selected_methods = st.sidebar.multiselect("Methods to compare", methods, default=["equal_weight", "inverse_vol"])

method_results = {}
for m in selected_methods:
    port = build_portfolio_series(
        predictions=predictions, method=m,
        K=params["K"], strategy_type=params["strategy_type"],
        K_short=params["K_short"], vol_tilt=params["vol_tilt"],
        regime_lookback=params["regime_lookback"], market_monthly=market,
        returns_history=st.session_state.get("returns_history"),
    )
    method_results[m] = port

# --- Tabs ---
tab_alloc, tab_holdings, tab_risk, tab_costs, tab_compare = st.tabs(
    ["Allocation", "Holdings", "Risk & factors", "Costs", "Compare methods"]
)

# =========================================================================
# ALLOCATION TAB
# =========================================================================
with tab_alloc:
    holdings = result.get("holdings", {})
    if holdings:
        last_month = sorted(holdings.keys())[-1]
        held = holdings[last_month]
        n_pos = len(held)
        weight_col = held["weight"] if "weight" in held.columns else pd.Series()
        eff_n = 1.0 / (weight_col ** 2).sum() if len(weight_col) > 0 and weight_col.sum() > 0 else 0
        top10_w = weight_col.abs().nlargest(10).sum() if len(weight_col) > 0 else 0

        alloc_cards = [
            {"label": "Positions", "value": str(n_pos), "accent": C["primary"], "variant": "bar"},
            {"label": "Effective N", "value": f"{eff_n:.1f}", "accent": C["primary"], "variant": "bar"},
            {"label": "Top-10 Weight", "value": f"{top10_w:.1%}", "accent": C["warning"] if top10_w > 0.6 else C["primary"], "variant": "bar"},
        ]
        if params.get("strategy_type") == "long_short" and "side" in held.columns:
            net_exp = weight_col.sum()
            alloc_cards.append({"label": "Net Exposure", "value": f"{net_exp:.1%}", "accent": C["primary"], "variant": "bar"})
        metric_card_row(alloc_cards)

    st.markdown("---")
    st.subheader("Sector Allocation Over Time")
    if result["holdings"]:
        fig = sector_allocation_chart(result["holdings"], params["strategy_type"])
        st.plotly_chart(fig, use_container_width=True)

# =========================================================================
# HOLDINGS TAB
# =========================================================================
with tab_holdings:
    holdings = result["holdings"]
    if holdings:
        tk_map = permno_ticker_map(st.session_state.df)
        months_sorted = sorted(holdings.keys())

        st.subheader("Holdings by Month")
        sel_month = st.selectbox(
            "Month", months_sorted, index=len(months_sorted) - 1,
            help="Latest month is the current portfolio; earlier months show past selections.",
        )
        held = holdings[sel_month].copy()
        held["ticker"] = held["permno"].map(tk_map).fillna(held["permno"].astype(str))
        if "weight" in held.columns and "y_raw" in held.columns:
            # This month's contribution to the portfolio return: weight x return.
            held["contribution"] = held["weight"] * held["y_raw"]

        display_cols = []
        for c in ["ticker", "side", "sector", "pred", "y_raw", "weight",
                  "contribution", "permno"]:
            if c in held.columns:
                display_cols.append(c)
        show = held[display_cols].copy()
        if "pred" in show.columns:
            show["pred"] = show["pred"].round(4)
        if "y_raw" in show.columns:
            show["y_raw"] = (show["y_raw"] * 100).round(2).astype(str) + "%"
        if "weight" in show.columns:
            show["weight"] = (show["weight"] * 100).round(2).astype(str) + "%"
        if "contribution" in show.columns:
            show["contribution"] = (show["contribution"] * 100).round(2).astype(str) + "%"
        st.markdown(f"**{sel_month}** ({len(held)} positions)")
        st.dataframe(show.reset_index(drop=True), use_container_width=True)

        # --- Selection history across the full backtest ---
        st.markdown("---")
        st.subheader("Selection History")
        st.caption(
            f"How often each stock was held across all {len(months_sorted)} "
            "out-of-sample months."
        )
        combined = pd.concat(
            [h.assign(ym=m) for m, h in holdings.items()], ignore_index=True
        )
        combined["ticker"] = (
            combined["permno"].map(tk_map).fillna(combined["permno"].astype(str))
        )
        agg_spec = {"months_held": ("ym", "nunique")}
        if "weight" in combined.columns:
            agg_spec["avg_weight"] = ("weight", "mean")
        if "pred" in combined.columns:
            agg_spec["avg_pred"] = ("pred", "mean")
        if "y_raw" in combined.columns:
            agg_spec["avg_return"] = ("y_raw", "mean")
            # Compounded return over the months the stock was actually held.
            agg_spec["total_return"] = (
                "y_raw", lambda s: (1 + s.dropna()).prod() - 1
            )
        if "weight" in combined.columns and "y_raw" in combined.columns:
            # Sum of monthly (weight x return): the stock's additive contribution
            # to portfolio return over the whole period.
            combined["_contribution"] = combined["weight"] * combined["y_raw"]
            agg_spec["total_contribution"] = ("_contribution", "sum")
        hist = combined.groupby(["ticker", "permno"]).agg(**agg_spec).reset_index()
        hist["pct_months"] = (hist["months_held"] / len(months_sorted) * 100).round(1).astype(str) + "%"
        hist = hist.sort_values("months_held", ascending=False)
        if "avg_pred" in hist.columns:
            hist["avg_pred"] = hist["avg_pred"].round(4)
        if "avg_weight" in hist.columns:
            hist["avg_weight"] = (hist["avg_weight"] * 100).round(2).astype(str) + "%"
        if "avg_return" in hist.columns:
            hist["avg_return"] = (hist["avg_return"] * 100).round(2).astype(str) + "%"
        if "total_return" in hist.columns:
            hist["total_return"] = (hist["total_return"] * 100).round(2).astype(str) + "%"
        if "total_contribution" in hist.columns:
            hist["total_contribution"] = (hist["total_contribution"] * 100).round(2).astype(str) + "%"
        order = ["ticker", "permno", "months_held", "pct_months",
                 "avg_weight", "avg_pred", "avg_return", "total_return",
                 "total_contribution"]
        hist = hist[[c for c in order if c in hist.columns]]
        st.dataframe(hist.reset_index(drop=True), use_container_width=True)

# =========================================================================
# RISK & FACTORS TAB
# =========================================================================
with tab_risk:
    risk_left, risk_right = st.columns(2)

    with risk_left:
        st.subheader("Risk Decomposition")
        if len(selected_methods) >= 1:
            for m_name in selected_methods[:2]:
                port = method_results[m_name]
                if port["holdings"]:
                    last_m = sorted(port["holdings"].keys())[-1]
                    held = port["holdings"][last_m]
                    if "weight" in held.columns:
                        w = held["weight"].abs().values
                        n = len(w)
                        cov = np.eye(n) * 0.01
                        rc = risk_contribution(w / w.sum(), cov)
                        if "sector" in held.columns:
                            rc.index = [
                                f"{p} ({s})" for p, s in
                                zip(held["permno"].values, held["sector"].values)
                            ]
                        else:
                            rc.index = [str(p) for p in held["permno"].values]
                        fig = risk_pie_chart(rc)
                        fig.update_layout(title=f"Risk Contribution — {m_name}")
                        st.plotly_chart(fig, use_container_width=True)

    with risk_right:
        if ff5 is not None:
            st.subheader("Factor Exposure")
            rets = result["monthly_returns"]
            exposure = factor_exposure(rets, ff5)
            if len(exposure) > 0:
                exp_cards = []
                for factor, beta in exposure.items():
                    accent = C["primary"] if abs(beta) < 0.10 else C["warning"]
                    exp_cards.append({"label": factor, "value": f"{beta:.3f}", "accent": accent, "variant": "bar"})
                metric_card_row(exp_cards[:3])
                if len(exp_cards) > 3:
                    metric_card_row(exp_cards[3:])
                render_interpretation(interpret_factor_exposure(exposure.to_dict()))

            rolling_exp = rolling_factor_exposure(rets, ff5, window=24)
            if not rolling_exp.dropna(how="all").empty:
                fig = go.Figure()
                for col in rolling_exp.columns:
                    fig.add_trace(go.Scatter(
                        x=rolling_exp.index, y=rolling_exp[col].values, name=col,
                    ))
                fig.add_hline(y=0.10, line_dash="dash", line_color=C["warning"])
                fig.add_hline(y=-0.10, line_dash="dash", line_color=C["warning"])
                fig.update_layout(
                    template="alpha", title="Rolling 24-Month Factor Exposures",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

    # Jensen's Alpha
    if ff5 is not None:
        st.markdown("---")
        theory_section("Jensen's Alpha — Do You Have Skill?", "jensens_alpha")

        rets = result["monthly_returns"]
        alpha_result = factor_alpha(rets, ff5)
        if alpha_result is not None:
            st.subheader("Jensen's Alpha")
            metric_card_row([
                {"label": "Annualized Alpha", "value": f"{alpha_result['annual_alpha']:.2%}",
                 "accent": C["positive"] if alpha_result["annual_alpha"] > 0 else C["negative"], "variant": "bar"},
                {"label": "t-statistic", "value": f"{alpha_result['t_stat']:.2f}",
                 "accent": C["positive"] if alpha_result["t_stat"] > 2 else C["warning"], "variant": "bar"},
                {"label": "p-value", "value": f"{alpha_result['p_value']:.4f}",
                 "accent": C["positive"] if alpha_result["p_value"] < 0.05 else C["warning"], "variant": "bar"},
                {"label": "R-squared", "value": f"{alpha_result['r_squared']:.3f}",
                 "accent": C["primary"], "variant": "bar"},
            ])
            render_interpretation(interpret_ff5_alpha(
                alpha_result["annual_alpha"], alpha_result["t_stat"],
                alpha_result["p_value"], alpha_result["r_squared"],
            ))
            render_interpretation(interpret_r_squared(alpha_result["r_squared"], "factor"))

        # Per-method factor exposure comparison
        if len(method_results) >= 2 and ff5 is not None:
            st.markdown("---")
            st.subheader("Factor Exposure by Construction Method")
            method_exposures = {}
            method_alphas = {}
            for method_name, port in method_results.items():
                m_rets = port["monthly_returns"]
                m_exp = factor_exposure(m_rets, ff5)
                if len(m_exp) > 0:
                    method_exposures[method_name] = m_exp
                m_alpha = factor_alpha(m_rets, ff5)
                if m_alpha is not None:
                    method_alphas[method_name] = m_alpha

            if method_exposures:
                exp_df = pd.DataFrame(method_exposures).T
                exp_df.index.name = "Method"

                def _color_exposure(val):
                    if abs(val) < 0.10:
                        return ""
                    if abs(val) < 0.20:
                        return f"background-color: rgba(245,177,61,0.15)"
                    return f"background-color: rgba(244,106,106,0.15)"

                styled = exp_df.style.format("{:.3f}").map(_color_exposure)
                st.dataframe(styled, use_container_width=True)

                for method_name, m_exp in method_exposures.items():
                    with st.expander(f"Interpretation: {method_name}"):
                        render_interpretation(interpret_factor_exposure(m_exp.to_dict()))
                        if method_name in method_alphas:
                            ma = method_alphas[method_name]
                            render_interpretation(interpret_ff5_alpha(
                                ma["annual_alpha"], ma["t_stat"],
                                ma["p_value"], ma["r_squared"],
                            ))

# =========================================================================
# COSTS TAB
# =========================================================================
with tab_costs:
    st.subheader("Turnover & Transaction Costs")
    tc_col1, tc_col2 = st.columns(2)
    with tc_col1:
        fig = bar_chart(result["turnover"].dropna(), name="Monthly Turnover")
        st.plotly_chart(fig, use_container_width=True)
    with tc_col2:
        cost_bps = st.number_input("Cost per trade (bps)", value=10.0, step=5.0, key="tc_bps_port")
        # Base this on the GROSS series. result["monthly_returns"] is already
        # net of the backtest's own cost assumption, so using it here would
        # charge the book twice.
        _gross = result.get("monthly_returns_gross", result["monthly_returns"])
        ann_vol = _gross.std() * np.sqrt(12)
        tc = transaction_cost_drag(result["turnover"], cost_bps, ann_vol)
        perf_cost = compute_performance_metrics(_gross)
        gross_sr = perf_cost["SR"]
        net_sr = gross_sr - tc["Cost_SR"]
        metric_card_row([
            {"label": "Avg Monthly TO", "value": f"{tc['mean_monthly_turnover']:.1%}", "variant": "bar", "accent": C["primary"]},
            {"label": "TC Drag (annual)", "value": f"{tc['TC_annual']:.2%}", "variant": "bar", "accent": C["warning"]},
        ])
        metric_card_row([
            {"label": "Gross SR", "value": f"{gross_sr:.2f}", "variant": "bar", "accent": C["positive"]},
            {"label": "Net SR", "value": f"{net_sr:.2f}", "variant": "bar",
             "accent": C["positive"] if net_sr > 0.5 else C["warning"]},
        ])
        render_interpretation(interpret_turnover(
            tc["mean_monthly_turnover"], cost_bps, gross_sr,
        ))

    # --- Capacity: net Sharpe vs AUM (square-root market impact) ---
    st.markdown("---")
    st.subheader("Capacity — Net Sharpe vs AUM")
    st.caption(
        "Flat bps ignores that trading larger sizes moves the price. This uses a "
        "square-root market-impact model: as AUM grows, each name's trade is a "
        "bigger share of daily volume, so impact — and cost — rise."
    )
    cap_c1, cap_c2, cap_c3 = st.columns(3)
    with cap_c1:
        adv_musd = st.number_input("Daily volume / name ($M)", value=50.0, step=10.0, key="cap_adv")
    with cap_c2:
        spread_bps_in = st.number_input("Bid-ask spread (bps)", value=5.0, step=1.0, key="cap_spread")
    with cap_c3:
        impact_coef = st.number_input("Impact coefficient", value=0.10, step=0.05, key="cap_impact")

    aum_grid = np.logspace(6, 11, 40)  # $1M -> $100B
    cap = capacity_curve(
        perf_cost["Ann Return"], ann_vol, tc["mean_monthly_turnover"],
        aum_grid, adv_musd * 1e6, params.get("K", 10),
        spread_bps=spread_bps_in, impact_coef=impact_coef,
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cap["aum"], y=cap["net_sr"], mode="lines",
        line=dict(color=C["primary"], width=2), name="Net Sharpe",
    ))
    fig.add_hline(y=gross_sr, line_dash="dash", line_color=C["positive"],
                  annotation_text="Gross Sharpe")
    fig.update_layout(
        template="alpha", title="Strategy Capacity", xaxis_type="log",
        xaxis_title="AUM ($, log scale)", yaxis_title="Net Sharpe",
        height=380, margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    below = cap[cap["net_sr"] < gross_sr / 2]
    if len(below):
        cap_aum = below["aum"].iloc[0]
        st.caption(f"Estimated capacity (net Sharpe stays above ½ of gross): "
                   f"~${cap_aum / 1e9:.1f}B AUM.")
    else:
        st.caption("Net Sharpe stays above ½ of gross across the AUM range shown.")

# =========================================================================
# COMPARE METHODS TAB
# =========================================================================
with tab_compare:
    if len(method_results) >= 2:
        st.subheader("Method Comparison")
        comp = []
        cost_bps_cmp = st.number_input("Cost per trade (bps)", value=10.0, step=5.0, key="tc_bps_cmp")
        for m_name, port in method_results.items():
            p = compute_performance_metrics(port["monthly_returns"])
            p["Strategy"] = m_name
            p["Mean Turnover"] = port["turnover"].dropna().mean()
            ann_vol_m = port["monthly_returns"].std() * np.sqrt(12)
            tc_m = transaction_cost_drag(port["turnover"], cost_bps_cmp, ann_vol_m)
            p["TC Drag"] = tc_m["TC_annual"]
            p["Net SR"] = p["SR"] - tc_m["Cost_SR"]
            comp.append(p)
        comparison_table(comp)
    else:
        st.info("Select at least 2 methods in the sidebar to compare.")

render_next_steps("portfolio")
