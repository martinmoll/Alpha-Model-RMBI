# dashboard/pages/4_Backtest_Results.py
"""Page 4: Backtest Results & Diagnostics — tabbed layout."""

import streamlit as st
import pandas as pd
import numpy as np
from core.diagnostics import (
    compute_performance_metrics, compute_ic_stats, fundamental_law, feature_ic,
    compute_r2_oos, bootstrap_sharpe_ci, bootstrap_alpha_ci, multiple_testing_hurdle,
    probabilistic_sharpe_ratio, deflated_sharpe_ratio,
    probability_of_backtest_overfitting, survivorship_premium,
    return_by_vol_decile,
)
from core.risk import factor_alpha
from components.charts import (
    cumulative_wealth_chart, drawdown_chart, monthly_heatmap,
    rolling_metric_chart, bar_chart, STYLE, PIN_COLORS,
)
from components.metrics import metric_card, metric_card_row, metric_row, banner, comparison_table
from components.theory import theory_section
from components.interpretations import (
    render_interpretation, interpret_sharpe, interpret_max_drawdown,
    interpret_ff5_alpha, interpret_r2_oos, interpret_turnover,
)
from components.workflow import render_workflow_status, render_empty_state, render_next_steps, config_label
from components.theme import inject_theme, COLORS, FONT_MONO, FONT_SANS
from core.currency import to_nok_unhedged
from core.data_loader import region_for_label
import plotly.graph_objects as go

st.set_page_config(page_title="Backtest Results", layout="wide")
inject_theme()

if render_empty_state("results"):
    st.stop()

result = st.session_state.get("backtest_result")
market = st.session_state.get("market_monthly")
pinned = st.session_state.get("pinned_configs", [])


@st.cache_data(show_spinner=False)
def _load_usdnok():
    """Monthly USDNOK return series, cached across reruns."""
    from pipeline.fetchers.fx import fetch_usdnok_monthly
    return fetch_usdnok_monthly()

# --- Config Selector ---
# Selected by index (0 = current run, 1..N = pinned) so two pins with identical
# labels never collide the way a label-keyed map does.
current_config = {
    "result": result,
    "params": st.session_state.get("backtest_params", {}),
    "predictions": st.session_state.get("backtest_predictions"),
}
configs = [current_config] + list(pinned)


def _config_name(i):
    return "Current Run" if i == 0 else pinned[i - 1]["label"]


# --- Header ---
active_idx = 0
C = COLORS
params = st.session_state.get("backtest_params", {})

head_left, head_right = st.columns([2, 1])
with head_left:
    st.markdown(
        f'<h1 style="font-family:{FONT_SANS};font-size:28px;font-weight:700;'
        f'color:{C["text"]};margin:0;">Backtest Results</h1>',
        unsafe_allow_html=True,
    )
    model_name = params.get("model_name", "—")
    strategy = params.get("strategy_type", "long_only").replace("_", "-")
    window = params.get("window_type", "expanding")
    rets_all = result["monthly_returns"]
    subtitle = f"{model_name} &middot; {strategy} &middot; {window} &middot; {rets_all.index.min()} &rarr; {rets_all.index.max()}"
    st.markdown(
        f'<p style="font-family:{FONT_SANS};font-size:13px;color:{C["text_secondary"]};'
        f'margin:4px 0 0;">{subtitle}</p>',
        unsafe_allow_html=True,
    )
with head_right:
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        if len(configs) > 1:
            active_idx = st.selectbox(
                "View config", range(len(configs)), format_func=_config_name,
                label_visibility="collapsed",
            )
    with r_col2:
        pin_clicked = st.button("Pin to compare", type="primary")

render_workflow_status("results")

# --- Survivorship warning -------------------------------------------------
# The universe is today's index constituents projected backwards, so every
# number on this page is measured on a book that could not have been held at
# the time. Show the size of that head start before any performance figure.
_panel = st.session_state.get("df")
_oos = st.session_state.get("backtest_params", {}).get("oos_start")
if _panel is not None:
    _surv = survivorship_premium(_panel, start=_oos)
    if _surv:
        banner(
            "warning",
            f"<b>Survivorship bias: +{_surv['gap_ann']:.1%} per year.</b> "
            f"Holding <i>every</i> name in this universe returned "
            f"{_surv['panel_ann']:.1%}/yr against a benchmark of "
            f"{_surv['bench_ann']:.1%}/yr, before any model. "
            f"Universe: {_surv['names_first']} names at the start, "
            f"{_surv['names_last']} at the end.",
            detail=(
                "The universe is **today's** index constituents projected "
                "backwards. Companies that were dropped, acquired or went "
                "bankrupt are absent, so the backtest picks only from names "
                "that survived. No investor could have held this list at the "
                "time.\n\n"
                "Treat the gap above as a **floor**, not a point estimate. It "
                "understates the true effect, because the same surviving names "
                "also dominate the training data. Subtract it from any Sharpe "
                "or return figure on this page before you act on it.\n\n"
                "Fixing this properly needs point-in-time index membership and "
                "delisting returns — see `ROADMAP.md` Tier 1."
            ),
        )

        with st.expander("Where the bias lives: return by volatility decile"):
            _dec = return_by_vol_decile(
                _panel,
                holdings=(result or {}).get("holdings"),
                start=_oos,
            )
            if not _dec.empty:
                st.caption(
                    "Deciles are formed within each month, so the split itself "
                    "carries no look-ahead. In a point-in-time universe the "
                    "highest-volatility decile should **not** reliably "
                    "outperform — those are the names that blow up, and the "
                    "ones that did are missing here. A large positive spread "
                    "is measuring the survivor filter, not a risk premium."
                )
                _show = _dec.copy()
                _show["ann_return"] = (_show["ann_return"] * 100).round(1)
                if "share_of_book" in _show.columns:
                    _show["share_of_book"] = (_show["share_of_book"] * 100).round(1)
                st.dataframe(
                    _show.rename(columns={
                        "ann_return": "Universe return %/yr",
                        "n_obs": "Observations",
                        "share_of_book": "Share of book %",
                    }),
                    use_container_width=True,
                )
                _spread = (_dec["ann_return"].iloc[-1] - _dec["ann_return"].iloc[0]) * 100
                st.caption(f"Top decile minus bottom: **{_spread:.1f}** pct pts/yr.")

active = configs[active_idx]
active_result = active["result"]
active_params = active.get("params", {})
is_current_run = active_idx == 0

rets = active_result["monthly_returns"]
ic = active_result["ic"]
turnover = active_result["turnover"]

# --- Display Controls (sidebar) ---
st.sidebar.header("Display Controls")
oos_months = sorted(rets.index)
display_start = st.sidebar.select_slider("Display from", options=oos_months, value=oos_months[0])
start_value = st.sidebar.number_input("Starting value ($)", min_value=1, value=10000, step=1000)
cash_flow = st.sidebar.number_input("Cash flow/period ($)", value=0, step=100)

# Reporting currency — a Norwegian investor funding USD purchases with NOK also
# earns the USDNOK move. Only offered for the USD (US) book; the Oslo book is
# already in NOK. Hedged/attribution follow-ups live in FUTURE_IMPROVEMENTS.md.
fx_ret = None
if region_for_label(st.session_state.get("active_region", "")) == "US":
    currency_mode = st.sidebar.radio(
        "Reporting currency",
        ["USD", "NOK (unhedged)"],
        help="Restate returns in NOK by compounding each month's USD return with "
             "the USDNOK move. Unhedged — no forward hedging applied.",
    )
    if currency_mode.startswith("NOK"):
        try:
            fx_ret = _load_usdnok()
        except Exception as e:  # network fetch can fail — degrade to USD
            st.sidebar.warning(f"USDNOK fetch failed ({e}); showing USD.")

rets = rets[rets.index >= display_start]
ic = ic[ic.index >= display_start]
turnover = turnover[turnover.index >= display_start]

if fx_ret is not None:
    rets = to_nok_unhedged(rets, fx_ret)

# --- KPI data ---
perf = compute_performance_metrics(rets)
ff5 = st.session_state.get("ff5_factors")
alpha_res = factor_alpha(rets, ff5) if ff5 is not None else None

# --- Bootstrap CIs ---
sr_ci = bootstrap_sharpe_ci(rets)
alpha_ci = bootstrap_alpha_ci(rets, ff5) if ff5 is not None else None

# --- Tabs ---
tab_overview, tab_signal, tab_attribution, tab_costs, tab_compare = st.tabs(
    ["Overview", "Signal quality", "Attribution", "Turnover & costs", "Compare"]
)

# =========================================================================
# OVERVIEW TAB
# =========================================================================
with tab_overview:
    if fx_ret is not None:
        banner("info",
               'Returns shown in <span class="mono">NOK</span> — unhedged',
               "Restated for a Norwegian investor: each month compounds the USD "
               "return with the USDNOK move. The FF5 alpha still regresses on USD "
               "factors, so in NOK mode it also absorbs currency swings — read it "
               "as indicative. Hedged NOK and full FX attribution are planned "
               "(see FUTURE_IMPROVEMENTS.md).")

    # KPI row
    sr_accent = C["positive"] if perf["SR"] >= 1.0 else C["primary"] if perf["SR"] >= 0.5 else C["warning"]
    ret_accent = C["positive"] if perf["Ann Return"] > 0 else C["negative"]
    vol_accent = C["primary"]
    mdd_accent = C["negative"]
    calmar_accent = C["positive"] if perf["Calmar"] >= 1.0 else C["warning"]

    sr_ci_str = f"95% CI [{sr_ci['lo']:.2f}, {sr_ci['hi']:.2f}]" if not np.isnan(sr_ci["lo"]) else None

    kpi_cards = [
        {"label": "Sharpe", "value": f"{perf['SR']:.2f}", "accent": sr_accent, "variant": "bar",
         "delta": sr_ci_str, "delta_color": C["text_secondary"]},
        {"label": "Ann. Return", "value": f"{perf['Ann Return']:.1%}", "accent": ret_accent, "variant": "bar"},
        {"label": "Ann. Vol", "value": f"{perf['Ann Vol']:.1%}", "accent": vol_accent, "variant": "bar"},
        {"label": "Max DD", "value": f"{perf['MDD']:.1%}", "accent": mdd_accent, "variant": "bar"},
        {"label": "Calmar", "value": f"{perf['Calmar']:.2f}" if not np.isnan(perf['Calmar']) else "N/A",
         "accent": calmar_accent, "variant": "bar"},
    ]
    if alpha_res is not None:
        sig = "**" if alpha_res["p_value"] < 0.01 else "*" if alpha_res["p_value"] < 0.05 else ""
        alpha_accent = C["positive"] if alpha_res["annual_alpha"] > 0 and alpha_res["p_value"] < 0.05 else C["primary"]
        alpha_ci_str = None
        if alpha_ci is not None and not np.isnan(alpha_ci["lo"]):
            alpha_ci_str = f"95% CI [{alpha_ci['lo']:.1%}, {alpha_ci['hi']:.1%}]"
        kpi_cards.append({
            "label": "FF5 Alpha",
            "value": f"{alpha_res['annual_alpha']:.1%}{sig}",
            "accent": alpha_accent,
            "variant": "bar",
            "delta": alpha_ci_str,
            "delta_color": C["text_secondary"],
        })
    metric_card_row(kpi_cards)

    # Compact interpretation banner — include CI in headline
    sr_val = perf["SR"]
    ci_note = f' <span class="mono">[{sr_ci["lo"]:.2f}, {sr_ci["hi"]:.2f}]</span>' if not np.isnan(sr_ci["lo"]) else ""
    interp = interpret_sharpe(sr_val)
    if interp and interp.get("text"):
        sentences = interp["text"].split(". ", 1)
        headline = sentences[0].rstrip(".") + ci_note
        detail = sentences[1] if len(sentences) > 1 else None
        zero_in_ci = not np.isnan(sr_ci["lo"]) and sr_ci["lo"] <= 0
        if zero_in_ci and detail:
            detail += (" Note: the 95% bootstrap CI includes zero, meaning the positive "
                       "Sharpe is not statistically distinguishable from chance at this sample size.")
        banner(interp["level"], headline, detail)

    # Probabilistic / Deflated Sharpe — is the Sharpe real given noise & #trials?
    psr = probabilistic_sharpe_ratio(rets)
    psr_cards = [{
        "label": "Prob. Sharpe > 0",
        "value": f"{psr:.0%}" if not np.isnan(psr) else "N/A",
        "accent": C["positive"] if (not np.isnan(psr) and psr > 0.95) else C["warning"],
        "variant": "bar", "delta": "P(true Sharpe > 0)", "delta_color": C["text_secondary"],
    }]
    if pinned:
        trial_srs = []
        for cfg in [current_config] + list(pinned):
            cr = cfg["result"]["monthly_returns"]
            cr = cr[cr.index >= display_start].dropna()
            if cr.std() > 0:
                trial_srs.append(cr.mean() / cr.std())
        dsr_res = deflated_sharpe_ratio(rets, trial_srs)
        dsr = dsr_res["dsr"]
        psr_cards.append({
            "label": "Deflated Sharpe",
            "value": f"{dsr:.0%}" if not np.isnan(dsr) else "N/A",
            "accent": C["positive"] if (not np.isnan(dsr) and dsr > 0.95) else C["warning"],
            "variant": "bar", "delta": f"deflated for {dsr_res['n_trials']} trials",
            "delta_color": C["text_secondary"],
        })
    metric_card_row(psr_cards)

    if alpha_res is not None:
        # Multiple-testing nudge: comparing pinned configs inflates false positives.
        if pinned:
            n_trials = len(pinned) + 1  # pinned configs + the current run
            hurdle = multiple_testing_hurdle(n_trials)
            t_obs = abs(alpha_res["t_stat"])
            if t_obs >= hurdle:
                banner("success",
                       f'Comparing <span class="mono">{n_trials}</span> configs — alpha t-stat '
                       f'<span class="mono">{alpha_res["t_stat"]:.2f}</span> still clears the '
                       f'multiple-testing hurdle (|t| &gt; <span class="mono">{hurdle:.2f}</span>)',
                       "Bonferroni raises the bar when you pick the best of several strategies. "
                       "This alpha survives the correction — evidence it is not just the luckiest "
                       "of the configs you tried.")
            else:
                banner("warning",
                       f'Comparing <span class="mono">{n_trials}</span> configs — alpha t-stat '
                       f'<span class="mono">{alpha_res["t_stat"]:.2f}</span> is below the '
                       f'multiple-testing hurdle (|t| &gt; <span class="mono">{hurdle:.2f}</span>)',
                       "Every config you pin is another test, so the naive |t| &gt; 1.96 bar no "
                       "longer holds. Bonferroni requires this tighter hurdle to keep the "
                       "family-wise false-positive rate at 5% — this alpha may be the luckiest "
                       "draw rather than genuine skill. See the theory below.")
        theory_section("Is It Really Alpha? Measuring Statistical Confidence", "alpha_significance")
        theory_section("Jensen's Alpha — Do You Have Skill?", "jensens_alpha")

    # Charts: wealth + drawdown left, heatmap + feature importance right
    st.markdown("---")
    chart_left, chart_right = st.columns([2, 1])

    returns_dict = {"Strategy": rets}
    if market is not None:
        spy = market.loc[market.index >= display_start, "spy_ret"]
        if fx_ret is not None:
            spy = to_nok_unhedged(spy, fx_ret)
        returns_dict["SPY"] = spy
    for p in pinned:
        p_rets = p["result"]["monthly_returns"]
        p_rets = p_rets[p_rets.index >= display_start]
        if fx_ret is not None:
            p_rets = to_nok_unhedged(p_rets, fx_ret)
        returns_dict[p["label"]] = p_rets

    with chart_left:
        fig = cumulative_wealth_chart(returns_dict, start_value, cash_flow)
        st.plotly_chart(fig, use_container_width=True, key="overview_wealth")
        fig = drawdown_chart(returns_dict, start_value)
        st.plotly_chart(fig, use_container_width=True, key="overview_dd")

    with chart_right:
        fig = monthly_heatmap(rets)
        st.plotly_chart(fig, use_container_width=True, key="overview_heatmap")

        importance = active.get("feature_importance", st.session_state.get("backtest_feature_importance"))
        if importance is not None and len(importance) > 0:
            top_imp = importance.nlargest(15, "importance")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top_imp["importance"].values, y=top_imp.index,
                orientation="h", marker_color=C["primary"],
            ))
            fig.update_layout(
                template="alpha", title="Feature Importance (Top 15)",
                height=400, margin=dict(t=40, b=40, l=150),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True, key="overview_feat_imp")

# =========================================================================
# SIGNAL QUALITY TAB
# =========================================================================
with tab_signal:
    theory_section("Information Coefficient & Fundamental Law", "ic_and_fundamental_law")

    ic_col1, ic_col2 = st.columns(2)
    with ic_col1:
        fig = bar_chart(ic.dropna(), name="Monthly IC")
        st.plotly_chart(fig, use_container_width=True, key="signal_ic_bar")
    with ic_col2:
        cum_ic = ic.dropna().cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cum_ic.index, y=cum_ic.values, name="Cumulative IC",
            line=dict(color=C["primary"], width=2),
        ))
        fig.update_layout(
            template="alpha", title="Cumulative IC", height=400,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True, key="signal_cum_ic")

    ic_stats = compute_ic_stats(ic)
    metric_card_row([
        {"label": "Mean IC", "value": f"{ic_stats['mean_ic']:.4f}",
         "accent": C["positive"] if ic_stats["mean_ic"] > 0.03 else C["warning"], "variant": "bar"},
        {"label": "IC t-stat", "value": f"{ic_stats['ic_tstat']:.2f}",
         "accent": C["positive"] if ic_stats["ic_tstat"] > 2 else C["warning"], "variant": "bar"},
        {"label": "ICIR", "value": f"{ic_stats['icir']:.3f}",
         "accent": C["positive"] if ic_stats["icir"] > 0.3 else C["warning"], "variant": "bar"},
        {"label": "Hit Rate", "value": f"{ic_stats['hit_rate']:.1%}",
         "accent": C["positive"] if ic_stats["hit_rate"] > 0.55 else C["warning"], "variant": "bar"},
    ])

    t = ic_stats["ic_tstat"]
    if abs(t) >= 2.58:
        banner("success",
               f'IC t-stat = <span class="mono">{t:.2f}</span> — highly significant (p < 0.01)',
               "The model's predictive signal is statistically reliable and very unlikely "
               "to be due to chance. This is strong evidence of genuine forecasting ability.")
    elif abs(t) >= 1.96:
        banner("info",
               f'IC t-stat = <span class="mono">{t:.2f}</span> — significant at 5% level',
               "The model's signal is statistically distinguishable from zero, suggesting "
               "real predictive content. Verify with out-of-sample performance before relying on it.")
    elif abs(t) >= 1.65:
        banner("warning",
               f'IC t-stat = <span class="mono">{t:.2f}</span> — marginally significant',
               "There is weak evidence of a signal. The IC could plausibly be zero — "
               "consider increasing the backtest window or simplifying the model.")
    else:
        banner("error",
               f'IC t-stat = <span class="mono">{t:.2f}</span> — not significant',
               "The model's predictions are statistically indistinguishable from random. "
               "Consider changing the feature set, model, or retraining frequency.")

    # Fundamental Law
    st.markdown("---")
    st.subheader("Fundamental Law of Active Management")
    fl_col1, fl_col2 = st.columns(2)
    with fl_col1:
        sr_target = st.number_input("SR Target", value=1.0, step=0.1, key="fl_sr_target")
        tc_bps = st.number_input("Transaction Cost (bps)", value=10.0, step=5.0, key="fl_tc_bps")
    with fl_col2:
        K_val = active_params.get("K", 10)
        fl = fundamental_law(ic_stats["mean_ic"], K_val, sr_target=sr_target, tc_bps=tc_bps)
        metric_card_row([
            {"label": "Breadth", "value": f"{fl['BR_nominal']}", "variant": "minimal"},
            {"label": "IR Upper Bound", "value": f"{fl['IR_upper_bound']:.3f}", "variant": "minimal"},
        ])
        metric_card_row([
            {"label": "IC Required", "value": f"{fl['IC_required']:.4f}", "variant": "minimal"},
            {"label": "Cost SR", "value": f"{fl['Cost_SR']:.3f}", "variant": "minimal"},
        ])

    # R2 OOS
    st.markdown("---")
    st.subheader("Out-of-Sample R² (Campbell & Thompson 2008)")
    predictions = active.get("predictions")
    if predictions:
        r2_oos = compute_r2_oos(predictions)
        r2_col1, r2_col2 = st.columns([1, 3])
        with r2_col1:
            metric_card("R²_OOS", f"{r2_oos:.4f}" if not np.isnan(r2_oos) else "N/A",
                        accent=C["positive"] if r2_oos > 0 else C["warning"], variant="bar")
        with r2_col2:
            render_interpretation(interpret_r2_oos(r2_oos))
    else:
        st.caption("R²_OOS is unavailable for this configuration (no stored predictions).")

# =========================================================================
# ATTRIBUTION TAB
# =========================================================================
with tab_attribution:
    render_interpretation(interpret_max_drawdown(perf["MDD"], perf["Calmar"]))
    if alpha_res is not None:
        render_interpretation(interpret_ff5_alpha(
            alpha_res["annual_alpha"], alpha_res["t_stat"],
            alpha_res["p_value"], alpha_res["r_squared"],
        ))

    # Feature Importance (full view)
    theory_section("Feature Importance", "feature_importance")
    importance = active.get("feature_importance", st.session_state.get("backtest_feature_importance"))
    if not is_current_run and importance is None:
        st.info("Feature importance is only available for the most recent run.")
    if importance is not None and len(importance) > 0:
        st.subheader("Feature Importance (Top 20)")
        top_imp = importance.nlargest(20, "importance")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_imp["importance"].values, y=top_imp.index,
            orientation="h", marker_color=C["primary"],
        ))
        fig.update_layout(
            template="alpha", title="Feature Importance",
            height=500, margin=dict(t=40, b=40, l=150),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True, key="attrib_feat_imp")

# =========================================================================
# TURNOVER & COSTS TAB
# =========================================================================
with tab_costs:
    hm_col1, hm_col2 = st.columns(2)
    with hm_col1:
        fig = monthly_heatmap(rets)
        st.plotly_chart(fig, use_container_width=True, key="costs_heatmap")
    with hm_col2:
        fig = bar_chart(turnover.dropna(), name="Monthly Turnover")
        st.plotly_chart(fig, use_container_width=True, key="costs_turnover")

    mean_to = turnover.dropna().mean()
    render_interpretation(interpret_turnover(
        mean_to, 10.0, perf["SR"],
        ic_mean=compute_ic_stats(ic)["mean_ic"],
        K=active_params.get("K", 10),
    ))

# =========================================================================
# COMPARE TAB
# =========================================================================
with tab_compare:
    if pinned:
        st.subheader("Strategy Comparison")
        configs = []

        cur_result = st.session_state.get("backtest_result")
        cur_rets = cur_result["monthly_returns"]
        cur_rets = cur_rets[cur_rets.index >= display_start]
        if fx_ret is not None:
            cur_rets = to_nok_unhedged(cur_rets, fx_ret)
        cur_ic = cur_result["ic"]
        cur_ic = cur_ic[cur_ic.index >= display_start]
        cur_perf = compute_performance_metrics(cur_rets)
        cur_perf["Strategy"] = "Current Run"
        cur_perf["Mean IC"] = cur_ic.mean()
        cur_perf["Mean Turnover"] = cur_result["turnover"].dropna().mean()
        configs.append(cur_perf)

        for p in pinned:
            p_rets = p["result"]["monthly_returns"]
            p_rets = p_rets[p_rets.index >= display_start]
            if fx_ret is not None:
                p_rets = to_nok_unhedged(p_rets, fx_ret)
            p_perf = compute_performance_metrics(p_rets)
            p_perf["Strategy"] = p["label"]
            p_ic = p["result"]["ic"]
            p_perf["Mean IC"] = p_ic[p_ic.index >= display_start].mean()
            p_perf["Mean Turnover"] = p["result"]["turnover"].dropna().mean()
            configs.append(p_perf)

        comparison_table(configs)

        # --- Probability of Backtest Overfitting (CSCV) ---
        ret_cols = {}
        _cur = st.session_state.get("backtest_result")["monthly_returns"]
        ret_cols["Current Run"] = _cur[_cur.index >= display_start]
        for p in pinned:
            pr = p["result"]["monthly_returns"]
            ret_cols[p["label"]] = pr[pr.index >= display_start]
        pbo = probability_of_backtest_overfitting(pd.DataFrame(ret_cols))

        st.markdown("---")
        st.subheader("Probability of Backtest Overfitting")
        if not np.isnan(pbo):
            metric_card_row([{
                "label": "PBO", "value": f"{pbo:.0%}",
                "accent": C["negative"] if pbo > 0.5 else C["positive"], "variant": "bar",
                "delta": "P(backtest winner disappoints out-of-sample)",
                "delta_color": C["text_secondary"],
            }])
            banner("warning" if pbo > 0.5 else "success",
                   f'PBO = <span class="mono">{pbo:.0%}</span>',
                   "Across combinatorial in-sample / out-of-sample splits, how often the "
                   "in-sample-best config landed below the out-of-sample median. Above 50% "
                   "means picking the backtest winner is likely overfitting. Most reliable "
                   "with several configs and a long history.")
        else:
            st.caption("Need at least 2 configs and enough history to estimate PBO.")
    else:
        st.info("Pin configurations from the Alpha Model Lab to compare them here.")

# --- Pin action ---
if pin_clicked:
    cur_result = st.session_state.get("backtest_result")
    cur_params = st.session_state.get("backtest_params")
    cur_pinned = st.session_state.get("pinned_configs", [])
    if cur_result is not None and len(cur_pinned) < 4:
        label = config_label(cur_params)
        cur_pinned.append({
            "label": label, "result": cur_result, "params": cur_params,
            "predictions": st.session_state.get("backtest_predictions"),
        })
        st.session_state.pinned_configs = cur_pinned
        st.rerun()

render_next_steps("results")
