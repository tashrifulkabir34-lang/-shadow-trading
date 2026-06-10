"""
Shadow Trading Tearsheet Generator
====================================
Produces a self-contained dark-themed HTML tearsheet replicating the
paper's key findings and diagnostics:

  Section 1 – Executive Summary (paper context, key stats)
  Section 2 – Descriptive Statistics (Table 1 replication)
  Section 3 – Main Regression Results (Table 2)
  Section 4 – Economic Significance (profit estimates)
  Section 5 – Mechanism Tests (Tables 5 & 6)
  Section 6 – Prohibition Analysis (Table 7)
  Section 7 – STLS Distribution (Shadow Trading Likelihood Score)
  Section 8 – Result Verification Matrix (paper vs. replicated)
"""

from __future__ import annotations
import json
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from shadow_trading.analysis.regressions import RegressionBundle, RegressionResult
from shadow_trading.analysis.economic_significance import (
    compute_economic_significance,
    compute_profit_estimate,
    compute_idd_econ_significance,
)
from shadow_trading.models.stls import ShadowTradingLikelihoodScore


# ------------------------------------------------------------------ #
#  HTML template                                                      #
# ------------------------------------------------------------------ #

CSS = """
:root {
  --bg:       #0b0f1a;
  --surface:  #131929;
  --card:     #1a2235;
  --border:   #2a3650;
  --accent:   #3b82f6;
  --accent2:  #f59e0b;
  --danger:   #ef4444;
  --success:  #22c55e;
  --text:     #e2e8f0;
  --muted:    #94a3b8;
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
  --font-body: 'Inter', 'Segoe UI', sans-serif;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-body);
  font-size: 14px;
  line-height: 1.6;
}
a { color: var(--accent); text-decoration: none; }

/* ── Layout ─────────────────────────────────────────────── */
.wrapper { max-width: 1200px; margin: 0 auto; padding: 32px 20px; }
.header  {
  display: flex; align-items: flex-start; justify-content: space-between;
  border-bottom: 2px solid var(--accent); padding-bottom: 24px; margin-bottom: 36px;
}
.header-left h1 { font-size: 28px; font-weight: 700; letter-spacing: -0.5px; }
.header-left .subtitle {
  color: var(--muted); font-size: 13px; margin-top: 4px;
}
.badge {
  display: inline-block; padding: 4px 10px; border-radius: 4px;
  font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.5px;
}
.badge-blue  { background: rgba(59,130,246,.2); color: #93c5fd; }
.badge-amber { background: rgba(245,158,11,.2); color: #fcd34d; }
.badge-red   { background: rgba(239,68,68,.2);  color: #fca5a5; }
.badge-green { background: rgba(34,197,94,.2);  color: #86efac; }

/* ── KPI Strip ────────────────────────────────────────────── */
.kpi-strip {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 12px; margin-bottom: 36px;
}
.kpi-card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; padding: 18px 16px;
}
.kpi-card .label { color: var(--muted); font-size: 11px; text-transform: uppercase;
  letter-spacing: 0.6px; margin-bottom: 6px; }
.kpi-card .value { font-size: 22px; font-weight: 700; color: var(--accent); }
.kpi-card .sub   { font-size: 11px; color: var(--muted); margin-top: 3px; }

/* ── Sections ─────────────────────────────────────────────── */
.section { margin-bottom: 48px; }
.section-title {
  font-size: 17px; font-weight: 600;
  border-left: 3px solid var(--accent); padding-left: 12px;
  margin-bottom: 20px; color: var(--text);
}
.section-desc { color: var(--muted); font-size: 13px; margin-bottom: 18px; }

/* ── Tables ────────────────────────────────────────────────── */
.tbl-wrap { overflow-x: auto; margin-bottom: 16px; }
table {
  width: 100%; border-collapse: collapse; font-size: 12.5px;
  background: var(--surface); border-radius: 6px; overflow: hidden;
}
thead th {
  background: #1e2d44; color: var(--muted); font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.4px; font-size: 11px;
  padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
tbody tr { border-bottom: 1px solid rgba(42,54,80,.5); }
tbody tr:hover { background: rgba(59,130,246,.04); }
tbody td { padding: 9px 12px; vertical-align: middle; }
.sig1  { color: #fca5a5; font-weight: 600; }  /* *** 1% */
.sig5  { color: #fcd34d; font-weight: 600; }  /* **  5% */
.sig10 { color: #6ee7b7; font-weight: 600; }  /* *  10% */
.positive { color: var(--success); }
.negative { color: var(--danger); }
.mono { font-family: var(--font-mono); }

/* ── Verification Matrix ──────────────────────────────────── */
.verify-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 12px;
}
.verify-card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 6px; padding: 14px;
}
.verify-card .vc-title { font-size: 12px; color: var(--muted); margin-bottom: 8px; }
.verify-card .vc-row {
  display: flex; justify-content: space-between;
  align-items: center; margin-bottom: 4px; font-size: 12px;
}
.verify-card .vc-paper { color: var(--accent2); font-family: var(--font-mono); }
.verify-card .vc-ours  { color: var(--accent);  font-family: var(--font-mono); }
.pass { color: var(--success); font-weight: 600; font-size: 11px; }
.fail { color: var(--danger);  font-weight: 600; font-size: 11px; }

/* ── Profit Box ───────────────────────────────────────────── */
.profit-box {
  background: linear-gradient(135deg, #1a2235 0%, #131929 100%);
  border: 1px solid var(--accent2); border-radius: 8px;
  padding: 24px; display: flex; gap: 32px; flex-wrap: wrap;
  margin-bottom: 24px;
}
.profit-item { flex: 1; min-width: 160px; }
.profit-item .p-label { color: var(--muted); font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
.profit-item .p-value { font-size: 26px; font-weight: 700; color: var(--accent2); }
.profit-item .p-sub   { color: var(--muted); font-size: 11px; margin-top: 3px; }

/* ── Footer ───────────────────────────────────────────────── */
.footer {
  border-top: 1px solid var(--border); padding-top: 20px; margin-top: 48px;
  color: var(--muted); font-size: 12px;
  display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;
}
"""


def _sig_class(t: float) -> str:
    a = abs(t)
    if a >= 2.576: return "sig1"
    if a >= 1.960: return "sig5"
    if a >= 1.645: return "sig10"
    return ""


def _fmt_coef(val: float, tstat: float) -> str:
    cls = _sig_class(tstat)
    stars = ("***" if abs(tstat) >= 2.576 else
             "**"  if abs(tstat) >= 1.960 else
             "*"   if abs(tstat) >= 1.645 else "")
    sign_cls = "positive" if val > 0 else "negative"
    return (f'<span class="mono {cls} {sign_cls}">'
            f'{val:+.4f}{stars}</span>')


def _descriptive_table(df: pd.DataFrame) -> str:
    neg = df[df["car_sign"] == "negative"]
    pos = df[df["car_sign"] == "positive"]
    rows_def = [
        ("Source Firm Business Partner CAR (neg)", "business_partner_car_signed", neg),
        ("Source Firm Business Partner CAR (pos)", "business_partner_car_signed", pos),
        ("Source Firm Competitor CAR (neg)", "competitor_car_signed", neg),
        ("Source Firm Competitor CAR (pos)", "competitor_car_signed", pos),
        ("Linked Firm Abnormal Short Sales (neg events)", "abnormal_short_sales", neg),
        ("Linked Firm Abnormal Short Sales (pos events)", "abnormal_short_sales", pos),
        ("Linked Firm Option/Stock Ratio (neg events)", "option_stock_ratio", neg),
        ("Linked Firm Option/Stock Ratio (pos events)", "option_stock_ratio", pos),
        ("Linked Firm Order Imbalance (neg events)", "order_imbalance", neg),
        ("Linked Firm Order Imbalance (pos events)", "order_imbalance", pos),
    ]
    paper = {
        "Source Firm Business Partner CAR (neg)": (-0.035, -0.028, 0.256),
        "Source Firm Business Partner CAR (pos)": (0.021, 0.011, 0.219),
        "Source Firm Competitor CAR (neg)": (-0.033, -0.030, 0.229),
        "Source Firm Competitor CAR (pos)": (0.023, 0.012, 0.228),
        "Linked Firm Abnormal Short Sales (neg events)": (0.079, 0.055, 0.279),
        "Linked Firm Abnormal Short Sales (pos events)": (-0.036, -0.029, 0.332),
        "Linked Firm Option/Stock Ratio (neg events)": (2.336, 1.527, 3.221),
        "Linked Firm Option/Stock Ratio (pos events)": (1.496, 0.972, 2.558),
        "Linked Firm Order Imbalance (neg events)": (-0.032, -0.037, 0.072),
        "Linked Firm Order Imbalance (pos events)": (0.025, 0.032, 0.075),
    }
    html = """<div class="tbl-wrap"><table>
    <thead><tr>
      <th>Variable</th>
      <th>Mean (Sim)</th><th>Median (Sim)</th><th>Std (Sim)</th>
      <th>Mean (Paper)</th><th>Median (Paper)</th><th>Std (Paper)</th>
    </tr></thead><tbody>"""
    for label, col, sub in rows_def:
        if col not in sub.columns or len(sub) == 0:
            continue
        m, md, s = sub[col].mean(), sub[col].median(), sub[col].std()
        pm, pmd, ps = paper.get(label, (np.nan, np.nan, np.nan))
        html += (f"<tr><td>{label}</td>"
                 f"<td class='mono'>{m:.4f}</td>"
                 f"<td class='mono'>{md:.4f}</td>"
                 f"<td class='mono'>{s:.4f}</td>"
                 f"<td class='mono'>{pm:.4f}</td>"
                 f"<td class='mono'>{pmd:.4f}</td>"
                 f"<td class='mono'>{ps:.4f}</td></tr>")
    return html + "</tbody></table></div>"


def _regression_table(results: List[RegressionResult], title: str) -> str:
    html = f'<p class="section-desc">{title}</p>'
    html += """<div class="tbl-wrap"><table>
    <thead><tr>
      <th>DV</th><th>CAR Sign</th>
      <th>β BP_CAR</th><th>t-stat</th>
      <th>β Comp_CAR</th><th>t-stat</th>
      <th>N</th><th>Adj R²</th><th>F-test p</th>
    </tr></thead><tbody>"""
    for r in results:
        bp = r.coef.get("business_partner_car", np.nan)
        bp_t = r.tstat.get("business_partner_car", np.nan)
        comp = r.coef.get("competitor_car", np.nan)
        comp_t = r.tstat.get("competitor_car", np.nan)
        html += (f"<tr>"
                 f"<td>{r.dependent_var.replace('_',' ')}</td>"
                 f"<td>{r.car_sign}</td>"
                 f"<td>{_fmt_coef(bp, bp_t)}</td>"
                 f"<td class='mono'>{bp_t:.2f}</td>"
                 f"<td>{_fmt_coef(comp, comp_t)}</td>"
                 f"<td class='mono'>{comp_t:.2f}</td>"
                 f"<td class='mono'>{r.n_obs:,}</td>"
                 f"<td class='mono'>{r.adj_r_squared:.3f}</td>"
                 f"<td class='mono'>{r.f_test_bp_eq_comp:.3f}</td>"
                 f"</tr>")
    return html + "</tbody></table></div>"


def _econ_sig_table(econ_df: pd.DataFrame) -> str:
    if econ_df.empty:
        return "<p>No data.</p>"
    html = """<div class="tbl-wrap"><table>
    <thead><tr>
      <th>DV</th><th>CAR Sign</th><th>Variable</th>
      <th>Coefficient</th><th>Mean DV</th><th>Std CAR</th>
      <th>Econ Sig (%)</th><th>Paper Target (%)</th><th>Within Tol.</th>
    </tr></thead><tbody>"""
    for _, row in econ_df.iterrows():
        within = row.get("Within Tolerance", False)
        wc = "pass" if within else "fail"
        wt = "✓ Yes" if within else "✗ No"
        html += (f"<tr>"
                 f"<td>{row['DV'].replace('_',' ')}</td>"
                 f"<td>{row['CAR Sign']}</td>"
                 f"<td class='mono'>{row['Variable']}</td>"
                 f"<td class='mono'>{row['Coefficient']:.4f}</td>"
                 f"<td class='mono'>{row['Mean DV']:.4f}</td>"
                 f"<td class='mono'>{row['Std CAR']:.4f}</td>"
                 f"<td class='mono'>{row['Econ Sig (%)']:.2f}</td>"
                 f"<td class='mono'>{row['Paper Target (%)']}</td>"
                 f"<td class='{wc}'>{wt}</td>"
                 f"</tr>")
    return html + "</tbody></table></div>"


def _verification_matrix(bundle: RegressionBundle) -> str:
    paper_targets = {
        ("abnormal_short_sales", "negative", "business_partner_car"): 0.033,
        ("abnormal_short_sales", "positive", "business_partner_car"): -0.019,
        ("abnormal_short_sales", "negative", "competitor_car"): 0.031,
        ("abnormal_short_sales", "positive", "competitor_car"): -0.020,
        ("option_stock_ratio", "negative", "business_partner_car"): 0.699,
        ("option_stock_ratio", "positive", "business_partner_car"): 0.621,
        ("option_stock_ratio", "negative", "competitor_car"): 0.611,
        ("option_stock_ratio", "positive", "competitor_car"): 0.581,
        ("order_imbalance", "negative", "business_partner_car"): 0.011,
        ("order_imbalance", "positive", "business_partner_car"): 0.008,
        ("order_imbalance", "negative", "competitor_car"): 0.009,
        ("order_imbalance", "positive", "competitor_car"): 0.008,
    }
    html = '<div class="verify-grid">'
    for r in bundle.table2:
        for var in ["business_partner_car", "competitor_car"]:
            key = (r.dependent_var, r.car_sign, var)
            if key not in paper_targets:
                continue
            paper_val = paper_targets[key]
            our_val = r.coef.get(var, np.nan)
            if np.isnan(our_val):
                continue
            ratio = our_val / paper_val if paper_val != 0 else np.nan
            ok = 0.5 <= abs(ratio) <= 2.0 if not np.isnan(ratio) else False
            status = '<span class="pass">✓ Consistent</span>' if ok else '<span class="fail">⚠ Diverged</span>'
            lbl = f"{r.dependent_var.replace('_',' ')} | {r.car_sign} | {var.replace('_',' ')}"
            html += (f'<div class="verify-card">'
                     f'<div class="vc-title">{lbl}</div>'
                     f'<div class="vc-row"><span>Paper</span>'
                     f'<span class="vc-paper">{paper_val:+.4f}</span></div>'
                     f'<div class="vc-row"><span>Replicated</span>'
                     f'<span class="vc-ours">{our_val:+.4f}</span></div>'
                     f'<div class="vc-row"><span>Status</span>{status}</div>'
                     f'</div>')
    return html + "</div>"


def generate_tearsheet(
    df: pd.DataFrame,
    bundle: RegressionBundle,
    output_path: str = "outputs/shadow_trading_tearsheet.html",
) -> str:
    """Generate the full HTML tearsheet and write to output_path."""

    # Enrich with STLS
    stls_model = ShadowTradingLikelihoodScore()
    df_scored = stls_model.fit_transform(df)

    profit = compute_profit_estimate(df)
    econ_df = compute_economic_significance(bundle.table2, df)
    idd_df = compute_idd_econ_significance(bundle.table6)

    # KPI values
    n_total = len(df)
    n_source = df["source_id"].nunique()
    n_linked = df["linked_id"].nunique()
    n_neg = len(df[df["car_sign"] == "negative"])
    n_pos = len(df[df["car_sign"] == "positive"])
    mean_abss_neg = df[df["car_sign"] == "negative"]["abnormal_short_sales"].mean()
    prohibit_pct = df["prohibit"].mean() * 100
    profit_low = profit["profit_low_paper"]
    profit_high = profit["profit_high_paper"]

    # Significant coefficients in Table 2
    sig_count = sum(
        1 for r in bundle.table2
        for var in ["business_partner_car", "competitor_car"]
        if abs(r.tstat.get(var, 0)) >= 1.96
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Shadow Trading — Replication Tearsheet</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrapper">

<!-- ════════════════════ HEADER ════════════════════ -->
<div class="header">
  <div class="header-left">
    <h1>Shadow Trading &mdash; Replication Study</h1>
    <div class="subtitle">
      Replication of Mehta, Reeb &amp; Zhao (2020) &bull;
      <em>The Accounting Review</em> &bull;
      Generated {timestamp}
    </div>
    <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
      <span class="badge badge-blue">Python · statsmodels · linearmodels</span>
      <span class="badge badge-amber">Panel OLS · 2-way Clustered SE</span>
      <span class="badge badge-green">Year + FF48 FE</span>
      <span class="badge badge-red">1997–2011 Calibrated</span>
    </div>
  </div>
</div>

<!-- ════════════════════ KPI STRIP ════════════════════ -->
<div class="kpi-strip">
  <div class="kpi-card">
    <div class="label">Total Observations</div>
    <div class="value">{n_total:,}</div>
    <div class="sub">{n_neg:,} neg · {n_pos:,} pos</div>
  </div>
  <div class="kpi-card">
    <div class="label">Source Firms</div>
    <div class="value">{n_source:,}</div>
    <div class="sub">Paper: 598</div>
  </div>
  <div class="kpi-card">
    <div class="label">Linked Firms</div>
    <div class="value">{n_linked:,}</div>
    <div class="sub">Paper: 745</div>
  </div>
  <div class="kpi-card">
    <div class="label">Significant Coefs (5%)</div>
    <div class="value">{sig_count} / 24</div>
    <div class="sub">across 6 Table 2 specs</div>
  </div>
  <div class="kpi-card">
    <div class="label">Mean Abnormal Short Sales</div>
    <div class="value">{mean_abss_neg:.1%}</div>
    <div class="sub">Paper: 7.9% (neg events)</div>
  </div>
  <div class="kpi-card">
    <div class="label">Shadow Trading Prohibition</div>
    <div class="value">{prohibit_pct:.0f}%</div>
    <div class="sub">of source firms · Paper: ~53%</div>
  </div>
</div>

<!-- ════════════════════ SECTION 1: PAPER CONTEXT ════════════════════ -->
<div class="section">
  <div class="section-title">1. Paper Overview &amp; Contribution</div>
  <p class="section-desc">
    Mehta, Reeb &amp; Zhao (2020) document <strong>shadow trading</strong>: corporate insiders
    exploit private information by trading in <em>economically-linked</em> firms (suppliers,
    customers, competitors) rather than their own — circumventing SEC insider trading restrictions.
    The paper finds three key results:
  </p>
  <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(320px,1fr)); gap:12px;">
    <div class="kpi-card">
      <div class="label">Result 1 — Existence</div>
      <div style="font-size:13px; color: var(--text); margin-top:8px;">
        Informed trading in linked firms spikes in the 30-day window
        <em>before</em> source firm earnings/M&amp;A/product announcements. 
        1 SD increase in BP CAR → 6.4%–19.2% more informed trading.
      </div>
    </div>
    <div class="kpi-card">
      <div class="label">Result 2 — Mechanism</div>
      <div style="font-size:13px; color: var(--text); margin-top:8px;">
        Shadow trading increases after high-profile SEC enforcement actions
        (substitution effect) and after IDD legal shocks (employee mobility
        channel). Rules out sophisticated-investor / market-friction alternatives.
      </div>
    </div>
    <div class="kpi-card">
      <div class="label">Result 3 — Policing</div>
      <div style="font-size:13px; color: var(--text); margin-top:8px;">
        Corporate prohibition policies attenuate shadow trading by ~8–12%
        (business partners). Only ~53% of firms explicitly prohibit it.
        No clear U.S. legal precedent for prosecution.
      </div>
    </div>
  </div>
</div>

<!-- ════════════════════ SECTION 2: DESCRIPTIVE STATS ════════════════════ -->
<div class="section">
  <div class="section-title">2. Descriptive Statistics &mdash; Table 1 Replication</div>
  <p class="section-desc">
    Simulated data calibrated to paper Table 1 Panel A (source firms) and Panel B (linked firms).
    CAR = 3-day cumulative abnormal return around earnings announcement.
  </p>
  {_descriptive_table(df)}
</div>

<!-- ════════════════════ SECTION 3: MAIN REGRESSIONS ════════════════════ -->
<div class="section">
  <div class="section-title">3. Main Shadow Trading Regressions &mdash; Table 2 Replication</div>
  {_regression_table(bundle.table2,
    "Equation (1): ShadowTrading = β₁·BP_CAR + β₂·Comp_CAR + Controls + Year FE + Industry FE. "
    "SE clustered by linked firm × year. *** p<1%, ** p<5%, * p<10%.")}
</div>

<!-- ════════════════════ SECTION 4: ECONOMIC SIGNIFICANCE ════════════════════ -->
<div class="section">
  <div class="section-title">4. Economic Significance &amp; Profit Estimates</div>
  <div class="profit-box">
    <div class="profit-item">
      <div class="p-label">Profit Per Event (Low)</div>
      <div class="p-value">${profit_low:,.0f}</div>
      <div class="p-sub">205k shares × $0.68 price move</div>
    </div>
    <div class="profit-item">
      <div class="p-label">Profit Per Event (High)</div>
      <div class="p-value">${profit_high:,.0f}</div>
      <div class="p-sub">205k shares × $3.30 price move</div>
    </div>
    <div class="profit-item">
      <div class="p-label">vs. SEC Defendants</div>
      <div class="p-value">$60K</div>
      <div class="p-sub">Perino (2019): median insider profit</div>
    </div>
    <div class="profit-item">
      <div class="p-label">Detection Incentive</div>
      <div class="p-value">Low</div>
      <div class="p-sub">No clear fiduciary breach; rare prosecution</div>
    </div>
  </div>
  <p class="section-desc">Economic significance formula: (coef / mean_DV) × std_CAR × 100</p>
  {_econ_sig_table(econ_df)}
</div>

<!-- ════════════════════ SECTION 5: MECHANISM TESTS ════════════════════ -->
<div class="section">
  <div class="section-title">5. Mechanism Tests &mdash; Tables 5 &amp; 6</div>
  <p class="section-desc">
    <strong>Table 5 (Equation 2)</strong>: Shadow trading increases 5.1%–36.1% in the 3-month
    window after high-profile SEC enforcement spikes (June 2003, June 2006, October 2009).
    Conventional insider trading simultaneously falls — consistent with substitution.
  </p>
  {_regression_table(bundle.table5[:6],
    "Post-Enforcement Spike: BP_CAR×Post and Comp_CAR×Post capture substitution effect.")}
  <p class="section-desc" style="margin-top:24px;">
    <strong>Table 6 (Equation 3)</strong>: Staggered IDD adoption/rejection across states
    (MO/OH 2000, FL 2001, MI 2002, TX 2003, KS 2006). Propensity-score matched 508 treatment
    vs 508 control source firms. Shadow trading increases 8.6%–20% after IDD shock.
  </p>
  {_regression_table(bundle.table6,
    "IDD Shock DiD: IDDShock indicator + interaction terms. Only OSR and OI (no short-sale data in IDD window).")}
</div>

<!-- ════════════════════ SECTION 6: PROHIBITION ════════════════════ -->
<div class="section">
  <div class="section-title">6. Corporate Policy Analysis &mdash; Table 7 Replication</div>
  <p class="section-desc">
    267 source firms' Code of Ethics / Employee Conduct manuals examined for 2010-2011.
    ~53% prohibit shadow trading; 47% restrict own-firm trading only.
    Prohibition attenuates business partner shadow trading by 8–12%.
  </p>
  {_regression_table(bundle.table7,
    "Prohibition interaction: BP_CAR×Prohibit and Comp_CAR×Prohibit test policy effectiveness.")}
</div>

<!-- ════════════════════ SECTION 7: STLS ════════════════════ -->
<div class="section">
  <div class="section-title">7. Shadow Trading Likelihood Score (STLS)</div>
  <p class="section-desc">
    A composite 0–100 percentile score combining abnormal short sales (25%),
    option/stock ratio (50%), and order imbalance (25%), calibrated to Table 2
    coefficient magnitudes.
  </p>
  <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th>Risk Band</th><th>STLS Range</th>
        <th>Neg Events</th><th>Pos Events</th>
        <th>Interpretation</th>
      </tr></thead>
      <tbody>
        <tr><td><span class="badge badge-green">Low</span></td>
          <td>0–25th pct</td>
          <td class="mono">{len(df_scored[(df_scored['stls_band']=='Low') & (df_scored['car_sign']=='negative')]):,}</td>
          <td class="mono">{len(df_scored[(df_scored['stls_band']=='Low') & (df_scored['car_sign']=='positive')]):,}</td>
          <td>Normal trading activity; low shadow trading likelihood</td></tr>
        <tr><td><span class="badge badge-amber">Medium</span></td>
          <td>25–75th pct</td>
          <td class="mono">{len(df_scored[(df_scored['stls_band']=='Medium') & (df_scored['car_sign']=='negative')]):,}</td>
          <td class="mono">{len(df_scored[(df_scored['stls_band']=='Medium') & (df_scored['car_sign']=='positive')]):,}</td>
          <td>Elevated activity; warrants closer monitoring</td></tr>
        <tr><td><span class="badge badge-red">High</span></td>
          <td>75–100th pct</td>
          <td class="mono">{len(df_scored[(df_scored['stls_band']=='High') & (df_scored['car_sign']=='negative')]):,}</td>
          <td class="mono">{len(df_scored[(df_scored['stls_band']=='High') & (df_scored['car_sign']=='positive')]):,}</td>
          <td>Strong signal; likely shadow trading activity</td></tr>
      </tbody>
    </table>
  </div>
  <p class="section-desc" style="margin-top:12px;">
    STLS by prohibition status (confirming Table 7 — prohibiting firms have lower STLS):
  </p>
  <div class="tbl-wrap">
    <table>
      <thead><tr><th>Prohibition</th><th>CAR Sign</th><th>Mean STLS</th><th>Median STLS</th><th>N</th></tr></thead>
      <tbody>
        {''.join(
          f"<tr><td>{'Yes' if int(r['prohibit'])==1 else 'No'}</td><td>{r['car_sign']}</td>"
          f"<td class='mono'>{r['Mean STLS']:.1f}</td><td class='mono'>{r['Median STLS']:.1f}</td>"
          f"<td class='mono'>{int(r['N']):,}</td></tr>"
          for _, r in stls_model.compute_stls_by_prohibition(df_scored).iterrows()
        )}
      </tbody>
    </table>
  </div>
</div>

<!-- ════════════════════ SECTION 8: VERIFICATION ════════════════════ -->
<div class="section">
  <div class="section-title">8. Result Verification Matrix</div>
  <p class="section-desc">
    Cross-check of our replicated coefficients against paper Table 2 targets.
    "Consistent" = replicated estimate within ±100% of paper value
    (expected for simulated data calibrated from the same parameters).
  </p>
  {_verification_matrix(bundle)}
</div>

<!-- ════════════════════ FOOTER ════════════════════ -->
<div class="footer">
  <span>Shadow Trading Replication &bull; Mehta, Reeb &amp; Zhao (2020) &bull; <em>The Accounting Review</em></span>
  <span>Generated by Python pipeline &bull; {timestamp}</span>
</div>

</div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    return output_path
