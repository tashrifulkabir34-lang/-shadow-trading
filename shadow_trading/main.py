"""
Shadow Trading Replication Pipeline
=====================================
End-to-end runner:
  1. Generate synthetic data (calibrated to Mehta, Reeb & Zhao 2020)
  2. Run all regression tables (2, 4, 5, 6, 7)
  3. Compute economic significance and profit estimates
  4. Compute Shadow Trading Likelihood Scores
  5. Generate HTML tearsheet
"""

import os
import sys
import time

# Ensure src/ is on the path when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from shadow_trading.data.simulator import ShadowTradingSimulator, SimulationConfig
from shadow_trading.analysis.regressions import run_all_regressions
from shadow_trading.analysis.economic_significance import (
    compute_economic_significance,
    compute_profit_estimate,
)
from shadow_trading.models.stls import ShadowTradingLikelihoodScore
from shadow_trading.visualization.tearsheet import generate_tearsheet


def main():
    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print("  Shadow Trading Replication Pipeline")
    print("  Mehta, Reeb & Zhao (2020) — The Accounting Review")
    print("=" * 60)

    # ── Step 1: Generate data ────────────────────────────────── #
    print("\n[1/5] Generating synthetic dataset...")
    t0 = time.time()
    cfg = SimulationConfig(random_seed=42)
    sim = ShadowTradingSimulator(cfg)
    df = sim.generate_full_dataset()
    print(f"      {len(df):,} observations  "
          f"({len(df[df['car_sign']=='negative']):,} neg, "
          f"{len(df[df['car_sign']=='positive']):,} pos)")
    print(f"      {df['source_id'].nunique()} source firms, "
          f"{df['linked_id'].nunique()} linked firms")
    print(f"      Elapsed: {time.time()-t0:.1f}s")

    # ── Step 2: Regressions ──────────────────────────────────── #
    print("\n[2/5] Running regression tables (2, 4, 5, 6, 7)...")
    t0 = time.time()
    bundle = run_all_regressions(df)

    # Quick validation print for Table 2
    print("\n  Table 2 key coefficients (Paper targets in parentheses):")
    for r in bundle.table2:
        bp = r.coef.get("business_partner_car", float("nan"))
        bp_t = r.tstat.get("business_partner_car", float("nan"))
        comp = r.coef.get("competitor_car", float("nan"))
        comp_t = r.tstat.get("competitor_car", float("nan"))
        print(f"    {r.dependent_var[:25]:<25} [{r.car_sign}]  "
              f"BP={bp:+.4f}(t={bp_t:.2f})  "
              f"Comp={comp:+.4f}(t={comp_t:.2f})  "
              f"N={r.n_obs:,}")
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    # ── Step 3: Economic significance ────────────────────────── #
    print("\n[3/5] Computing economic significance...")
    econ_df = compute_economic_significance(bundle.table2, df)
    profit = compute_profit_estimate(df)
    print(f"      Profit range (paper): "
          f"${profit['profit_low_paper']:,.0f} – "
          f"${profit['profit_high_paper']:,.0f}")
    within_tol = econ_df["Within Tolerance"].sum() if "Within Tolerance" in econ_df.columns else "N/A"
    print(f"      Econ sig estimates within tolerance: {within_tol} / {len(econ_df)}")

    # ── Step 4: STLS ─────────────────────────────────────────── #
    print("\n[4/5] Computing Shadow Trading Likelihood Scores...")
    stls = ShadowTradingLikelihoodScore()
    df_scored = stls.fit_transform(df)
    high_risk = stls.high_risk_firms(df_scored)
    print(f"      High-risk events (STLS ≥ 75th pct): "
          f"{len(high_risk):,} ({len(high_risk)/len(df_scored)*100:.1f}%)")
    prohib_tbl = stls.compute_stls_by_prohibition(df_scored)
    no_prohib_mean = prohib_tbl.loc[prohib_tbl["prohibit"] == 0, "Mean STLS"].mean()
    prohib_mean = prohib_tbl.loc[prohib_tbl["prohibit"] == 1, "Mean STLS"].mean()
    print(f"      Mean STLS — No prohibition: {no_prohib_mean:.1f}, "
          f"Prohibition: {prohib_mean:.1f}  "
          f"(Paper: prohibition attenuates shadow trading)")

    # ── Step 5: Tearsheet ─────────────────────────────────────── #
    print("\n[5/5] Generating HTML tearsheet...")
    t0 = time.time()
    out = generate_tearsheet(
        df, bundle,
        output_path="outputs/shadow_trading_tearsheet.html"
    )
    print(f"      Written to: {out}  ({time.time()-t0:.1f}s)")

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)
    return bundle, df, df_scored


if __name__ == "__main__":
    main()
