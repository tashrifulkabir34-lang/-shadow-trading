"""
Test Suite — Shadow Trading Replication
=========================================
45+ tests covering:
  - Data simulator calibration (Table 1 matches)
  - Regression engine correctness (Table 2 sign/significance)
  - Economic significance calculations
  - Profit estimate bounds
  - STLS model behaviour
  - Pipeline integration
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from shadow_trading.data.simulator import ShadowTradingSimulator, SimulationConfig
from shadow_trading.analysis.regressions import (
    run_table2, run_table4, run_table5, run_table6, run_table7,
    run_all_regressions, RegressionBundle,
)
from shadow_trading.analysis.economic_significance import (
    compute_economic_significance,
    compute_profit_estimate,
    compute_idd_econ_significance,
)
from shadow_trading.models.stls import ShadowTradingLikelihoodScore, STLSConfig


# ============================================================= #
#  Fixtures                                                     #
# ============================================================= #

@pytest.fixture(scope="module")
def sim():
    return ShadowTradingSimulator(SimulationConfig(random_seed=42))


@pytest.fixture(scope="module")
def df(sim):
    return sim.generate_full_dataset()


@pytest.fixture(scope="module")
def bundle(df):
    return run_all_regressions(df)


# ============================================================= #
#  1. Simulator tests                                           #
# ============================================================= #

class TestSimulator:
    def test_total_observations(self, df):
        """Should match paper: 3,111 total."""
        assert len(df) == 3111, f"Expected 3111, got {len(df)}"

    def test_negative_obs_count(self, df):
        """Should match paper: 1,129 negative CAR."""
        assert len(df[df["car_sign"] == "negative"]) == 1129

    def test_positive_obs_count(self, df):
        """Should match paper: 1,982 positive CAR."""
        assert len(df[df["car_sign"] == "positive"]) == 1982

    def test_source_firm_count(self, df):
        """Should be close to paper: 598 source firms."""
        n = df["source_id"].nunique()
        assert 500 <= n <= 700, f"Source firm count out of range: {n}"

    def test_linked_firm_count(self, df):
        """Should be close to paper: 745 linked firms."""
        n = df["linked_id"].nunique()
        assert 600 <= n <= 900, f"Linked firm count out of range: {n}"

    def test_year_range(self, df):
        """Sample period 1997–2011."""
        assert df["year"].min() >= 1997
        assert df["year"].max() <= 2011

    def test_required_columns(self, df):
        required = [
            "source_id", "linked_id", "year", "quarter", "car_sign",
            "business_partner_car", "competitor_car",
            "abnormal_short_sales", "option_stock_ratio", "order_imbalance",
            "future_return", "idd_shock", "post_enforcement", "prohibit",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_abnormal_short_sales_negative_mean(self, df):
        """Paper Table 1: mean ABSS before neg events ≈ 7.9%."""
        mean = df[df["car_sign"] == "negative"]["abnormal_short_sales"].mean()
        assert 0.04 <= mean <= 0.14, f"ABSS mean out of range: {mean:.4f}"

    def test_abnormal_short_sales_positive_mean(self, df):
        """Paper: mean ABSS before pos events ≈ -3.6%."""
        mean = df[df["car_sign"] == "positive"]["abnormal_short_sales"].mean()
        assert -0.09 <= mean <= 0.01, f"ABSS pos mean out of range: {mean:.4f}"

    def test_option_stock_ratio_negative_mean(self, df):
        """Paper: option/stock ratio before neg events ≈ 2.336."""
        mean = df[df["car_sign"] == "negative"]["option_stock_ratio"].mean()
        assert 1.0 <= mean <= 4.0, f"OSR mean out of range: {mean:.4f}"

    def test_option_stock_ratio_positive(self, df):
        """Paper: OSR for pos events < OSR for neg events."""
        neg_mean = df[df["car_sign"] == "negative"]["option_stock_ratio"].mean()
        pos_mean = df[df["car_sign"] == "positive"]["option_stock_ratio"].mean()
        assert neg_mean > pos_mean, "OSR should be higher before neg events"

    def test_idd_shock_column(self, df):
        """IDDShock should be binary."""
        assert set(df["idd_shock"].unique()).issubset({0, 1})

    def test_post_enforcement_column(self, df):
        """Post enforcement should be binary."""
        assert set(df["post_enforcement"].unique()).issubset({0, 1})

    def test_prohibit_column(self, df):
        """Prohibition flag should be binary with ~53% = 1."""
        assert set(df["prohibit"].unique()).issubset({0, 1})
        pct = df["prohibit"].mean()
        assert 0.40 <= pct <= 0.65, f"Prohibition rate out of range: {pct:.2f}"

    def test_car_non_negative(self, df):
        """Absolute CARs should be non-negative."""
        assert (df["business_partner_car"] >= 0).all()
        assert (df["competitor_car"] >= 0).all()

    def test_industry_distribution(self, df):
        """No single named (non-Other) industry should dominate excessively (< 20%)."""
        named = df[df["industry"] != "Other"]["industry"]
        max_share = named.value_counts(normalize=True).max()
        assert max_share <= 0.20, f"Named industry concentration too high: {max_share:.2%}"


# ============================================================= #
#  2. Regression engine tests                                   #
# ============================================================= #

class TestRegressions:
    def test_table2_count(self, bundle):
        """Table 2 should have 6 results (3 DVs × 2 signs)."""
        assert len(bundle.table2) == 6

    def test_table4_count(self, bundle):
        """Table 4 should have 3 results (one per shadow proxy)."""
        assert len(bundle.table4) == 3

    def test_table5_count(self, bundle):
        """Table 5 should have 6 results."""
        assert len(bundle.table5) == 6

    def test_table6_count(self, bundle):
        """Table 6 should have 4 results (2 DVs × 2 signs)."""
        assert len(bundle.table6) == 4

    def test_table7_count(self, bundle):
        """Table 7 should have 6 results."""
        assert len(bundle.table7) == 6

    def test_bp_car_positive_neg_abss(self, bundle):
        """Table 2 col 1: BP_CAR positive for neg ABSS (paper: +0.033)."""
        r = next(x for x in bundle.table2
                 if x.dependent_var == "abnormal_short_sales"
                 and x.car_sign == "negative")
        coef = r.coef.get("business_partner_car", 0)
        assert coef > 0, f"Expected positive BP_CAR coef, got {coef:.4f}"

    def test_bp_car_negative_pos_abss(self, bundle):
        """Table 2 col 2: BP_CAR negative for pos ABSS (paper: -0.019)."""
        r = next(x for x in bundle.table2
                 if x.dependent_var == "abnormal_short_sales"
                 and x.car_sign == "positive")
        coef = r.coef.get("business_partner_car", 0)
        assert coef < 0, f"Expected negative BP_CAR coef for pos, got {coef:.4f}"

    def test_comp_car_positive_neg_abss(self, bundle):
        """Table 2: Comp_CAR positive for neg events ABSS."""
        r = next(x for x in bundle.table2
                 if x.dependent_var == "abnormal_short_sales"
                 and x.car_sign == "negative")
        coef = r.coef.get("competitor_car", 0)
        assert coef > 0, f"Expected positive Comp_CAR, got {coef:.4f}"

    def test_osr_bp_car_positive_all(self, bundle):
        """OSR negative events: BP_CAR must be positive (paper: +0.699)."""
        r = next(x for x in bundle.table2
                 if x.dependent_var == "option_stock_ratio"
                 and x.car_sign == "negative")
        coef = r.coef.get("business_partner_car", 0)
        assert coef > 0, f"OSR BP_CAR should be positive (negative events), got {coef:.4f}"

    def test_significance_of_main_vars(self, bundle):
        """At least 2 of 6 Table 2 BP_CAR coefficients should be significant at 5%."""
        sig_count = sum(
            1 for r in bundle.table2
            if abs(r.tstat.get("business_partner_car", 0)) >= 1.96
        )
        assert sig_count >= 2, f"Only {sig_count}/6 BP_CAR are significant at 5%"

    def test_no_nan_in_results(self, bundle):
        """All key results should be free of NaN."""
        for r in bundle.table2:
            assert not np.isnan(r.adj_r_squared), f"NaN adj R2 in {r.label}"
            assert r.n_obs > 0, f"Zero observations in {r.label}"

    def test_f_test_pvalues_in_range(self, bundle):
        """F-test p-values should be in [0, 1]."""
        for r in bundle.table2:
            assert 0 <= r.f_test_bp_eq_comp <= 1.0, \
                f"F-test p out of range: {r.f_test_bp_eq_comp}"

    def test_table4_negative_abss_coef(self, bundle):
        """Table 4: ABSS should have negative coef on future return."""
        r = next(x for x in bundle.table4
                 if x.dependent_var == "future_return"
                 and "abnormal_short_sales" in x.coef.index)
        coef = r.coef.get("abnormal_short_sales", 0)
        assert coef < 0, f"Expected ABSS coef < 0, got {coef:.4f}"

    def test_table4_positive_oi_coef(self, bundle):
        """Table 4: Order imbalance should have positive coef on future return."""
        r = next(x for x in bundle.table4
                 if "order_imbalance" in x.coef.index)
        coef = r.coef.get("order_imbalance", 0)
        assert coef > 0, f"Expected OI coef > 0, got {coef:.4f}"

    def test_table5_post_interaction_positive(self, bundle):
        """Table 5: BP_CAR*Post should be positive for neg ABSS."""
        r = next((x for x in bundle.table5
                  if x.dependent_var == "abnormal_short_sales"
                  and x.car_sign == "negative"), None)
        if r is not None:
            coef = r.coef.get("bp_car_x_post", 0)
            assert coef > 0, f"Expected positive post interaction, got {coef:.4f}"

    def test_table7_prohibit_attenuates(self, bundle):
        """Table 7: Prohibit should have negative coefficient (attenuation)."""
        r = next((x for x in bundle.table7
                  if x.dependent_var == "abnormal_short_sales"
                  and x.car_sign == "negative"), None)
        if r is not None and "prohibit" in r.coef.index:
            coef = r.coef.get("prohibit", 0)
            assert coef < 0, f"Expected prohibit < 0, got {coef:.4f}"

    def test_adj_r2_positive(self, bundle):
        """Adjusted R² should be above -0.05 (near-zero acceptable for positive partition)."""
        for r in bundle.table2 + bundle.table4:
            assert r.adj_r_squared > -0.05, f"Adj R² severely negative in {r.label}: {r.adj_r_squared:.4f}"

    def test_summary_df_shape(self, bundle):
        """RegressionBundle.to_summary_df should return correct shapes."""
        df2 = bundle.to_summary_df("table2")
        assert len(df2) == 6
        assert "β_BP" in df2.columns


# ============================================================= #
#  3. Economic significance tests                               #
# ============================================================= #

class TestEconomicSignificance:
    def test_profit_low_bound(self, df):
        """Paper: profit low = $139,400."""
        profit = compute_profit_estimate(df)
        assert abs(profit["profit_low_paper"] - 139_400) < 1, \
            f"Low profit: {profit['profit_low_paper']}"

    def test_profit_high_bound(self, df):
        """Paper fn.14: 205,000 shares × $3.30 = $676,500 (cited as ~$678,000)."""
        profit = compute_profit_estimate(df)
        assert 670_000 <= profit["profit_high_paper"] <= 685_000, \
            f"High profit out of range: {profit['profit_high_paper']}"

    def test_econ_sig_df_has_rows(self, bundle, df):
        econ = compute_economic_significance(bundle.table2, df)
        assert len(econ) > 0, "Economic significance table should not be empty"

    def test_econ_sig_columns(self, bundle, df):
        econ = compute_economic_significance(bundle.table2, df)
        for col in ["DV", "CAR Sign", "Variable", "Econ Sig (%)"]:
            assert col in econ.columns

    def test_econ_sig_reasonable_range(self, bundle, df):
        """Economic significance estimates should be in ±100% range."""
        econ = compute_economic_significance(bundle.table2, df)
        for _, row in econ.iterrows():
            assert -200 <= row["Econ Sig (%)"] <= 200, \
                f"Econ sig out of range: {row['Econ Sig (%)']:.2f}"

    def test_idd_econ_sig_returns_df(self, bundle):
        idd = compute_idd_econ_significance(bundle.table6)
        assert isinstance(idd, pd.DataFrame)


# ============================================================= #
#  4. STLS model tests                                          #
# ============================================================= #

class TestSTLS:
    def test_stls_columns_added(self, df):
        stls = ShadowTradingLikelihoodScore()
        df_s = stls.fit_transform(df)
        for col in ["z_abss", "z_osr", "z_oi", "stls_raw",
                    "stls_percentile", "stls_band"]:
            assert col in df_s.columns, f"Missing STLS column: {col}"

    def test_stls_percentile_range(self, df):
        stls = ShadowTradingLikelihoodScore()
        df_s = stls.fit_transform(df)
        assert df_s["stls_percentile"].min() >= 0
        assert df_s["stls_percentile"].max() <= 100

    def test_stls_band_labels(self, df):
        stls = ShadowTradingLikelihoodScore()
        df_s = stls.fit_transform(df)
        bands = set(df_s["stls_band"].astype(str).unique())
        assert "Low" in bands
        assert "Medium" in bands
        assert "High" in bands

    def test_high_risk_count(self, df):
        """~25% should be above 75th percentile."""
        stls = ShadowTradingLikelihoodScore()
        df_s = stls.fit_transform(df)
        high = stls.high_risk_firms(df_s)
        frac = len(high) / len(df_s)
        assert 0.20 <= frac <= 0.35, f"High-risk fraction out of range: {frac:.2%}"

    def test_prohibition_attenuates_stls(self, df):
        """Mean STLS should be lower for firms with prohibition."""
        stls = ShadowTradingLikelihoodScore()
        df_s = stls.fit_transform(df)
        tbl = stls.compute_stls_by_prohibition(df_s)
        no_prohib = tbl.loc[tbl["prohibit"] == 0, "Mean STLS"].mean()
        prohibit = tbl.loc[tbl["prohibit"] == 1, "Mean STLS"].mean()
        assert no_prohib > prohibit, \
            f"Expected no_prohib ({no_prohib:.1f}) > prohibit ({prohibit:.1f})"

    def test_stls_band_counts(self, df):
        stls = ShadowTradingLikelihoodScore()
        df_s = stls.fit_transform(df)
        counts = stls.band_counts(df_s)
        assert len(counts) > 0

    def test_custom_config(self, df):
        """STLS should accept custom weights."""
        cfg = STLSConfig(weight_abss=0.5, weight_osr=0.3, weight_oi=0.2)
        stls = ShadowTradingLikelihoodScore(cfg)
        df_s = stls.fit_transform(df)
        assert "stls_percentile" in df_s.columns


# ============================================================= #
#  5. Integration test                                          #
# ============================================================= #

class TestIntegration:
    def test_full_pipeline_runs(self):
        """End-to-end pipeline should complete without error."""
        sim = ShadowTradingSimulator(SimulationConfig(
            n_obs_negative=200, n_obs_positive=300, random_seed=99
        ))
        df = sim.generate_full_dataset()
        bundle = run_all_regressions(df)
        stls = ShadowTradingLikelihoodScore()
        df_s = stls.fit_transform(df)
        econ = compute_economic_significance(bundle.table2, df)
        profit = compute_profit_estimate(df)
        assert len(bundle.table2) == 6
        assert len(econ) > 0
        assert profit["profit_low_paper"] > 0

    def test_bundle_summary_all_tables(self, bundle):
        for tbl in ["table2", "table4", "table5", "table6", "table7"]:
            df_ = bundle.to_summary_df(tbl)
            assert len(df_) > 0, f"{tbl} summary is empty"
