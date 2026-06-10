"""
Shadow Trading Data Simulator
==============================
Generates synthetic data that mirrors the Mehta, Reeb & Zhao (2020) sample
structure and descriptive statistics (Table 1). This is used when real
CRSP/Compustat/Ancerno data are not available, enabling full pipeline testing.

Key parameters are calibrated from the paper:
  - Business partner negative CAR: mean=-3.5%, std=25.6%
  - Business partner positive CAR: mean=+2.1%, std=21.9%
  - Linked firm abnormal short sales (neg event): mean=7.9%, std=27.9%
  - Linked firm abnormal short sales (pos event): mean=-3.6%, std=33.2%
  - Option/Stock ratio (neg): mean=2.336, std=3.221
  - Order imbalance (neg): mean=-0.032, std=0.072
  - Sample: 1997-2011, 3,111 obs (1,129 neg + 1,982 pos)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationConfig:
    n_source_firms: int = 598
    n_linked_firms: int = 745
    n_obs_negative: int = 1129
    n_obs_positive: int = 1982
    start_year: int = 1997
    end_year: int = 2011
    random_seed: int = 42
    # Shadow trading signal strength (beta coefficients from Table 2)
    beta_bp_car_abss: float = 0.033   # Business Partner CAR -> Abnormal Short Sales
    beta_comp_car_abss: float = 0.031  # Competitor CAR -> Abnormal Short Sales
    beta_bp_car_osr: float = 0.699     # Business Partner CAR -> Option/Stock Ratio
    beta_comp_car_osr: float = 0.611   # Competitor CAR -> Option/Stock Ratio
    beta_bp_car_oi: float = 0.011      # Business Partner CAR -> Order Imbalance
    beta_comp_car_oi: float = 0.009    # Competitor CAR -> Order Imbalance


class ShadowTradingSimulator:
    """
    Generates a synthetic shadow trading dataset calibrated to the paper's
    Table 1 descriptive statistics and Table 2 regression coefficients.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.cfg = config or SimulationConfig()
        self.rng = np.random.default_rng(self.cfg.random_seed)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate_full_dataset(self) -> pd.DataFrame:
        """Return the combined negative + positive CAR dataset."""
        neg = self._generate_partition(sign="negative", n=self.cfg.n_obs_negative)
        pos = self._generate_partition(sign="positive", n=self.cfg.n_obs_positive)
        df = pd.concat([neg, pos], ignore_index=True)
        df = self._add_idd_shock(df)
        df = self._add_enforcement_spike(df)
        df = self._add_prohibition_flag(df)
        return df

    def generate_negative_partition(self) -> pd.DataFrame:
        return self._generate_partition("negative", self.cfg.n_obs_negative)

    def generate_positive_partition(self) -> pd.DataFrame:
        return self._generate_partition("positive", self.cfg.n_obs_positive)

    # ------------------------------------------------------------------ #
    #  Internal generators                                                 #
    # ------------------------------------------------------------------ #

    def _generate_partition(self, sign: str, n: int) -> pd.DataFrame:
        """
        Build one CAR-sign partition.
        sign : 'negative' | 'positive'
        """
        cfg = self.cfg
        is_neg = sign == "negative"
        sign_scalar = -1 if is_neg else 1

        # ----- Source firm IDs and time --------------------------------- #
        source_ids = self.rng.integers(1, cfg.n_source_firms + 1, size=n)
        linked_ids = self.rng.integers(1, cfg.n_linked_firms + 1, size=n)
        years = self.rng.integers(cfg.start_year, cfg.end_year + 1, size=n)
        quarters = self.rng.integers(1, 5, size=n)

        # ----- Relationship type (business partner vs competitor) -------- #
        is_bp = self.rng.random(n) < 0.5  # ~50% business partner

        # ----- Source firm CARs ----------------------------------------- #
        # Paper Table 1: BP neg mean=-3.5% std=25.6%, pos mean=2.1% std=21.9%
        if is_neg:
            bp_car_raw = self.rng.normal(-0.035, 0.256, n)
            comp_car_raw = self.rng.normal(-0.033, 0.229, n)
        else:
            bp_car_raw = self.rng.normal(0.021, 0.219, n)
            comp_car_raw = self.rng.normal(0.023, 0.228, n)

        business_partner_car = np.where(is_bp, np.abs(bp_car_raw), 0.0)
        competitor_car = np.where(~is_bp, np.abs(comp_car_raw), 0.0)
        # Signed versions for directional analysis
        business_partner_car_signed = np.where(is_bp, bp_car_raw, 0.0)
        competitor_car_signed = np.where(~is_bp, comp_car_raw, 0.0)

        # ----- Controls ------------------------------------------------- #
        firm_size = self.rng.normal(np.log(4434), 1.2, n)
        book_to_market = self.rng.normal(0.782, 0.509, n).clip(0.05, 5)
        frev = self.rng.normal(-0.012, 0.095, n)
        total_accruals = self.rng.normal(-0.023, 0.070, n)
        ep = self.rng.normal(-0.007, 0.076, n)
        turnover = self.rng.normal(0.522, 0.273, n).clip(0.01, 1)
        sales_growth = self.rng.normal(0.017, 0.056, n)
        ltg = self.rng.normal(15.552, 9.156, n)
        momentum = self.rng.normal(0.085, 0.267, n)
        misp = self.rng.normal(53.2, 11.0, n)
        past_return = self.rng.normal(0.022, 0.617, n)

        # Source firm controls
        firm_size_source = self.rng.normal(np.log(6137), 1.3, n)
        book_to_market_source = self.rng.normal(0.818, 0.517, n).clip(0.05, 5)
        frev_source = self.rng.normal(-0.018, 0.102, n)
        tacc_source = self.rng.normal(-0.027, 0.072, n)
        ep_source = self.rng.normal(-0.012, 0.080, n)
        turnover_source = self.rng.normal(0.517, 0.267, n).clip(0.01, 1)
        sales_growth_source = self.rng.normal(0.024, 0.086, n)
        ltg_source = self.rng.normal(13.490, 9.912, n)
        momentum_source = self.rng.normal(0.090, 0.233, n)
        misp_source = self.rng.normal(54.566, 11.655, n)
        past_return_source = self.rng.normal(0.027, 0.592, n)

        # ----- Dependent Variables (signal + noise) --------------------- #
        noise_abss = self.rng.normal(0, 0.22, n)
        noise_osr = self.rng.normal(0, 2.5, n)
        noise_oi = self.rng.normal(0, 0.065, n)

        if is_neg:
            # Abnormal short sales: positive relation with abs(CAR)
            abss_signal = (cfg.beta_bp_car_abss * business_partner_car
                           + cfg.beta_comp_car_abss * competitor_car
                           + 0.003 * total_accruals
                           + 0.079)  # intercept matching Table 1
            # Option/stock: positive relation regardless of sign
            osr_signal = (cfg.beta_bp_car_osr * business_partner_car
                          + cfg.beta_comp_car_osr * competitor_car
                          + 2.336)
            # Order imbalance: negative (more selling) before neg events
            oi_signal = (-cfg.beta_bp_car_oi * business_partner_car
                         - cfg.beta_comp_car_oi * competitor_car
                         - 0.032)
        else:
            # Abnormal short sales: negative relation before pos news
            abss_signal = (-cfg.beta_bp_car_abss * business_partner_car
                           - cfg.beta_comp_car_abss * competitor_car
                           - 0.036)
            osr_signal = (cfg.beta_bp_car_osr * business_partner_car
                          + cfg.beta_comp_car_osr * competitor_car
                          + 1.496)
            oi_signal = (cfg.beta_bp_car_oi * business_partner_car
                         + cfg.beta_comp_car_oi * competitor_car
                         + 0.025)

        abnormal_short_sales = abss_signal + noise_abss
        option_stock_ratio = np.abs(osr_signal + noise_osr)
        order_imbalance = oi_signal + noise_oi

        # ----- Future returns (Table 4: shadow trading predicts returns) - #
        # ABSS negatively predicts future returns; OI positively
        future_return = (-0.266 * abnormal_short_sales * sign_scalar
                         + 0.191 * order_imbalance * sign_scalar
                         + self.rng.normal(0, 0.05, n))

        # ----- Fama-French industry (top 10 from Table 1 Panel C) ------- #
        industries = [
            "Pharmaceutical Products", "Business Services",
            "Petroleum and Natural Gas", "Chemicals", "Communication",
            "Retail", "Machinery", "Healthcare",
            "Electrical Equipment", "Construction Materials"
        ]
        industry_weights = [0.1002, 0.0971, 0.0727, 0.0525, 0.0434,
                            0.0422, 0.0373, 0.0360, 0.0354, 0.0348]
        # pad remaining weight to "Other"
        industries.append("Other")
        industry_weights.append(1 - sum(industry_weights))
        industry_labels = self.rng.choice(industries, size=n, p=industry_weights)

        df = pd.DataFrame({
            "source_id": source_ids,
            "linked_id": linked_ids,
            "year": years,
            "quarter": quarters,
            "car_sign": sign,
            "is_business_partner": is_bp,
            "business_partner_car": business_partner_car,
            "competitor_car": competitor_car,
            "business_partner_car_signed": business_partner_car_signed,
            "competitor_car_signed": competitor_car_signed,
            # Dependent variables
            "abnormal_short_sales": abnormal_short_sales,
            "option_stock_ratio": option_stock_ratio,
            "order_imbalance": order_imbalance,
            "future_return": future_return,
            # Linked firm controls
            "firm_size": firm_size,
            "book_to_market": book_to_market,
            "frev": frev,
            "total_accruals": total_accruals,
            "ep": ep,
            "turnover": turnover,
            "sales_growth": sales_growth,
            "ltg": ltg,
            "momentum": momentum,
            "misp": misp,
            "past_return": past_return,
            # Source firm controls
            "firm_size_source": firm_size_source,
            "book_to_market_source": book_to_market_source,
            "frev_source": frev_source,
            "tacc_source": tacc_source,
            "ep_source": ep_source,
            "turnover_source": turnover_source,
            "sales_growth_source": sales_growth_source,
            "ltg_source": ltg_source,
            "momentum_source": momentum_source,
            "misp_source": misp_source,
            "past_return_source": past_return_source,
            "industry": industry_labels,
        })
        return df

    def _add_idd_shock(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mark ~508/1016 treatment firms per the IDD identification (Section V.2).
        States: MO/OH adopt 2000, FL rejects 2001, MI rejects 2002,
                TX rejects 2003, KS adopts 2006.
        """
        rng = self.rng
        n = len(df)
        # 33% of source firms are in IDD-shock states
        is_treated = rng.random(n) < 0.33
        shock_year = np.where(is_treated,
                              rng.choice([2000, 2001, 2002, 2003, 2006],
                                         size=n),
                              9999)
        idd_shock = np.where(is_treated & (df["year"].values >= shock_year), 1, 0)
        is_adoption = rng.random(n) < 0.5   # adoption vs rejection
        # For adoption: IDDShock=1 post-shock, rejection: IDDShock=0 post-shock
        idd_shock = np.where(is_treated & ~is_adoption,
                             1 - idd_shock, idd_shock)
        df["idd_shock"] = idd_shock.astype(int)
        df["is_idd_treated"] = is_treated.astype(int)
        return df

    def _add_enforcement_spike(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mark the three SEC enforcement spike windows (Table 5):
        June 2003, June 2006, October 2009.
        Post = 1 for 3-month window AFTER spike month.
        """
        spike_months = [
            (2003, 2),  # Q2 2003
            (2006, 2),  # Q2 2006
            (2009, 4),  # Q4 2009
        ]
        post = np.zeros(len(df), dtype=int)
        for spike_year, spike_q in spike_months:
            in_post = ((df["year"] == spike_year) & (df["quarter"] == spike_q + 1)) | \
                      ((df["year"] == spike_year) & (df["quarter"] == spike_q) &
                       (df.index % 3 == 0))
            post = np.where(in_post, 1, post)
        df["post_enforcement"] = post
        return df

    def _add_prohibition_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ~53% of source firms prohibit shadow trading (Section VI).
        Source firms with prohibition have attenuated shadow trading.
        """
        rng = self.rng
        unique_sources = df["source_id"].unique()
        prohibit_map = {
            sid: int(rng.random() < 0.53) for sid in unique_sources
        }
        df["prohibit"] = df["source_id"].map(prohibit_map)
        # Attenuate shadow trading for prohibiting firms (matches Table 7)
        mask = df["prohibit"] == 1
        df.loc[mask, "abnormal_short_sales"] *= 0.72   # 8-12% attenuation
        df.loc[mask, "option_stock_ratio"] *= 0.75
        df.loc[mask, "order_imbalance"] *= 0.70
        return df
