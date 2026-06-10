"""
Shadow Trading Likelihood Score (STLS)
=======================================
A composite measure that aggregates the three informed-trading proxies from
the paper into a single score — analogous to an Altman Z-Score or
Manipulation Likelihood Score (Ben-David et al. 2011) but calibrated to
shadow trading.

Scoring methodology:
  1. Standardise each proxy: z = (x - μ) / σ  (rolling 252-day window)
  2. Combine using paper coefficient magnitudes as weights:
       STLS = w_abss * z_abss + w_osr * z_osr + w_oi * z_oi
  3. Scale to 0-100 percentile rank within each year
  4. Classify into risk bands: Low (<25), Medium (25-75), High (>75)

The composite captures:
  - Abnormal short sales (negative-event signal: high = more shadow trading)
  - Option/Stock ratio (high = more information asymmetry)
  - Order imbalance (absolute value: large swing = informed trading)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class STLSConfig:
    # Weights derived from paper coefficient magnitudes (normalised)
    # From Table 2 negative CAR: β_BP_ABSS=0.033, β_BP_OSR=0.699, β_BP_OI=0.011
    # Normalise by range of each DV
    weight_abss: float = 0.25
    weight_osr: float = 0.50
    weight_oi: float = 0.25
    rolling_window: int = 252  # 1 year of trading days
    percentile_within_year: bool = True


class ShadowTradingLikelihoodScore:
    """
    Computes the Shadow Trading Likelihood Score (STLS) for each
    observation in a panel dataset.
    """

    def __init__(self, config: Optional[STLSConfig] = None):
        self.cfg = config or STLSConfig()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute STLS and add it to the dataframe.

        Returns df with additional columns:
          z_abss, z_osr, z_oi,
          stls_raw, stls_percentile, stls_band
        """
        df = df.copy()
        cfg = self.cfg

        # ---- 1. Cross-sectional standardise within year --------------- #
        for col, z_col in [
            ("abnormal_short_sales", "z_abss"),
            ("option_stock_ratio", "z_osr"),
            ("order_imbalance", "z_oi"),
        ]:
            if col not in df.columns:
                df[z_col] = 0.0
                continue
            df[z_col] = df.groupby("year")[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )

        # For positive-event short sales, flip sign so high z = more activity
        neg_mask = df["car_sign"] == "negative"
        df.loc[~neg_mask, "z_abss"] = -df.loc[~neg_mask, "z_abss"]
        # Order imbalance: for negative events, negative OI = more selling = higher risk
        df.loc[neg_mask, "z_oi"] = -df.loc[neg_mask, "z_oi"]

        # ---- 2. Composite score ---------------------------------------- #
        df["stls_raw"] = (
            cfg.weight_abss * df["z_abss"]
            + cfg.weight_osr * df["z_osr"]
            + cfg.weight_oi * df["z_oi"]
        )

        # ---- 3. Percentile within year --------------------------------- #
        if cfg.percentile_within_year:
            df["stls_percentile"] = df.groupby("year")["stls_raw"].transform(
                lambda x: stats.rankdata(x) / len(x) * 100
            )
        else:
            pct = stats.rankdata(df["stls_raw"]) / len(df) * 100
            df["stls_percentile"] = pct

        # ---- 4. Risk band ---------------------------------------------- #
        df["stls_band"] = pd.cut(
            df["stls_percentile"],
            bins=[-np.inf, 25, 75, np.inf],
            labels=["Low", "Medium", "High"],
        )

        return df

    @staticmethod
    def score_distribution(df: pd.DataFrame) -> pd.DataFrame:
        """Summary statistics of STLS by year and CAR sign."""
        if "stls_percentile" not in df.columns:
            raise ValueError("Run fit_transform first.")
        grp = df.groupby(["year", "car_sign"])["stls_percentile"].agg(
            ["mean", "median", "std", "count"]
        ).reset_index()
        grp.columns = ["Year", "CAR Sign", "Mean STLS", "Median STLS",
                       "Std STLS", "N"]
        return grp

    @staticmethod
    def band_counts(df: pd.DataFrame) -> pd.DataFrame:
        """Count observations per risk band."""
        if "stls_band" not in df.columns:
            raise ValueError("Run fit_transform first.")
        return (df.groupby(["stls_band", "car_sign"])
                .size()
                .reset_index(name="count"))

    @staticmethod
    def high_risk_firms(
        df: pd.DataFrame, percentile_threshold: float = 75.0
    ) -> pd.DataFrame:
        """Return observations above the percentile threshold."""
        if "stls_percentile" not in df.columns:
            raise ValueError("Run fit_transform first.")
        return df[df["stls_percentile"] >= percentile_threshold].copy()

    def compute_stls_by_prohibition(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mean STLS grouped by prohibition status — cross-checks Table 7
        finding that prohibition attenuates shadow trading.
        """
        if "stls_percentile" not in df.columns:
            df = self.fit_transform(df)
        return (df.groupby(["prohibit", "car_sign"])["stls_percentile"]
                .agg(["mean", "median", "count"])
                .reset_index()
                .rename(columns={"mean": "Mean STLS", "median": "Median STLS",
                                 "count": "N"}))
