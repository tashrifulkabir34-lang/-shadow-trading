"""
Shadow Trading Regression Engine
==================================
Replicates the core OLS/panel specifications from Mehta, Reeb & Zhao (2020):

  Table 2: Equation (1) — main shadow trading regressions (3 DVs × 2 signs)
  Table 4: Future return predictability regressions
  Table 5: Equation (2) — post-enforcement spike interactions
  Table 6: Equation (3) — IDD shock difference-in-differences
  Table 7: Prohibition interaction regressions

All regressions use:
  - Standard errors clustered by firm AND year
  - Year + Fama-French 48 industry fixed effects
  - Panel OLS via linearmodels / statsmodels

Results are returned as RegressionResult dataclass objects, and the
module exposes a top-level runner that builds the full paper-replication
result table (RegressionBundle).
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

warnings.filterwarnings("ignore")


# ============================================================= #
#  Data structures                                              #
# ============================================================= #

@dataclass
class RegressionResult:
    """Holds one fitted regression with summary statistics."""
    label: str
    dependent_var: str
    car_sign: str
    coef: pd.Series
    se: pd.Series
    tstat: pd.Series
    pvalue: pd.Series
    r_squared: float
    adj_r_squared: float
    n_obs: int
    f_test_bp_eq_comp: float  # p-value of H0: β_BP = β_Comp
    economic_significance: Dict[str, float] = field(default_factory=dict)

    # Paper coefficient targets (Table 2)
    PAPER_TARGETS = {
        "abnormal_short_sales": {
            "negative": {"business_partner_car": 0.033, "competitor_car": 0.031},
            "positive": {"business_partner_car": -0.019, "competitor_car": -0.020},
        },
        "option_stock_ratio": {
            "negative": {"business_partner_car": 0.699, "competitor_car": 0.611},
            "positive": {"business_partner_car": 0.621, "competitor_car": 0.581},
        },
        "order_imbalance": {
            "negative": {"business_partner_car": 0.011, "competitor_car": 0.009},
            "positive": {"business_partner_car": 0.008, "competitor_car": 0.008},
        },
    }

    def paper_match(self) -> Dict[str, float]:
        """Return ratio of estimated / paper coef for key variables."""
        targets = self.PAPER_TARGETS.get(self.dependent_var, {}).get(self.car_sign, {})
        ratios = {}
        for var, paper_val in targets.items():
            if var in self.coef.index and paper_val != 0:
                ratios[var] = self.coef[var] / paper_val
        return ratios

    def summary_row(self) -> dict:
        bp = "business_partner_car"
        comp = "competitor_car"
        return {
            "Label": self.label,
            "DV": self.dependent_var,
            "Sign": self.car_sign,
            "β_BP": round(self.coef.get(bp, np.nan), 4),
            "t_BP": round(self.tstat.get(bp, np.nan), 2),
            "β_Comp": round(self.coef.get(comp, np.nan), 4),
            "t_Comp": round(self.tstat.get(comp, np.nan), 2),
            "N": self.n_obs,
            "Adj_R2": round(self.adj_r_squared, 3),
            "F-test p": round(self.f_test_bp_eq_comp, 3),
        }


@dataclass
class RegressionBundle:
    """Container for all replicated regression tables."""
    table2: List[RegressionResult] = field(default_factory=list)
    table4: List[RegressionResult] = field(default_factory=list)
    table5: List[RegressionResult] = field(default_factory=list)
    table6: List[RegressionResult] = field(default_factory=list)
    table7: List[RegressionResult] = field(default_factory=list)

    def to_summary_df(self, table: str = "table2") -> pd.DataFrame:
        results = getattr(self, table, [])
        return pd.DataFrame([r.summary_row() for r in results])


# ============================================================= #
#  Core OLS with two-way cluster-robust SE                      #
# ============================================================= #

CONTROL_VARS = [
    "firm_size", "past_return", "frev", "book_to_market", "total_accruals",
    "ep", "turnover", "sales_growth", "ltg", "momentum", "misp",
    "firm_size_source", "past_return_source", "frev_source",
    "book_to_market_source", "tacc_source", "ep_source", "turnover_source",
    "sales_growth_source", "ltg_source", "momentum_source", "misp_source",
]


def _add_fixed_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Add year and industry dummy columns."""
    df = df.copy()
    year_dummies = pd.get_dummies(df["year"], prefix="yr", drop_first=True)
    ind_dummies = pd.get_dummies(df["industry"], prefix="ind", drop_first=True)
    return pd.concat([df, year_dummies, ind_dummies], axis=1)


def _cluster_se_ols(
    y: np.ndarray,
    X: np.ndarray,
    cluster1: np.ndarray,
    cluster2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-way cluster-robust standard errors (Cameron, Gelbach & Miller 2011).
    Returns (coef, se).
    """
    n, k = X.shape
    XTX_inv = np.linalg.pinv(X.T @ X)
    coef = XTX_inv @ X.T @ y
    resid = y - X @ coef

    def _meat(cluster: np.ndarray) -> np.ndarray:
        B = np.zeros((k, k))
        for c in np.unique(cluster):
            idx = cluster == c
            Xc = X[idx]
            ec = resid[idx]
            score = Xc.T @ ec
            B += np.outer(score, score)
        return B

    B1 = _meat(cluster1)
    B2 = _meat(cluster2)
    # Combine cluster labels for intersection
    intersection = np.array([f"{a}_{b}" for a, b in zip(cluster1, cluster2)])
    B12 = _meat(intersection)

    V = XTX_inv @ (B1 + B2 - B12) @ XTX_inv
    se = np.sqrt(np.abs(np.diag(V)))
    return coef, se


def _run_ols(
    df: pd.DataFrame,
    dep_var: str,
    indep_vars: List[str],
    label: str,
    car_sign: str,
    cluster_firm: str = "linked_id",
    cluster_time: str = "year",
) -> RegressionResult:
    """Fit OLS with two-way clustered SE and FE dummies."""
    df_fe = _add_fixed_effects(df)
    fe_cols = [c for c in df_fe.columns if c.startswith("yr_") or c.startswith("ind_")]
    all_x_cols = indep_vars + [c for c in fe_cols if c in df_fe.columns]

    # Drop rows with any NaN
    cols_needed = [dep_var] + all_x_cols + [cluster_firm, cluster_time]
    df_clean = df_fe[cols_needed].dropna()

    y = df_clean[dep_var].values.astype(float)
    X_raw = df_clean[all_x_cols].values.astype(float)
    # Add intercept
    X = np.column_stack([np.ones(len(y)), X_raw])
    col_names = ["const"] + all_x_cols

    cluster1 = df_clean[cluster_firm].values
    cluster2 = df_clean[cluster_time].values

    coef, se = _cluster_se_ols(y, X, cluster1, cluster2)
    tstat = coef / (se + 1e-12)
    pvalue = 2 * (1 - stats.norm.cdf(np.abs(tstat)))

    # R-squared
    ss_res = np.sum((y - X @ coef) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    n, k = len(y), X.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k else 0

    coef_s = pd.Series(coef, index=col_names)
    se_s = pd.Series(se, index=col_names)
    tstat_s = pd.Series(tstat, index=col_names)
    pval_s = pd.Series(pvalue, index=col_names)

    # F-test: β_BP = β_Comp
    bp_idx = col_names.index("business_partner_car") if "business_partner_car" in col_names else None
    comp_idx = col_names.index("competitor_car") if "competitor_car" in col_names else None
    f_pval = 1.0
    if bp_idx is not None and comp_idx is not None:
        diff = coef[bp_idx] - coef[comp_idx]
        var_diff = (se[bp_idx] ** 2 + se[comp_idx] ** 2
                    - 2 * 0.5 * se[bp_idx] * se[comp_idx])  # approx
        f_stat = (diff ** 2) / (var_diff + 1e-12)
        f_pval = 1 - stats.chi2.cdf(f_stat, df=1)

    dep_var_clean = dep_var.replace("_", " ")
    return RegressionResult(
        label=label,
        dependent_var=dep_var,
        car_sign=car_sign,
        coef=coef_s,
        se=se_s,
        tstat=tstat_s,
        pvalue=pval_s,
        r_squared=r2,
        adj_r_squared=adj_r2,
        n_obs=n,
        f_test_bp_eq_comp=f_pval,
    )


# ============================================================= #
#  Table 2 — Main shadow trading regressions                    #
# ============================================================= #

def run_table2(df: pd.DataFrame) -> List[RegressionResult]:
    """
    Equation (1): ShadowTrading = β1*BP_CAR + β2*Comp_CAR + controls + FE
    Six specifications: 3 DVs × 2 CAR signs.
    """
    dvs = ["abnormal_short_sales", "option_stock_ratio", "order_imbalance"]
    signs = ["negative", "positive"]
    x_vars = ["business_partner_car", "competitor_car"] + CONTROL_VARS
    results = []
    for dv in dvs:
        for sign in signs:
            sub = df[df["car_sign"] == sign].copy()
            label = f"Table2 | {dv} | {sign} CAR"
            r = _run_ols(sub, dv, x_vars, label, sign)
            results.append(r)
    return results


# ============================================================= #
#  Table 4 — Future return predictability                       #
# ============================================================= #

def run_table4(df: pd.DataFrame) -> List[RegressionResult]:
    """
    Future Return ~ ShadowTrading proxy + linked firm controls + FE
    Three specifications: one per proxy.
    """
    shadow_proxies = [
        "abnormal_short_sales", "option_stock_ratio", "order_imbalance"
    ]
    ctrl = ["firm_size", "past_return", "frev", "total_accruals",
            "book_to_market", "ep", "turnover", "sales_growth",
            "ltg", "momentum", "misp"]
    results = []
    for proxy in shadow_proxies:
        x_vars = [proxy] + ctrl
        label = f"Table4 | future_return ~ {proxy}"
        r = _run_ols(df, "future_return", x_vars, label, "pooled")
        results.append(r)
    return results


# ============================================================= #
#  Table 5 — Post-enforcement spike interactions                #
# ============================================================= #

def run_table5(df: pd.DataFrame) -> List[RegressionResult]:
    """
    Equation (2): adds Post, BP_CAR*Post, Comp_CAR*Post interactions.
    Tests whether shadow trading intensifies after high-profile SEC actions.
    """
    df = df.copy()
    df["bp_car_x_post"] = df["business_partner_car"] * df["post_enforcement"]
    df["comp_car_x_post"] = df["competitor_car"] * df["post_enforcement"]
    dvs = ["abnormal_short_sales", "option_stock_ratio", "order_imbalance"]
    signs = ["negative", "positive"]
    x_vars = (["business_partner_car", "competitor_car", "post_enforcement",
                "bp_car_x_post", "comp_car_x_post"] + CONTROL_VARS)
    results = []
    for dv in dvs:
        for sign in signs:
            sub = df[df["car_sign"] == sign].copy()
            label = f"Table5 | {dv} | {sign}"
            r = _run_ols(sub, dv, x_vars, label, sign)
            results.append(r)
    return results


# ============================================================= #
#  Table 6 — IDD shock D-i-D                                   #
# ============================================================= #

def run_table6(df: pd.DataFrame) -> List[RegressionResult]:
    """
    Equation (3): adds IDDShock, BP_CAR*IDDShock, Comp_CAR*IDDShock.
    Only Option/Stock Ratio and Order Imbalance (no short sale data
    over IDD window per paper footnote).
    """
    df = df.copy()
    df["bp_car_x_idd"] = df["business_partner_car"] * df["idd_shock"]
    df["comp_car_x_idd"] = df["competitor_car"] * df["idd_shock"]
    dvs = ["option_stock_ratio", "order_imbalance"]
    signs = ["negative", "positive"]
    x_vars = (["business_partner_car", "competitor_car", "idd_shock",
                "bp_car_x_idd", "comp_car_x_idd"] + CONTROL_VARS)
    results = []
    for dv in dvs:
        for sign in signs:
            sub = df[df["car_sign"] == sign].copy()
            label = f"Table6 | {dv} | {sign}"
            r = _run_ols(sub, dv, x_vars, label, sign)
            results.append(r)
    return results


# ============================================================= #
#  Table 7 — Prohibition interactions                           #
# ============================================================= #

def run_table7(df: pd.DataFrame) -> List[RegressionResult]:
    """
    Adds Prohibit, BP_CAR*Prohibit, Comp_CAR*Prohibit.
    Tests whether corporate policy attenuates shadow trading.
    """
    df = df.copy()
    df["bp_car_x_prohibit"] = df["business_partner_car"] * df["prohibit"]
    df["comp_car_x_prohibit"] = df["competitor_car"] * df["prohibit"]
    dvs = ["abnormal_short_sales", "option_stock_ratio", "order_imbalance"]
    signs = ["negative", "positive"]
    x_vars = (["prohibit", "business_partner_car", "bp_car_x_prohibit",
                "competitor_car", "comp_car_x_prohibit"] + CONTROL_VARS)
    results = []
    for dv in dvs:
        for sign in signs:
            sub = df[df["car_sign"] == sign].copy()
            label = f"Table7 | {dv} | {sign}"
            r = _run_ols(sub, dv, x_vars, label, sign)
            results.append(r)
    return results


# ============================================================= #
#  Main runner                                                  #
# ============================================================= #

def run_all_regressions(df: pd.DataFrame) -> RegressionBundle:
    """Run all five regression tables and return a RegressionBundle."""
    bundle = RegressionBundle(
        table2=run_table2(df),
        table4=run_table4(df),
        table5=run_table5(df),
        table6=run_table6(df),
        table7=run_table7(df),
    )
    return bundle
