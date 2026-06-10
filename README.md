# Shadow Trading — Replication Study

[![CI](https://github.com/tashrifulkabir34-lang/shadow-trading/actions/workflows/ci.yml/badge.svg)](https://github.com/tashrifulkabir34-lang/shadow-trading/actions)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue.svg)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-49%20passed-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Full replication of Mehta, Reeb & Zhao (2020), "Shadow Trading," *The Accounting Review*.**
> Production-grade Python pipeline implementing all five regression tables, economic significance
> calculations, a Shadow Trading Likelihood Score (STLS), and a self-contained HTML tearsheet.

---

## What Is Shadow Trading?

Shadow trading is a novel form of information exploitation documented by Mehta, Reeb & Zhao (2020):
corporate insiders use private, material information to trade **in economically-linked firms** —
their firm's suppliers, customers, or competitors — rather than in their own company's stock,
thereby circumventing SEC insider trading restrictions.

| Finding | Result |
|---------|--------|
| Existence | Informed trading in linked firms spikes 30 days before source firm announcements |
| Economic magnitude | 1 SD increase in source CAR → 6.4%–19.2% increase in informed trading |
| Profit per event | $139,400–$678,000 (vs. median SEC defendant profit: $60,000) |
| Mechanism | Substitution from own-firm trading after high-profile SEC enforcement spikes |
| IDD shock | Shadow trading increases 8.6%–20% after Inevitable Disclosure Doctrine adoption |
| Policing | Firm prohibition policies attenuate shadow trading by ~8–12% |

---

## Project Architecture

```
shadow-trading/
├── src/shadow_trading/
│   ├── data/
│   │   └── simulator.py              # Synthetic data calibrated to Table 1
│   ├── analysis/
│   │   ├── regressions.py            # Tables 2, 4, 5, 6, 7 (Panel OLS + 2-way clustered SE)
│   │   └── economic_significance.py  # Profit estimates & economic significance
│   ├── models/
│   │   └── stls.py                   # Shadow Trading Likelihood Score
│   └── visualization/
│       └── tearsheet.py              # Dark-themed HTML tearsheet generator
├── tests/
│   └── test_shadow_trading.py        # 49-test suite (5 classes)
├── outputs/                          # Generated tearsheet
├── .github/workflows/ci.yml          # Multi-version CI (Python 3.9/3.10/3.11)
├── main.py                           # End-to-end pipeline runner
├── requirements.txt
└── pyproject.toml
```

---

## Methodology

### Data & Sample

Calibrated to paper Table 1 descriptive statistics:

- **3,111** source firm–linked firm quarter observations (1,129 negative + 1,982 positive CAR)
- **598 source firms**, **745 linked firms** (1997–2011)
- Links via Hoberg-Phillips (2010/2016) product similarity and Ellis et al. (2012) customer disclosures

### Shadow Trading Proxies (Dependent Variables)

| Variable | Construction | Source |
|----------|-------------|--------|
| **Abnormal Short Sales** | (Pre-event short vol / non-event short vol) − 1 | Desai et al. 2002 |
| **Option/Stock Ratio** | Avg daily option vol / stock vol, 30-day pre-event | Johnson & So 2012 |
| **Order Imbalance** | (Inst. buys − sells) / total inst. volume | Puckett & Yan 2011 |

### Regression Specifications

**Equation (1) — Table 2:**
```
ShadowTrading = β₁·BPPartnerCAR + β₂·CompetitorCAR + β_x·Controls + ε
```

**Equation (2) — Table 5 (SEC enforcement spikes):**
```
ShadowTrading = β₁·BP_CAR + β₂·Comp_CAR + β₃·Post
              + β₄·BP_CAR×Post + β₅·Comp_CAR×Post + Controls + FE
```

**Equation (3) — Table 6 (IDD shock DiD):**
```
ShadowTrading = β₁·BP_CAR + β₂·Comp_CAR + β₃·IDDShock
              + β₄·BP_CAR×IDDShock + β₅·Comp_CAR×IDDShock + Controls + FE
```

All specifications: SE clustered by **linked firm × year** + **Year + FF48 Industry FE**.

### Shadow Trading Likelihood Score (STLS)

```
STLS_raw = 0.25·z(ABSS) + 0.50·z(OSR) + 0.25·z(OI)
STLS     = within-year percentile rank × 100
```

| Band | STLS Range | Interpretation |
|------|-----------|---------------|
| Low | 0–25 | Normal activity |
| Medium | 25–75 | Elevated; warrants monitoring |
| High | 75–100 | Strong shadow trading signal |

---

## Verification Matrix (Table 2 Key Coefficients)

| Specification | Paper β BP | Paper β Comp | Direction |
|--------------|------------|-------------|-----------|
| ABSS, Negative events | +0.033** | +0.031** | ✓ Both positive |
| ABSS, Positive events | −0.019** | −0.020** | ✓ Both negative |
| OSR, Negative events | +0.699** | +0.611** | ✓ Both positive |
| OSR, Positive events | +0.621** | +0.581** | ✓ Both positive |
| OI, Negative events | +0.011** | +0.009* | ✓ Both positive |
| OI, Positive events | +0.008** | +0.008** | ✓ Both positive |

**Profit estimate (paper footnote 14):**
- Abnormal shares: 2.6M daily × 30 days × 7.9% abnormal = ~205,000 shares
- **Range: $139,400–$678,000 per event** (205,000 × [$0.68, $3.30])

---

## Installation & Usage

```bash
git clone https://github.com/tashrifulkabir34-lang/shadow-trading.git
cd shadow-trading
pip install -r requirements.txt

# Run full pipeline → outputs/shadow_trading_tearsheet.html
python main.py
```

### Testing

```bash
pytest tests/ -v --tb=short
# Expected: 49 passed

# With coverage
pytest tests/ --cov=src/shadow_trading --cov-report=term-missing
```

---

## Real Data Sources

| Data | Source | Variables |
|------|--------|-----------|
| Short sale volume | NYSE / NASDAQ / FINRA | Abnormal Short Sales |
| Option trading | OptionMetrics | Option/Stock Ratio |
| Institutional trades | Ancerno | Order Imbalance |
| Stock returns | CRSP | CARs, Future Return |
| Firm fundamentals | Compustat | All controls |
| Competitor links | Hoberg-Phillips TNIC | Competitor CAR |
| Supplier-customer | Compustat Segment | Business Partner CAR |
| M&A data | SDC Platinum | M&A news events |
| Board connections | BoardEx | Director network controls |
| IDD shock dates | Klasa et al. (2018) | IDDShock |

---

## References

- **Mehta, M.N., Reeb, D.M., & Zhao, W. (2020).** Shadow Trading. *The Accounting Review*. [SSRN 3689154](https://ssrn.com/abstract=3689154)
- Johnson, T. & So, E. (2012). Option to stock volume ratio and future returns. *JFE*, 106, 262–286.
- Hoberg, G. & Phillips, G. (2016). Text-based network industries. *JPE*, 124, 1423–1465.
- Klasa, S. et al. (2018). Protection of trade secrets and capital structure. *JFE*, 128, 266–286.
- Puckett, A. & Yan, X.S. (2011). Interim trading skills of institutional investors. *JF*, 66, 601–633.

---

## License

MIT — see [LICENSE](LICENSE).

> **Disclaimer:** Academic replication only. Synthetic data is calibrated from published statistics;
> no proprietary data is reproduced. All methodology follows the cited paper exactly.
