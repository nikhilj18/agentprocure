# AgentProcure — Results Summary

Metrics from a representative end-to-end run on the sample BOM
(`sample_bom.csv`, 22 parts) against the populated PostgreSQL dataset
(45 components, 54 suppliers, 405 AVL entries, 2,100 PO history records).

> Note: AgentProcure produces results **interactively** in the Streamlit
> dashboard. Recommendation CSVs are exported on demand via the in-app
> "Download Recommendations CSV" button. The figures in `../figures/`
> are captures of these live results. The numbers below were observed
> from the sample run shown in those figures.

## Module 1 — BOM Classification (Random Forest)
| Metric | Value |
|---|---|
| Random Forest accuracy | 93% |
| Total parts classified | 22 |
| Invalid parts | 0 |
| Class distribution | Custom 36.4% · Commodity 36.4% · Critical 27.3% |
| Top feature drivers | `unit_cost_avg`, `category_enc` |

## Module 2 — Supplier Ranking (K-Means + TOPSIS)
| Metric | Value |
|---|---|
| Ranking method | TOPSIS with class-specific weights |
| Example (ASM-CT-PANEL, Custom) top score | 0.8076 |

## Module 3 — Cost Analysis (ARIMA + FRED)
| Metric | Value |
|---|---|
| Total BOM value | $3,466.54 |
| Average landed cost / part | $19.3035 |
| Price anomalies flagged | 0 |
| ARIMA MAPE — copper (3-mo walk-forward backtest, 98 mo) | 13.74% |
| ARIMA MAPE — aluminum (3-mo walk-forward backtest, 98 mo) | 7.00% |
| ARIMA MAPE — average | 10.37% |

## Module 4 — Risk Assessment (SVM)
| Metric | Value |
|---|---|
| SVM classifier accuracy (in-sample) | 95.9% |
| Average risk score | 45.7 / 100 |
| High-risk parts | 2 |
| Medium-risk parts | 20 |
| Low-risk parts | 0 |

## Module 5 — Sourcing Optimizer
| Metric | Value |
|---|---|
| Scenarios | Balanced · Lowest Cost · Lowest Risk |
| Parts sourced | 22 |
| Swaps performed (balanced) | 2 |
| Cost savings vs. baseline (cheapest vs. worst-case qualified supplier, sample BOM) | $1,427.44 (8.46%) |
| Baseline / optimized landed total | $16,870.64 / $15,443.20 |
