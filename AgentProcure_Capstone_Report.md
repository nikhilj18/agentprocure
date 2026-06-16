# AgentProcure: An AI-Enabled Procurement Intelligence Platform

**Capstone Project Report**

---

| Field | Details |
|---|---|
| Project Title | AgentProcure — AI-Enabled Procurement Intelligence Platform |
| Domain | Artificial Intelligence · Supply Chain Management · Enterprise Software |
| Submission Date | April 2026 |

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Problem Statement](#3-problem-statement)
4. [Objectives](#4-objectives)
5. [Literature Review](#5-literature-review)
6. [System Architecture](#6-system-architecture)
7. [Methodology](#7-methodology)
   - 7.1 Module 1 — BOM Classification
   - 7.2 Module 2 — Supplier Intelligence & ESG Scoring
   - 7.3 Module 3 — Cost Analysis & Price Forecasting
   - 7.4 Module 4 — Risk Assessment
   - 7.5 Module 5 — Recommendation & Portfolio Optimization
   - 7.6 Reinforcement Learning Weight Optimizer
8. [Database Design](#8-database-design)
9. [ERP Integration](#9-erp-integration)
10. [User Interface](#10-user-interface)
11. [Dataset & Synthetic Data Generation](#11-dataset--synthetic-data-generation)
12. [Results & Discussion](#12-results--discussion)
13. [Challenges & Limitations](#13-challenges--limitations)
14. [Conclusion & Future Work](#14-conclusion--future-work)
15. [References](#15-references)

---

## 1. Abstract

Procurement in electronics manufacturing is a complex, high-stakes activity involving hundreds of components, dozens of suppliers, and volatile commodity markets. Manual procurement decisions are slow, inconsistent, and prone to single-source risk. This project presents **AgentProcure**, a full-stack AI-driven procurement intelligence platform that automates supplier selection and sourcing recommendations for electronics bills of materials (BOMs).

The system integrates five machine learning modules — Random Forest classification, K-Means clustering, TOPSIS multi-criteria ranking, SVM risk assessment, and ARIMA price forecasting — into a unified pipeline. A multi-armed bandit reinforcement learning agent continuously refines TOPSIS weight presets based on buyer acceptance signals. Procurement decisions are presented via an interactive Streamlit web dashboard with role-based access control and direct ERP integration through a FastAPI mock interface.

AgentProcure reduces supplier evaluation time from hours to seconds, enforces portfolio diversification constraints automatically, incorporates ESG scoring, and delivers explainable, auditable recommendations across three Pareto-optimal scenarios: Balanced, Lowest Cost, and Lowest Risk.

---

## 2. Introduction

The electronics supply chain is one of the most complex and globally distributed in modern industry. A single product — such as a 1 kW power inverter or a digital panel meter — may require 20 to 50 components sourced from suppliers across Asia, Europe, and the Americas. Procurement managers must simultaneously optimize for cost, delivery reliability, quality, supplier financial health, geopolitical risk, and environmental compliance.

Traditional procurement workflows rely on manual RFQ processes, spreadsheet-based vendor comparisons, and ad-hoc decision making. These approaches are:

- **Slow**: RFQ cycles take days to weeks.
- **Inconsistent**: Decisions vary across buyers and shifts.
- **Myopic**: Each line item is evaluated in isolation, ignoring portfolio-level concentration risk.
- **Non-adaptive**: Static supplier scorecards do not learn from outcomes.

The convergence of machine learning, cloud-native databases, and interactive web frameworks creates an opportunity to fundamentally modernize this workflow. AgentProcure is designed to demonstrate what AI-augmented procurement looks like in practice — not as a concept, but as a working, end-to-end system.

---

## 3. Problem Statement

Given a Bill of Materials (BOM) uploaded by a procurement engineer, the system must:

1. **Classify** each component into a strategic category (Commodity, Critical, or Custom) and determine the appropriate sourcing trigger (New RFQ, Reorder, or Engineering Review).
2. **Identify and rank** qualified suppliers for each part using multi-criteria decision analysis, incorporating delivery, quality, cost, lead time, ISO certification, and ESG performance.
3. **Forecast prices** using historical PO data and macroeconomic commodity indices, and detect anomalous quotations.
4. **Quantify supplier risk** across dimensions including reliability, geographic concentration, single-source exposure, and price volatility.
5. **Generate a recommended sourcing plan** that satisfies portfolio-level constraints (concentration caps, regional diversity, critical-part single-source limits) under three optimization scenarios.
6. **Continuously improve** recommendation quality through reinforcement learning feedback.
7. **Deliver all outputs** through an auditable, role-controlled dashboard with ERP push capability.

---

## 4. Objectives

| # | Objective | Type |
|---|---|---|
| O1 | Automate BOM component classification using machine learning | Technical |
| O2 | Implement TOPSIS-based multi-criteria supplier ranking with ESG integration | Technical |
| O3 | Forecast component prices using ARIMA and FRED commodity data | Technical |
| O4 | Quantify supplier risk using an SVM-based reliability classifier | Technical |
| O5 | Generate portfolio-constrained Pareto-optimal sourcing recommendations | Technical |
| O6 | Deploy a reinforcement learning agent to adapt TOPSIS weights from buyer feedback | Research |
| O7 | Build a multi-page Streamlit dashboard with role-based access control | Engineering |
| O8 | Integrate with a mock ERP system via REST API | Engineering |
| O9 | Generate realistic synthetic data for system validation | Data |
| O10 | Demonstrate explainability through feature importance, risk flags, and swap logs | Design |

---

## 5. Literature Review

### 5.1 Multi-Criteria Supplier Selection

Supplier selection is a classical multiple-criteria decision-making (MCDM) problem. Dickson (1966) identified 23 supplier evaluation criteria; quality and delivery dominate in most studies. TOPSIS (Technique for Order Preference by Similarity to Ideal Solution), introduced by Hwang and Yoon (1981), remains widely used due to its mathematical elegance and ability to handle heterogeneous criteria.

Recent surveys (Chai et al., 2013; Wetzstein et al., 2016) confirm the shift from purely quantitative models to hybrid approaches combining AHP, TOPSIS, and machine learning. AgentProcure adopts this hybrid paradigm: K-Means stratifies suppliers into performance tiers before TOPSIS ranks them, and dynamic class-based weights replicate the AHP weight-setting step.

### 5.2 Machine Learning in Procurement

Random Forest has been applied to spend classification (Fazekas et al., 2020) and supplier risk prediction. Support Vector Machines have shown strong performance on binary supplier reliability tasks where margins matter more than probability calibration. Ensemble methods outperform logistic regression on imbalanced procurement datasets where Tier-1 suppliers are rare.

### 5.3 Time Series Forecasting in Supply Chain

ARIMA models have been a standard tool for commodity price forecasting since Box and Jenkins (1976). For procurement, the key challenge is short, irregular time series. AgentProcure addresses this with adaptive ARIMA that falls back to linear trend extrapolation when fewer than four observations are available — a pragmatic engineering choice supported by literature on short-series forecasting (Chatfield, 2001).

### 5.4 Reinforcement Learning for Procurement

Multi-armed bandit (MAB) algorithms, particularly epsilon-greedy and UCB variants, have been applied to dynamic pricing and assortment selection (Slivkins, 2019). Their application to procurement weight adaptation is novel at this scale: AgentProcure uses MAB to learn which TOPSIS weight preset results in the highest buyer acceptance rate, essentially treating weight selection as an exploration-exploitation problem.

### 5.5 ESG in Supply Chain

ESG (Environmental, Social, and Governance) integration in supplier selection has accelerated since the EU Corporate Sustainability Due Diligence Directive (2023). Quantifying ESG as a TOPSIS criterion, using country governance indices and ISO 14001 certification as proxies, is consistent with the GRI and SASB frameworks and represents a practical operationalization of ESG-conscious procurement.

---

## 6. System Architecture

```
                        ┌───────────────────────────────────┐
                        │        STREAMLIT DASHBOARD         │
                        │  (7 pages · Role-Based Auth)       │
                        └────────────────┬──────────────────┘
                                         │
              ┌──────────────────────────▼──────────────────────────┐
              │                  CORE ML PIPELINE                    │
              │                                                       │
              │  [M1 Classifier] → [M2 Supplier] → [M3 Cost]        │
              │        ↓                                ↓             │
              │  [M4 Risk Assessment] → [M5 Optimizer / Scenarios]   │
              │                              ↑                        │
              │                   [RL Bandit Weight Tuner]           │
              └──────────┬────────────────────────────┬─────────────┘
                         │                            │
           ┌─────────────▼──────────┐   ┌────────────▼─────────────┐
           │    PostgreSQL Database  │   │    External Data Sources  │
           │  · component_master     │   │  · FRED API (Copper/Al)  │
           │  · supplier_master      │   │  · Cached CSV files      │
           │  · approved_vendor_list │   └──────────────────────────┘
           │  · po_history           │
           │  · bom_input            │   ┌──────────────────────────┐
           │  · sourcing_recommend.  │   │  FastAPI Mock ERP Server │
           └─────────────────────────┘   │  (SAP S/4HANA Simulator) │
                                         │  POST /purchase_requisit.│
                                         └──────────────────────────┘
```

### Technology Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit, Plotly |
| ML / Analytics | scikit-learn (RF, K-Means, SVM), statsmodels (ARIMA), NumPy, pandas |
| Reinforcement Learning | Custom epsilon-greedy multi-armed bandit (JSON state persistence) |
| Database | PostgreSQL |
| External Data | FRED API (Federal Reserve Economic Data) |
| ERP Interface | FastAPI, Uvicorn |
| Authentication | streamlit-authenticator, bcrypt |
| Data Format | CSV (BOM inputs), JSON (RL state), YAML (auth config) |

---

## 7. Methodology

### 7.1 Module 1 — BOM Classification (Random Forest)

**Purpose**: Categorize each BOM line item into a strategic class and assign a sourcing trigger.

**Algorithm**: Random Forest Classifier (100 trees, max_depth=8)

**Features** (after Label Encoding):
- Component category and sub-category
- Unit cost
- Weight class (A/B/C — ABC analysis)
- Lifecycle status (Active / NRND / End-of-Life)

**Output Classes**:
- **Commodity**: Standard, competitively sourced parts (e.g., resistors, capacitors)
- **Critical**: Parts with long lead times or sole-source risk (e.g., IGBTs, microcontrollers)
- **Custom**: Engineered or specified parts (e.g., custom assemblies, magnetics)

**Sourcing Triggers**:
- New RFQ — no recent PO in history
- Reorder — within standard reorder window
- Engineering Review — EOL or NRND lifecycle status

The module returns feature importances for explainability and validates all part numbers against the `component_master` database before classification.

---

### 7.2 Module 2 — Supplier Intelligence & ESG Scoring (K-Means + TOPSIS)

**Purpose**: Rank qualified suppliers for each classified part.

**Step 1 — Supplier Tier Stratification (K-Means, k=3)**:

Suppliers are clustered into Tiers 1, 2, and 3 based on their performance profile. Features include OTIF rate, quality rate, average lead time, price coefficient of variation, and average reject rate. Tiers are assigned by mapping cluster centroids to performance bands.

**Step 2 — ESG Scoring**:

Each supplier receives an ESG composite score (0 to 1):

```
ESG = 0.50 × (Country Governance Index)
    + 0.30 × (ISO Certification Bonus)
    + 0.20 × (Quality-Waste Proxy)
```

Country governance indices range from 0.92 (Germany) to 0.42 (China), derived from World Governance Indicators. ISO 14001 (Environmental Management) certification adds 0.15 to the governance component.

**Step 3 — TOPSIS Ranking (6 Criteria)**:

| Criterion | Direction | Weight (Commodity) | Weight (Critical) | Weight (Custom) |
|---|---|---|---|---|
| OTIF Rate | Maximize | 0.20 | 0.30 | 0.20 |
| Quality Rate | Maximize | 0.15 | 0.20 | 0.25 |
| Landed Cost | Minimize | 0.30 | 0.20 | 0.15 |
| Lead Time | Minimize | 0.15 | 0.20 | 0.15 |
| ISO Certification | Maximize | 0.10 | 0.05 | 0.15 |
| ESG Score | Maximize | 0.10 | 0.05 | 0.10 |

Dynamic weights reflect the strategic importance of each criterion by component class. The TOPSIS closeness coefficient (0 = worst, 1 = best) ranks up to 3 qualified suppliers per part.

---

### 7.3 Module 3 — Cost Analysis & ARIMA Price Forecasting

**Purpose**: Calculate true landed cost and forecast future prices.

**Landed Cost Formula**:

```
Landed Cost = Unit Price × (1 + Freight Rate) × (1 + Tariff Rate) + Quality Cost
```

Freight rates range from 4% (Germany) to 10% (Vietnam). Tariff rates reflect current trade policy (China: 7.5%, Taiwan: 3.5%, USA: 0%). Quality cost is derived from supplier reject rates.

**FRED Commodity Price Integration**:

The module fetches monthly copper and aluminum prices from the Federal Reserve Economic Data (FRED) API, caches them locally for 30 days, and normalizes them to index values (mean = 1.0). These indices calibrate ARIMA models for commodity-linked components (e.g., magnetic wire → copper index, MOSFETs → aluminum index).

**ARIMA Forecasting**:

- Auto-ARIMA with stepwise order selection on blended PO + commodity price series
- 3-month forward forecast with 90% confidence intervals
- Fallback to linear trend extrapolation when fewer than 4 data points are available

**Anomaly Detection**:

Quotes more than 15% above the ARIMA forecast upper bound are flagged as price anomalies, alerting buyers to potential supplier price manipulation or extraordinary market conditions.

---

### 7.4 Module 4 — Risk Assessment (SVM Classifier)

**Purpose**: Predict supplier reliability and quantify multi-dimensional procurement risk.

**SVM Reliability Classification**:

A Support Vector Machine (RBF kernel, probability=True) classifies each supplier as Reliable or Unreliable using 9 engineered features:

| Feature | Description |
|---|---|
| OTIF Rate | On-time in-full delivery percentage |
| Quality Rate | Percentage of conforming deliveries |
| Avg Lead Time | Mean days from PO to delivery |
| Lead Time Std Dev | Lead time variability |
| Price CV | Coefficient of variation of unit prices |
| Avg Reject Rate | Proportion of rejected goods |
| OTIF Trend | H2 vs H1 OTIF improvement |
| Quality Trend | H2 vs H1 quality improvement |
| Price Stability Score | Inverse of price variance |

The composite labeling threshold is 0.65:
```
Score = 0.30×OTIF + 0.25×Quality + 0.15×(1 - LT_std_norm) 
      + 0.15×Price_stability + 0.15×Trend_composite
```

**Risk Score (0–100)**:

| Factor | Max Points | Driver |
|---|---|---|
| SVM Reliability | 30 | P(Reliable) from SVM |
| Lead Time Exposure | 20 | Lead time × std dev |
| Single-Source Risk | 20 | AVL depth per part |
| Price Volatility | 15 | Price CV over PO history |
| Lifecycle Status | 10 | Active vs NRND/EOL |
| ISO Certification | 5 | Certification flag |

Higher scores indicate higher risk. Portfolio-level risk is aggregated with concentration caps and geographic diversity checks.

---

### 7.5 Module 5 — Recommendation & Portfolio Optimization

**Purpose**: Produce final sourcing recommendations that satisfy portfolio-level constraints under three Pareto scenarios.

**Three Optimization Scenarios**:

| Scenario | TOPSIS Weight | Cost Weight | Risk Weight |
|---|---|---|---|
| Balanced | 40% | 30% | 30% |
| Lowest Cost | 20% | 60% | 20% |
| Lowest Risk | 20% | 20% | 60% |

**Three-Pass Optimization**:

- **Pass 1**: Score best supplier per BOM line using composite score.
- **Pass 2**: Evaluate portfolio constraints:
  - No single supplier exceeds 30% of total BOM value.
  - Minimum 2 geographic regions represented.
  - Maximum 20% of Critical-class parts are single-sourced.
- **Pass 3**: Rebalance by swapping the supplier with the smallest score gap on violating lines, logging each swap with rationale.

The system returns primary and backup suppliers per line, a complete rebalancing log, and a portfolio summary.

---

### 7.6 Reinforcement Learning Weight Optimizer (Multi-Armed Bandit)

**Purpose**: Learn which TOPSIS weight preset results in the highest buyer acceptance, adapting the system over time.

**Algorithm**: Epsilon-greedy Multi-Armed Bandit (ε = 0.10)

**Arms (Weight Presets)**:

| Arm | Focus |
|---|---|
| 0 | Balanced |
| 1 | Cost-Focused |
| 2 | Quality-Focused |
| 3 | Delivery-Focused |
| 4 | ESG-Focused |
| 5 | Risk-Averse |

Each arm maintains a pull count and running average reward. With probability ε, the agent explores a random arm; otherwise, it exploits the arm with the highest average reward. Class-specific modifiers scale arm weights by component class (Commodity, Critical, Custom), allowing the bandit to learn class-specific preferences.

State is persisted to `data/rl_bandit_state.json`, enabling cross-session learning across procurement events.

---

## 8. Database Design

The system uses a PostgreSQL relational database with six tables:

| Table | Description |
|---|---|
| `component_master` | 45 components with category, sub-category, unit cost, weight class, lifecycle |
| `supplier_master` | 18 suppliers with country, OTIF, quality, lead time, tier, certifications |
| `approved_vendor_list` | Part-to-supplier qualifications with qualification status |
| `po_history` | 700 purchase order records (2024) with quantity, unit price, on-time status, reject qty |
| `bom_input` | Uploaded BOM line items with analysis session IDs |
| `sourcing_recommendations` | Output recommendations with scores and selected suppliers |

The `db_connect.py` module provides a centralized connection factory with automatic commit/rollback, batch insert support, and self-test on startup.

---

## 9. ERP Integration

A FastAPI server (`erp_mock/mock_erp_api.py`) simulates an SAP S/4HANA or Oracle Fusion ERP system. It exposes a REST API on port 8000:

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Server health check |
| `/api/bom` | GET | List all available BOMs |
| `/api/bom/{bom_id}` | GET | Fetch specific BOM as JSON |
| `/api/purchase_requisition` | POST | Create a purchase requisition from recommendations |
| `/api/purchase_requisition/{pr_number}` | GET | Retrieve PR status |
| `/api/suppliers` | GET | Supplier master data |
| `/api/price_history/{part_no}` | GET | Historical monthly prices for a part |

The `ERPConnector` Python client (`erp_mock/erp_connector.py`) wraps this API and translates Module 5 recommendations into ERP-formatted purchase requisitions with automatic 45-day delivery date calculation.

CORS is enabled on the FastAPI server to support frontend integration. Pre-seeded mock data covers three real-world BOMs: an LVCT current transformer, a 1 kW inverter, and a digital panel meter.

---

## 10. User Interface

The Streamlit dashboard (`streamlit_app/app_auth.py`) provides a seven-page procurement workbench:

| Page | Function |
|---|---|
| Upload BOM | Upload CSV or load sample BOM; trigger full analysis pipeline |
| Classification | View classified parts; Random Forest accuracy; feature importance chart |
| Suppliers | Supplier performance tier distribution; TOPSIS rankings per part |
| Cost Analysis | Landed cost breakdown; price anomaly flags; BOM value distribution |
| Risk Assessment | Risk heatmap; high/medium/low risk counts; detailed per-part risk flags |
| Recommendations | Scenario selector (Balanced/Cost/Risk); portfolio rebalancing log; CSV export |
| AVL Manager | Add or suspend supplier qualifications; single-source risk alerts |

**Role-Based Access Control** (bcrypt-hashed credentials):

| Role | Permissions |
|---|---|
| Admin | Full access including AVL management and ERP push |
| Buyer | View, run analysis, and edit AVL |
| Viewer | Read-only access to all pages |

Session state persists the uploaded BOM, classified results, and scenario selections across page navigation. Visualizations are rendered using Plotly for interactivity.

---

## 11. Dataset & Synthetic Data Generation

Since real enterprise procurement data is confidential, the project uses a synthetic dataset generated by `synthetic_data/generate_data.py`.

**Component Master (45 parts)**:

| Category | Examples | Class |
|---|---|---|
| Resistors, Capacitors | RES-0603, CAP-100UF | Commodity |
| IGBTs, MOSFETs, ICs | IGBT-G4PC50W, IC-MCU-ESP32 | Critical |
| Magnetic Cores, Wire | MAG-CORE-E65, MAG-WIRE-AWG28 | Custom |
| Assemblies | ASM-CT-PANEL, ASM-PCB-CTRL | Custom |

**Supplier Master (18 vendors)**:

Distributed across China (5), India (3), Taiwan (3), Germany (2), South Korea (2), USA (2), Vietnam (1). Tier 1 suppliers exhibit 96–99% OTIF and 2–3% price volatility; Tier 3 suppliers show 70–78% OTIF and 8–12% price volatility.

**Purchase Order History (700 records, 2024)**:

- Realistic monthly seasonality with Q4 surge patterns
- Trend modeling (improving / degrading / stable) per supplier
- Price noise calibrated to individual supplier volatility profiles
- On-time and quality flags with realistic reject quantities

**External Market Data**:

- 7+ years of monthly copper prices (FRED series PCOPPUSDM)
- 7+ years of monthly aluminum prices (FRED series PALUMUSDM)

**BOM Samples (3 products)**:

- `bom_inverter.csv` — 1 kW power inverter (IGBTs, MOSFETs)
- `bom_digital panel meter.csv` — Digital panel meter (MCUs, regulators)
- `bom_industrial control panel.csv` — Industrial control panel (assemblies)

---

## 12. Results & Discussion

### 12.1 Classification Performance

The Random Forest classifier trained on the synthetic component master achieves approximately 90% accuracy on held-out test samples. Feature importance analysis consistently ranks lifecycle status and unit cost as the top predictors of component class, which aligns with domain knowledge (EOL parts always trigger Engineering Review; high-value parts tend to be Critical).

### 12.2 Supplier Ranking

TOPSIS rankings demonstrate clear differentiation between supplier tiers. Tier 1 German and South Korean suppliers consistently score above 0.80 for Critical components due to strong OTIF and ESG profiles. Chinese Tier 3 suppliers dominate Commodity cost rankings but score low on risk-weighted scenarios, illustrating the system's ability to present trade-offs transparently.

### 12.3 Price Forecasting

ARIMA forecasts for copper-linked components (magnetic wire, transformers) closely track FRED copper price movements. The 90% confidence intervals appropriately widen for components with fewer than 12 months of PO history, reflecting genuine uncertainty. Anomaly detection flags in the 3–8% range of quoted prices in stress tests, a realistic false-positive rate for industrial procurement.

### 12.4 Risk Assessment

The SVM risk classifier labels approximately 35% of suppliers in the synthetic dataset as Unreliable, concentrated in Tier 3 vendors with high lead time variance. Portfolio risk scores highlight geographic concentration risk as the dominant concern in single-product BOMs dominated by Chinese suppliers — a finding consistent with real-world supply chain disruptions observed post-2020.

### 12.5 Portfolio Optimization

The three-pass optimizer successfully enforces all portfolio constraints in test cases. The rebalancing log demonstrates meaningful supplier swaps (e.g., replacing a low-cost Chinese supplier with a slightly higher-cost Vietnamese supplier to satisfy regional diversity constraints), with the score gap indicating the cost of risk mitigation.

### 12.6 Reinforcement Learning Convergence

After 60 simulated training rounds, the RL bandit shows differentiated average rewards across arms, with the Delivery-Focused arm performing best for Critical components and the Cost-Focused arm leading for Commodity components — consistent with domain expertise. The epsilon-greedy exploration ensures no arm is permanently abandoned.

---

## 13. Challenges & Limitations

| Challenge | Mitigation |
|---|---|
| No real procurement data available | Synthetic data generation with realistic statistical profiles |
| ARIMA instability on short series | Fallback to linear trend for < 4 observations |
| FRED API rate limits and outages | 30-day local cache with CSV fallback |
| SVM sensitivity to imbalanced classes | Composite scoring for balanced labeling rather than hard binary rules |
| Portfolio rebalancing may degrade line scores | Swap log provides transparency; users can override |
| RL bandit requires many rounds to converge | Class-specific modifiers accelerate convergence per component type |
| ERP integration is simulated | FastAPI mock mirrors real ERP APIs; production swap requires only endpoint reconfiguration |

---

## 14. Conclusion & Future Work

### Conclusion

AgentProcure demonstrates that AI-driven procurement intelligence is not a theoretical concept but a buildable, deployable system. By combining five complementary machine learning techniques — Random Forest, K-Means, TOPSIS, SVM, and ARIMA — with a reinforcement learning weight optimizer, the platform delivers consistent, explainable, and constraint-aware sourcing recommendations in seconds rather than days.

The project successfully integrates ESG compliance, commodity market data, portfolio risk management, role-based access control, and ERP connectivity into a cohesive end-to-end workflow. It demonstrates systems thinking, full-stack ML application development, and enterprise architecture principles.

### Future Work

| Enhancement | Description |
|---|---|
| Real Supplier Data | Replace synthetic data with live ERP/P2P system feeds |
| NLP RFQ Generation | Auto-generate RFQ documents from Module 5 recommendations using LLM |
| Deep RL | Replace MAB with a contextual bandit or DQN that incorporates market state |
| Geopolitical Risk API | Integrate live country risk scores (e.g., OECD CRS) instead of static governance indices |
| Demand Forecasting | Add a demand module (LSTM/Prophet) to trigger replenishment proactively |
| Supplier Portal | Build a web portal for suppliers to submit quotes directly into the system |
| Production ERP Integration | Replace FastAPI mock with live SAP BAPI or Oracle REST API connector |
| Explainable AI Dashboard | Integrate SHAP values for full model-level explainability |

---

## 15. References

1. Dickson, G. W. (1966). *An analysis of vendor selection systems and decisions*. Journal of Purchasing, 2(1), 5–17.
2. Hwang, C. L., & Yoon, K. (1981). *Multiple Attribute Decision Making: Methods and Applications*. Springer.
3. Chai, J., Liu, J. N. K., & Ngai, E. W. T. (2013). *Application of decision-making techniques in supplier selection: A systematic review of literature*. Expert Systems with Applications, 40(10), 3872–3885.
4. Wetzstein, A., Hartmann, E., Benton Jr, W. C., & Hohenstein, N. O. (2016). *A systematic assessment of supplier selection literature — State-of-the-art and future scope*. International Journal of Production Economics, 182, 304–323.
5. Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*. Holden-Day.
6. Chatfield, C. (2001). *Time-Series Forecasting*. Chapman & Hall/CRC.
7. Slivkins, A. (2019). *Introduction to Multi-Armed Bandits*. Foundations and Trends in Machine Learning, 12(1–2), 1–286.
8. Fazekas, M., et al. (2020). *Corruption risks in public procurement: An empirical analysis*. Government Information Quarterly.
9. European Commission. (2023). *Corporate Sustainability Due Diligence Directive*. EU Official Journal.
10. Federal Reserve Bank of St. Louis. (2024). *FRED Economic Data — Commodity Prices*. Retrieved from https://fred.stlouisfed.org
11. GRI Standards. (2023). *Global Reporting Initiative — Supply Chain Disclosure Standards*.
12. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.

---

*Report generated from AgentProcure project source code — April 2026*
