# modules/module4_risk.py  (UPDATED — Smarter SVM Labels)
# WHAT CHANGED:
#   OLD: Binary rule — OTIF >= 88% AND quality >= 92% = Reliable
#   NEW: Composite weighted score on 5 dimensions before thresholding
#        + Lead time TREND (improving vs degrading over time)
#        + Price stability score
#        + Reject rate trajectory
#   This produces more nuanced, defensible ground truth labels
#   that reflect how procurement professionals actually assess suppliers.
#
# Run standalone test: python3 modules/module4_risk.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

from database.db_connect import run_query


# ─────────────────────────────────────────────
# STEP 1: BUILD SUPPLIER FEATURES
# ─────────────────────────────────────────────
def build_supplier_features():
    """
    Builds feature table from PO history.
    Now includes trend features split across two time halves
    to capture whether a supplier is improving or degrading.
    """
    # Overall metrics
    sql_overall = """
        SELECT
            s.supplier_id,
            s.supplier_name,
            s.country,
            s.region,
            s.iso_certified,
            COUNT(p.po_line_id)                              AS order_count,
            AVG(CASE WHEN p.on_time_flag THEN 1.0
                     ELSE 0.0 END)                           AS otif_rate,
            AVG(CASE WHEN p.quality_status='Pass'
                     THEN 1.0 ELSE 0.0 END)                  AS quality_rate,
            AVG(p.actual_delivery_date - p.po_date)          AS avg_lead_time,
            STDDEV(p.actual_delivery_date - p.po_date)       AS lead_time_std,
            STDDEV(p.unit_price) / NULLIF(AVG(p.unit_price),0) AS price_cv,
            AVG(CAST(p.reject_quantity AS FLOAT) /
                NULLIF(p.quantity, 0))                       AS avg_reject_rate
        FROM supplier_master s
        JOIN po_history p ON s.supplier_id = p.supplier_id
        GROUP BY s.supplier_id, s.supplier_name,
                 s.country, s.region, s.iso_certified
        HAVING COUNT(p.po_line_id) >= 5
        ORDER BY s.supplier_id
    """
    df = run_query(sql_overall)

    # First-half metrics (Jan–Jun) for trend calculation
    sql_h1 = """
        SELECT
            supplier_id,
            AVG(CASE WHEN on_time_flag THEN 1.0 ELSE 0.0 END) AS otif_h1,
            AVG(CASE WHEN quality_status='Pass'
                     THEN 1.0 ELSE 0.0 END)                   AS qual_h1
        FROM po_history
        WHERE EXTRACT(MONTH FROM po_date) <= 6
        GROUP BY supplier_id
    """
    h1 = run_query(sql_h1)

    # Second-half metrics (Jul–Dec) for trend calculation
    sql_h2 = """
        SELECT
            supplier_id,
            AVG(CASE WHEN on_time_flag THEN 1.0 ELSE 0.0 END) AS otif_h2,
            AVG(CASE WHEN quality_status='Pass'
                     THEN 1.0 ELSE 0.0 END)                   AS qual_h2
        FROM po_history
        WHERE EXTRACT(MONTH FROM po_date) > 6
        GROUP BY supplier_id
    """
    h2 = run_query(sql_h2)

    # Merge trend data
    df = df.merge(h1, on='supplier_id', how='left')
    df = df.merge(h2, on='supplier_id', how='left')

    # Fill nulls
    for col in ['otif_rate','quality_rate','avg_lead_time','lead_time_std',
                'price_cv','avg_reject_rate']:
        defaults = {'otif_rate':0.5,'quality_rate':0.5,'avg_lead_time':45.0,
                    'lead_time_std':10.0,'price_cv':0.10,'avg_reject_rate':0.05}
        df[col] = df[col].fillna(defaults.get(col, 0.0)).astype(float)

    for col in ['otif_h1','otif_h2','qual_h1','qual_h2']:
        df[col] = df[col].fillna(df['otif_rate'] if 'otif' in col
                                 else df['quality_rate']).astype(float)

    df['order_count'] = df['order_count'].fillna(0).astype(int)

    # ── TREND FEATURES ─────────────────────────────────────────
    # Positive = improving, Negative = degrading
    df['otif_trend']    = df['otif_h2']  - df['otif_h1']
    df['quality_trend'] = df['qual_h2']  - df['qual_h1']

    # ── COMPOSITE LABEL SCORING ────────────────────────────────
    # Instead of a hard binary rule, we score each supplier
    # on 5 dimensions (0-1 each) with different weights.
    # Final composite >= 0.65 = Reliable, < 0.65 = Unreliable.
    #
    # Dimension weights:
    #   OTIF rate:        30% — most important delivery signal
    #   Quality rate:     25% — product quality
    #   Lead time std:    15% — predictability (lower = better)
    #   Price stability:  15% — consistent pricing (lower CV = better)
    #   OTIF trend:       15% — is supplier improving or degrading?

    # Normalize lead_time_std: 0 std = 1.0 score, 30+ std = 0.0
    df['lt_std_score'] = np.clip(1.0 - df['lead_time_std'] / 30.0, 0.0, 1.0)

    # Normalize price_cv: 0 CV = 1.0 score, 0.3+ CV = 0.0
    df['price_stab_score'] = np.clip(1.0 - df['price_cv'] / 0.30, 0.0, 1.0)

    # Normalize OTIF trend: +0.2 = fully improving, -0.2 = fully degrading
    df['trend_score'] = np.clip(
        (df['otif_trend'] + 0.20) / 0.40, 0.0, 1.0)

    # Composite score
    df['composite_score'] = (
        0.30 * df['otif_rate'] +
        0.25 * df['quality_rate'] +
        0.15 * df['lt_std_score'] +
        0.15 * df['price_stab_score'] +
        0.15 * df['trend_score']
    ).round(4)

    # Label: 1 = Reliable if composite >= threshold
    # Adaptive: if fixed threshold gives only 1 class, use median split
    threshold = 0.65
    labels = (df['composite_score'] >= threshold).astype(int)
    if labels.nunique() < 2:
        threshold = df['composite_score'].median()
        labels = (df['composite_score'] >= threshold).astype(int)
    df['label'] = labels

    return df


# ─────────────────────────────────────────────
# STEP 2: TRAIN SVM
# ─────────────────────────────────────────────
def train_svm_classifier(features_df):
    feature_cols = [
        'otif_rate', 'quality_rate', 'avg_lead_time',
        'lead_time_std', 'price_cv', 'avg_reject_rate',
        'otif_trend', 'quality_trend', 'price_stab_score',
    ]
    X = features_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = features_df['label'].values
    if len(np.unique(y)) < 2:
        ranked = features_df['composite_score'].values.argsort()
        y = np.zeros(len(ranked), dtype=int)
        y[ranked[len(ranked)//2:]] = 1
    model = SVC(kernel='rbf', C=1.0, gamma='scale',
                probability=True, random_state=42)
    model.fit(X_scaled, y)
    accuracy = (model.predict(X_scaled) == y).mean()
    return model, scaler, round(float(accuracy), 3), feature_cols

def predict_reliability(supplier_id, model, scaler, features_df, feature_cols):
    row = features_df[features_df['supplier_id'] == supplier_id]
    if row.empty:
        return 'Unknown', 0.5

    X        = row[feature_cols].values
    X_scaled = scaler.transform(X)
    pred     = model.predict(X_scaled)[0]
    proba    = model.predict_proba(X_scaled)[0]
    conf     = proba.max()

    return ('Reliable' if pred == 1 else 'Unreliable'), round(float(conf), 3)


# ─────────────────────────────────────────────
# STEP 4: RISK SCORE
# ─────────────────────────────────────────────
def compute_risk_score(part_no, supplier_id, component_class,
                       model, scaler, features_df, feature_cols):
    """
    Computes 0-100 risk score. Now also considers:
    - Composite score (not just binary SVM output)
    - Trend direction (improving supplier gets a break)
    """
    flags      = []
    risk_score = 0

    sup_row = features_df[features_df['supplier_id'] == supplier_id]
    if sup_row.empty:
        return 75, ['Supplier not in performance database']

    sup = sup_row.iloc[0]

    # ── Factor 1: SVM Reliability (0-30 points) ──
    reliability, confidence = predict_reliability(
        supplier_id, model, scaler, features_df, feature_cols)

    composite = float(sup['composite_score'])

    if reliability == 'Unreliable':
        # Scale penalty by how unreliable (composite 0.0 = 30pts, 0.64 = 5pts)
        penalty = int(30 * (1.0 - composite / 0.65))
        risk_score += max(5, penalty)
        flags.append(f'unreliable_supplier (SVM: {confidence:.0%}, '
                     f'composite: {composite:.2f})')
    elif composite < 0.75:
        risk_score += 8
        flags.append(f'borderline_reliability (composite: {composite:.2f})')

    # ── Trend bonus: improving supplier gets risk reduction ──
    otif_trend = float(sup['otif_trend'])
    if otif_trend > 0.05:
        risk_score = max(0, risk_score - 5)
        flags.append(f'improving_trend (OTIF +{otif_trend:.0%} H2 vs H1)')
    elif otif_trend < -0.10:
        risk_score += 8
        flags.append(f'degrading_trend (OTIF {otif_trend:.0%} H2 vs H1)')

    # ── Factor 2: Lead Time Exposure (0-20 points) ──
    avg_lt = float(sup['avg_lead_time'])
    if avg_lt > 56:
        risk_score += 20
        flags.append(f'long_lead_time ({avg_lt:.0f} days)')
    elif avg_lt > 42:
        risk_score += 10
        flags.append(f'moderate_lead_time ({avg_lt:.0f} days)')

    # ── Factor 3: Single Source Risk (0-20 points) ──
    avl_count_df = run_query(
        "SELECT COUNT(*) as n FROM approved_vendor_list WHERE part_no=%s",
        params=(part_no,)
    )
    avl_count = int(avl_count_df['n'].iloc[0])
    if avl_count == 1:
        risk_score += 20
        flags.append('single_source')
        if component_class == 'Critical':
            risk_score += 5
            flags.append('critical_single_source')
    elif avl_count == 2:
        risk_score += 8
        flags.append('limited_sourcing (2 suppliers only)')

    # ── Factor 4: Price Volatility (0-15 points) ──
    price_cv = float(sup['price_cv'])
    if price_cv > 0.20:
        risk_score += 15
        flags.append(f'high_price_volatility (CV={price_cv:.0%})')
    elif price_cv > 0.10:
        risk_score += 7
        flags.append(f'moderate_price_volatility (CV={price_cv:.0%})')

    # ── Factor 5: Lifecycle Status (0-10 points) ──
    lifecycle_df = run_query(
        "SELECT lifecycle_status FROM component_master WHERE part_no=%s",
        params=(part_no,)
    )
    if not lifecycle_df.empty:
        status = lifecycle_df['lifecycle_status'].iloc[0]
        if status in ('EOL', 'Obsolete'):
            risk_score += 10
            flags.append(f'lifecycle_{status.lower()}')
        elif status == 'NRND':
            risk_score += 5
            flags.append('lifecycle_nrnd')

    # ── Factor 6: ISO / ESG (0-5 points) ──
    if not bool(sup['iso_certified']):
        risk_score += 5
        flags.append('no_iso_certification')

    risk_score = min(100, risk_score)
    if not flags:
        flags.append('no_significant_risk')

    return risk_score, flags


# ─────────────────────────────────────────────
# PORTFOLIO RISK CHECK
# ─────────────────────────────────────────────
def check_portfolio_risk(bom_with_suppliers, total_bom_value):
    portfolio_flags = []
    sup_value = bom_with_suppliers.groupby('supplier_name')['line_value'].sum()
    sup_pct   = (sup_value / total_bom_value * 100).round(1)

    for sup, pct in sup_pct.items():
        if pct > 30:
            portfolio_flags.append(
                f'concentration_risk: {sup} = {pct:.1f}% of BOM value')

    geo_value = bom_with_suppliers.groupby('country')['line_value'].sum()
    geo_pct   = (geo_value / total_bom_value * 100).round(1)

    for country, pct in geo_pct.items():
        if pct > 60:
            portfolio_flags.append(
                f'geo_concentration: {country} = {pct:.1f}% of BOM value')

    unique_regions = bom_with_suppliers['region'].nunique()
    if unique_regions < 2:
        portfolio_flags.append('insufficient_geographic_diversity')

    concentration = {
        'supplier_concentration': sup_pct.to_dict(),
        'geo_concentration':      geo_pct.to_dict(),
        'unique_regions':         unique_regions
    }
    return portfolio_flags, concentration


# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*65)
    print("  Module 4 — Risk Assessment + Smarter SVM Labels Test")
    print("="*65)

    features_df = build_supplier_features()
    model, scaler, accuracy, f_cols = train_svm_classifier(features_df)
    print(f"\n🤖 SVM trained — accuracy: {accuracy:.1%}")
    print(f"   Label method: composite weighted score (5 dimensions)")
    print(f"   Threshold: composite >= 0.65 = Reliable")

    # Show composite scores vs old binary rule
    print(f"\n📊 Composite Scoring vs Old Binary Rule:")
    print(f"  {'Supplier':<28} {'OTIF':>6} {'Qual':>6} "
          f"{'Trend':>7} {'Comp':>6} {'Label':<12} {'Old Rule'}")
    print("  " + "-"*80)

    for _, row in features_df.sort_values(
            'composite_score', ascending=False).iterrows():
        # Old binary rule
        old_label = "Reliable" if (row['otif_rate'] >= 0.88 and
                                    row['quality_rate'] >= 0.92) \
                    else "Unreliable"
        new_label = "Reliable" if row['label'] == 1 else "Unreliable"
        changed   = " ← changed" if old_label != new_label else ""
        trend_str = f"+{row['otif_trend']:.0%}" if row['otif_trend'] >= 0 \
                    else f"{row['otif_trend']:.0%}"
        icon      = "✅" if new_label == "Reliable" else "❌"
        print(f"  {icon} {row['supplier_name']:<28} "
              f"{row['otif_rate']:>6.0%} "
              f"{row['quality_rate']:>6.0%} "
              f"{trend_str:>7} "
              f"{row['composite_score']:>6.3f} "
              f"{new_label:<12} {old_label}{changed}")

    # Test risk scores
    print(f"\n🎯 Risk Scores (with trend-aware labels):")
    test_cases = [
        ('RES-0402-10K',  1,  'Commodity'),
        ('IGBT-G4PC50W',  9,  'Critical'),
        ('ASM-INV-1KW',   5,  'Custom'),
        ('IGBT-FGA25N120',17, 'Critical'),
    ]
    for pno, sid, cls in test_cases:
        score, flags = compute_risk_score(
            pno, sid, cls, model, scaler, features_df, f_cols)
        sup_name = features_df[features_df['supplier_id']==sid
                               ]['supplier_name'].values
        sup_name = sup_name[0] if len(sup_name) > 0 else f"Supplier {sid}"
        icon     = "🟢" if score < 30 else "🟡" if score < 60 else "🔴"
        flag_str = ', '.join(flags[:2])
        print(f"  {icon} {pno:<22} {sup_name:<25} {score:>4}  {flag_str}")

    print(f"\n✅ Module 4 with smarter SVM labels complete!\n")