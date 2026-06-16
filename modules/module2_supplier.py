# modules/module2_supplier.py  (UPDATED — ESG scoring added)
# PURPOSE: For each component in the BOM, ranks suppliers using:
#   1. K-Means clustering — groups suppliers into performance tiers
#   2. TOPSIS scoring — ranks suppliers by 6 criteria:
#      OTIF, quality, cost, lead time, ISO cert, ESG score
#      with weights that SHIFT based on component class:
#      Commodity → cost matters most
#      Critical  → delivery matters most
#      Custom    → quality + ESG matters most
#
# Run standalone test: python3 modules/module2_supplier.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from database.db_connect import run_query


# ─────────────────────────────────────────────
# ESG SCORING CONFIG
# ─────────────────────────────────────────────
# Country-level ESG risk index (0-1, higher = better ESG environment)
# Source: World Bank Governance Indicators + ESG country rankings
# 1.0 = best ESG environment, 0.0 = worst
COUNTRY_ESG_INDEX = {
    'Germany':     0.92,
    'USA':         0.85,
    'South Korea': 0.78,
    'Taiwan':      0.75,
    'India':       0.52,
    'Vietnam':     0.45,
    'China':       0.42,
}

# ISO 14001 (environmental) adds 0.15 bonus on top of country index
# ISO 9001 (quality) adds 0.10 bonus
# These are already tracked in supplier_master.iso_certified
ISO_ESG_BONUS = 0.15


def compute_esg_score(country, iso_certified, reject_rate=0.0):
    """
    Computes a composite ESG score (0-1) for a supplier.

    Three components:
    1. Country governance index (environmental + social policy proxy)
    2. ISO certification bonus (14001 environmental management)
    3. Quality practice score (low reject rate = less waste)

    Higher score = better ESG performance.
    """
    # Component 1: Country-level ESG baseline (50% weight)
    country_score = COUNTRY_ESG_INDEX.get(country, 0.50)

    # Component 2: ISO certification bonus (30% weight)
    iso_score = (1.0 + ISO_ESG_BONUS) if iso_certified else 1.0
    iso_component = min(1.0, country_score * iso_score)

    # Component 3: Quality/waste proxy (20% weight)
    # Lower reject rate = less material waste = better ESG
    quality_score = max(0.0, 1.0 - (reject_rate * 5))

    # Weighted composite
    esg = (0.50 * country_score +
           0.30 * iso_component +
           0.20 * quality_score)

    return round(min(1.0, esg), 4)


# ─────────────────────────────────────────────
# STEP 1: BUILD SUPPLIER PERFORMANCE METRICS
# ─────────────────────────────────────────────
def build_supplier_metrics():
    """
    Calculates real performance metrics for each supplier
    from PO history. Now includes ESG score.
    """
    sql = """
        SELECT
            s.supplier_id,
            s.supplier_name,
            s.country,
            s.region,
            s.supplier_tier,
            s.iso_certified,
            COUNT(p.po_line_id)                          AS order_count,
            AVG(CASE WHEN p.on_time_flag THEN 1.0
                     ELSE 0.0 END)                       AS otif_rate,
            AVG(CASE WHEN p.quality_status = 'Pass'
                     THEN 1.0 ELSE 0.0 END)              AS quality_rate,
            AVG(p.unit_price)                            AS avg_unit_price,
            AVG(p.actual_delivery_date - p.po_date)      AS avg_lead_time_days,
            AVG(CAST(p.reject_quantity AS FLOAT) /
                NULLIF(p.quantity, 0))                   AS avg_reject_rate
        FROM supplier_master s
        LEFT JOIN po_history p ON s.supplier_id = p.supplier_id
        GROUP BY s.supplier_id, s.supplier_name, s.country,
                 s.region, s.supplier_tier, s.iso_certified
        HAVING COUNT(p.po_line_id) > 0
        ORDER BY s.supplier_id
    """
    df = run_query(sql)

    df['otif_rate']         = df['otif_rate'].fillna(0.5).astype(float)
    df['quality_rate']      = df['quality_rate'].fillna(0.5).astype(float)
    df['avg_unit_price']    = df['avg_unit_price'].fillna(999.0).astype(float)
    df['avg_lead_time_days']= df['avg_lead_time_days'].fillna(60.0).astype(float)
    df['order_count']       = df['order_count'].fillna(0).astype(int)
    df['avg_reject_rate']   = df['avg_reject_rate'].fillna(0.05).astype(float)

    # ── NEW: Compute ESG score for each supplier ──────────────
    df['esg_score'] = df.apply(
        lambda row: compute_esg_score(
            country=row['country'],
            iso_certified=bool(row['iso_certified']),
            reject_rate=float(row['avg_reject_rate'])
        ), axis=1
    )

    return df


# ─────────────────────────────────────────────
# STEP 2: K-MEANS CLUSTERING
# ─────────────────────────────────────────────
def cluster_suppliers(metrics_df):
    """
    Groups suppliers into 3 performance tiers using K-Means.
    Now includes ESG score as a clustering feature.
    """
    features = ['otif_rate', 'quality_rate', 'avg_lead_time_days', 'esg_score']
    X = metrics_df[features].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    metrics_df = metrics_df.copy()
    metrics_df['cluster'] = kmeans.labels_

    cluster_quality = metrics_df.groupby('cluster')['otif_rate'].mean()
    sorted_clusters = cluster_quality.sort_values(ascending=False).index.tolist()
    tier_map = {
        sorted_clusters[0]: 'Tier 1 — High Performer',
        sorted_clusters[1]: 'Tier 2 — Mid Performer',
        sorted_clusters[2]: 'Tier 3 — Low Performer',
    }
    metrics_df['performance_tier'] = metrics_df['cluster'].map(tier_map)

    return metrics_df, kmeans, scaler


# ─────────────────────────────────────────────
# STEP 3: TOPSIS SCORING
# ─────────────────────────────────────────────
def topsis_score(decision_matrix, weights, benefit_criteria):
    """
    TOPSIS multi-criteria scoring.
    Now supports 6 criteria including ESG.
    """
    m, n = decision_matrix.shape
    col_norms = np.sqrt((decision_matrix**2).sum(axis=0))
    col_norms[col_norms == 0] = 1e-10
    norm_matrix = decision_matrix / col_norms
    weighted    = norm_matrix * np.array(weights)

    ideal_best  = np.where(benefit_criteria,
                           weighted.max(axis=0),
                           weighted.min(axis=0))
    ideal_worst = np.where(benefit_criteria,
                           weighted.min(axis=0),
                           weighted.max(axis=0))

    dist_best  = np.sqrt(((weighted - ideal_best)**2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst)**2).sum(axis=1))

    denom = dist_best + dist_worst
    denom[denom == 0] = 1e-10
    return dist_worst / denom


# ─────────────────────────────────────────────
# STEP 4: DYNAMIC WEIGHTS — NOW WITH ESG
# ─────────────────────────────────────────────
def get_weights_for_class(component_class):
    """
    Returns TOPSIS weights for 6 criteria:
    [otif_rate, quality_rate, cost_index, lead_time, iso_cert, esg_score]

    ESG weight is highest for Custom components (bespoke manufacturing
    relationships carry higher ESG accountability) and lowest for
    Commodity (price dominates).

    All weights sum to 1.0.
    """
    weights = {
        #              otif   qual   cost   lt     iso    esg
        'Commodity': [0.20,  0.18,  0.37,  0.13,  0.04,  0.08],
        'Critical':  [0.32,  0.23,  0.14,  0.18,  0.04,  0.09],
        'Custom':    [0.18,  0.35,  0.13,  0.13,  0.08,  0.13],
    }
    return weights.get(component_class, weights['Commodity'])


# ─────────────────────────────────────────────
# STEP 5: RANK SUPPLIERS FOR A COMPONENT
# ─────────────────────────────────────────────
def rank_suppliers_for_part(part_no, component_class, metrics_df):
    """
    Ranks approved suppliers for a given part using TOPSIS
    with ESG as the 6th criterion.
    """
    avl_sql = """
        SELECT a.supplier_id
        FROM approved_vendor_list a
        WHERE a.part_no = %s AND a.qualification_status = 'Qualified'
    """
    avl_df = run_query(avl_sql, params=(part_no,))

    if avl_df.empty:
        return pd.DataFrame()

    approved_ids = avl_df['supplier_id'].tolist()
    sup_df = metrics_df[metrics_df['supplier_id'].isin(approved_ids)].copy()

    if sup_df.empty or len(sup_df) < 2:
        if not sup_df.empty:
            sup_df['topsis_score'] = 1.0
            sup_df['rank'] = 1
        return sup_df

    # Build decision matrix with ESG as 6th column
    max_price = sup_df['avg_unit_price'].max()
    sup_df['cost_index'] = max_price / sup_df['avg_unit_price'].clip(lower=0.001)
    sup_df['iso_num']    = sup_df['iso_certified'].astype(float)

    decision_matrix = sup_df[[
        'otif_rate',
        'quality_rate',
        'cost_index',
        'avg_lead_time_days',
        'iso_num',
        'esg_score'           # ← NEW 6th criterion
    ]].values.astype(float)

    weights = get_weights_for_class(component_class)

    # benefit: True = higher is better
    # lead_time = False (lower is better)
    benefit = [True, True, True, False, True, True]

    scores = topsis_score(decision_matrix, weights, benefit)
    sup_df = sup_df.copy()
    sup_df['topsis_score'] = np.round(scores, 4)
    sup_df['rank'] = sup_df['topsis_score'].rank(ascending=False).astype(int)
    sup_df = sup_df.sort_values('topsis_score', ascending=False)

    return sup_df.head(3)


# ─────────────────────────────────────────────
# STEP 6: RANK ALL PARTS IN BOM
# ─────────────────────────────────────────────
def rank_suppliers_for_bom(classified_bom_df):
    metrics_df, _, _ = cluster_suppliers(build_supplier_metrics())
    rankings = {}
    for _, row in classified_bom_df.iterrows():
        if row.get('validation_status') != 'Valid':
            continue
        pno  = row['part_no']
        cls  = row['component_class']
        top3 = rank_suppliers_for_part(pno, cls, metrics_df)
        rankings[pno] = top3
    return rankings, metrics_df


# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Module 2 — Supplier Intelligence + ESG Scoring Test")
    print("="*60)

    metrics_df, _, _ = cluster_suppliers(build_supplier_metrics())

    # Show ESG scores for all suppliers
    print(f"\n🌱 Supplier ESG Scores:")
    print(f"  {'Supplier':<30} {'Country':<14} {'ESG':>5} "
          f"{'ISO':>5} {'Tier'}")
    print("  " + "-"*70)

    for _, row in metrics_df.sort_values('esg_score', ascending=False).iterrows():
        iso  = "✅" if row['iso_certified'] else "❌"
        tier = row['performance_tier'].split('—')[0].strip()
        print(f"  {row['supplier_name']:<30} {row['country']:<14} "
              f"{row['esg_score']:>5.3f} {iso:>5}  {tier}")

    # Test TOPSIS with ESG for 3 representative parts
    test_parts = [
        ('RES-0402-10K',  'Commodity'),
        ('IGBT-G4PC50W',  'Critical'),
        ('ASM-XFMR-LVCT', 'Custom'),
    ]

    print(f"\n🏆 TOPSIS Rankings with ESG (6 criteria):")
    for pno, cls in test_parts:
        top3 = rank_suppliers_for_part(pno, cls, metrics_df)
        print(f"\n  [{cls}] {pno}")
        print(f"  {'Rank':<5} {'Supplier':<28} {'TOPSIS':>7} "
              f"{'OTIF':>6} {'ESG':>6} {'ISO':>5}")
        print("  " + "-"*62)
        if top3.empty:
            print("  No approved suppliers found.")
        else:
            for _, r in top3.iterrows():
                iso = "✅" if r['iso_certified'] else "❌"
                print(f"  #{int(r['rank']):<4} {r['supplier_name']:<28} "
                      f"{r['topsis_score']:>7.4f} "
                      f"{r['otif_rate']:>6.0%} "
                      f"{r['esg_score']:>6.3f} "
                      f"{iso:>5}")

    # Show weight comparison
    print(f"\n⚖️  TOPSIS Weight Comparison (with ESG as 6th criterion):")
    print(f"  {'Class':<12} {'OTIF':>6} {'Qual':>6} {'Cost':>6} "
          f"{'LT':>6} {'ISO':>6} {'ESG':>6}")
    print("  " + "-"*50)
    for cls in ['Commodity', 'Critical', 'Custom']:
        w = get_weights_for_class(cls)
        print(f"  {cls:<12} {w[0]:>6.0%} {w[1]:>6.0%} {w[2]:>6.0%} "
              f"{w[3]:>6.0%} {w[4]:>6.0%} {w[5]:>6.0%}")

    print(f"\n✅ Module 2 with ESG scoring complete!\n")