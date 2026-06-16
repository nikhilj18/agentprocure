# modules/module5_optimizer.py
# PURPOSE: The final decision engine. Takes outputs from all
#   previous modules and produces the sourcing recommendation.
#
#   PASS 1: Score and pick best supplier per BOM line
#   PASS 2: Check portfolio constraints
#   PASS 3: Rebalance if constraints violated
#   PARETO: Generate 3 scenarios (cost / risk / balanced)
#
# Run standalone test: python3 modules/module5_optimizer.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from database.db_connect import run_query
from modules.module1_classifier import classify_bom
from modules.module2_supplier   import build_supplier_metrics, cluster_suppliers, rank_suppliers_for_part
from modules.module3_cost       import analyze_cost_for_part
from modules.module4_risk       import build_supplier_features, train_svm_classifier, compute_risk_score


# Portfolio constraints
CONCENTRATION_CAP   = 0.30   # No supplier > 30% of total BOM value
GEO_DIVERSITY_MIN   = 2      # At least 2 regions must be represented
SINGLE_SOURCE_CAP   = 0.20   # Max 20% of critical parts can be single-sourced


# ─────────────────────────────────────────────
# STEP 1: SCORE EACH BOM LINE
# ─────────────────────────────────────────────
def score_bom_lines(classified_bom, scenario='balanced'):
    """
    For each valid BOM line, picks the best supplier by
    combining TOPSIS score, landed cost, and risk score
    into one composite score.

    Scenario weights:
    - balanced:   equal weight to cost, quality, risk
    - lowest_cost: maximise cost saving
    - lowest_risk: minimise risk above all

    Returns a list of recommendation dicts.
    """
    # Prepare all ML components
    metrics_df, _, _       = cluster_suppliers(build_supplier_metrics())
    features_df            = build_supplier_features()
    svm, scaler, _, f_cols = train_svm_classifier(features_df)

    # Scenario weights: [topsis_weight, cost_weight, risk_weight]
    scenario_weights = {
        'balanced':    [0.40, 0.30, 0.30],
        'lowest_cost': [0.20, 0.60, 0.20],
        'lowest_risk': [0.20, 0.20, 0.60],
    }
    w_topsis, w_cost, w_risk = scenario_weights.get(
        scenario, scenario_weights['balanced'])

    recommendations = []

    for _, row in classified_bom.iterrows():
        if row.get('validation_status') != 'Valid':
            continue

        pno = row['part_no']
        qty = int(row.get('quantity_required', 1))
        cls = row['component_class']

        # Get top 3 suppliers from Module 2
        top3 = rank_suppliers_for_part(pno, cls, metrics_df)
        if top3.empty:
            continue

        best_rec    = None
        best_score  = -1
        backup_name = None

        for rank_idx, (_, sup_row) in enumerate(top3.iterrows()):
            sid     = int(sup_row['supplier_id'])
            country = sup_row['country']
            region  = sup_row['region']

            # Module 3: landed cost
            cost_result = analyze_cost_for_part(
                pno, sid, country, qty, cls)
            landed = float(cost_result['landed_cost'])

            # Module 4: risk score
            risk_score, risk_flags = compute_risk_score(
                pno, sid, cls, svm, scaler, features_df, f_cols)

            # TOPSIS score from Module 2
            topsis = float(sup_row['topsis_score'])

            # Normalize cost (lower is better → invert)
            # Use 1 / (1 + landed) so higher landed = lower score
            cost_score = 1.0 / (1.0 + landed)

            # Normalize risk (lower is better → invert)
            risk_norm = 1.0 - (risk_score / 100.0)

            # Composite score
            composite = (w_topsis * topsis +
                         w_cost   * cost_score +
                         w_risk   * risk_norm)

            if rank_idx == 0:
                # Store #1 as candidate
                best_score  = composite
                best_rec    = {
                    'part_no':             pno,
                    'quantity_required':   qty,
                    'component_class':     cls,
                    'recommended_supplier':sup_row['supplier_name'],
                    'supplier_id':         sid,
                    'supplier_country':    country,
                    'supplier_region':     region,
                    'topsis_score':        round(topsis, 4),
                    'composite_score':     round(composite, 4),
                    'landed_cost':         round(landed, 4),
                    'line_value':          round(landed * qty, 2),
                    'risk_score':          risk_score,
                    'risk_flags':          risk_flags,
                    'is_price_anomaly':    cost_result['is_price_anomaly'],
                    'anomaly_flag':        cost_result['anomaly_flag'],
                    'scenario':            scenario,
                    'swap_log':            [],
                }
            elif rank_idx == 1:
                backup_name = sup_row['supplier_name']
                # Track score gap for rebalancing later
                if best_rec:
                    best_rec['score_gap'] = round(best_score - composite, 4)
                    best_rec['backup_supplier']       = backup_name
                    best_rec['backup_supplier_id']    = sid
                    best_rec['backup_topsis']         = round(topsis, 4)
                    best_rec['backup_landed_cost']    = round(landed, 4)
                    best_rec['backup_risk_score']     = risk_score

        if best_rec:
            if 'backup_supplier' not in best_rec:
                best_rec['backup_supplier']    = 'None'
                best_rec['backup_supplier_id'] = None
                best_rec['backup_topsis']      = 0.0
                best_rec['backup_landed_cost'] = 0.0
                best_rec['backup_risk_score']  = 100
                best_rec['score_gap']          = 1.0
            recommendations.append(best_rec)

    return recommendations


# ─────────────────────────────────────────────
# STEP 2: CHECK PORTFOLIO CONSTRAINTS
# ─────────────────────────────────────────────
def check_constraints(recommendations):
    """
    Checks if the current allocation violates any constraints.
    Returns a list of violations with details.
    """
    violations = []
    df = pd.DataFrame(recommendations)

    if df.empty:
        return violations

    total_value = df['line_value'].sum()
    if total_value == 0:
        return violations

    # Check 1: Supplier concentration
    sup_value = df.groupby('recommended_supplier')['line_value'].sum()
    for sup, val in sup_value.items():
        pct = val / total_value
        if pct > CONCENTRATION_CAP:
            violations.append({
                'type':     'concentration',
                'supplier': sup,
                'pct':      round(pct, 3),
                'message':  f"{sup} = {pct:.0%} of BOM value (cap: {CONCENTRATION_CAP:.0%})"
            })

    # Check 2: Geographic diversity
    unique_regions = df['supplier_region'].nunique()
    if unique_regions < GEO_DIVERSITY_MIN:
        violations.append({
            'type':    'geo_diversity',
            'regions': unique_regions,
            'message': f"Only {unique_regions} region(s) — need at least {GEO_DIVERSITY_MIN}"
        })

    # Check 3: Critical single-source cap
    critical_df = df[df['component_class'] == 'Critical']
    if not critical_df.empty:
        single_src = critical_df[
            critical_df['risk_flags'].apply(
                lambda f: 'single_source' in ' '.join(f))
        ]
        single_pct = len(single_src) / len(critical_df)
        if single_pct > SINGLE_SOURCE_CAP:
            violations.append({
                'type':    'single_source',
                'pct':     round(single_pct, 3),
                'message': f"{single_pct:.0%} of critical parts are single-sourced (cap: {SINGLE_SOURCE_CAP:.0%})"
            })

    return violations


# ─────────────────────────────────────────────
# STEP 3: REBALANCE TO FIX VIOLATIONS
# ─────────────────────────────────────────────
def rebalance(recommendations, violations):
    """
    Swaps suppliers on violating lines to fix constraint violations.

    Strategy: swap lines with the SMALLEST score gap first
    (least quality loss) until constraints are satisfied.

    Logs every swap with a rationale.
    """
    if not violations:
        return recommendations, []

    swap_log = []
    recs = [r.copy() for r in recommendations]

    for violation in violations:
        if violation['type'] == 'concentration':
            # Find lines assigned to the over-concentrated supplier
            sup_name   = violation['supplier']
            candidates = [r for r in recs
                          if r['recommended_supplier'] == sup_name
                          and r.get('backup_supplier') not in ('None', None)
                          and r.get('backup_supplier_id') is not None]

            # Sort by smallest score gap (least quality sacrifice)
            candidates.sort(key=lambda x: x.get('score_gap', 1.0))

            df_temp = pd.DataFrame(recs)
            total   = df_temp['line_value'].sum()

            for cand in candidates:
                # Check if still violated
                sup_val = sum(r['line_value'] for r in recs
                              if r['recommended_supplier'] == sup_name)
                if total > 0 and sup_val / total <= CONCENTRATION_CAP:
                    break

                # Perform swap to backup supplier
                old_sup  = cand['recommended_supplier']
                new_sup  = cand['backup_supplier']
                old_cost = cand['landed_cost']
                new_cost = cand['backup_landed_cost']

                log_entry = (
                    f"SWAP [{cand['part_no']}]: "
                    f"{old_sup} → {new_sup} | "
                    f"Score gap: {cand.get('score_gap',0):.3f} | "
                    f"Cost: ${old_cost:.4f} → ${new_cost:.4f} | "
                    f"Reason: {violation['message']}"
                )
                swap_log.append(log_entry)

                # Update the recommendation
                for r in recs:
                    if r['part_no'] == cand['part_no']:
                        r['recommended_supplier'] = new_sup
                        r['landed_cost']          = new_cost
                        r['line_value']           = round(new_cost * r['quantity_required'], 2)
                        r['topsis_score']         = r.get('backup_topsis', 0)
                        r['risk_score']           = r.get('backup_risk_score', 50)
                        r['swap_log'].append(log_entry)
                        break

    return recs, swap_log


# ─────────────────────────────────────────────
# STEP 4: RUN ALL 3 PARETO SCENARIOS
# ─────────────────────────────────────────────
def generate_pareto_scenarios(classified_bom):
    """
    Runs the full pipeline for all 3 scenarios and returns
    a dict with results for each.
    """
    scenarios = {}
    for scenario in ['balanced', 'lowest_cost', 'lowest_risk']:
        print(f"  Running scenario: {scenario}...")
        recs         = score_bom_lines(classified_bom, scenario)
        violations   = check_constraints(recs)
        recs, swaps  = rebalance(recs, violations)
        scenarios[scenario] = {
            'recommendations': recs,
            'violations':      violations,
            'swap_log':        swaps,
            'total_value':     round(sum(r['line_value'] for r in recs), 2),
            'avg_risk':        round(np.mean([r['risk_score'] for r in recs]), 1) if recs else 0,
        }
    return scenarios


# ─────────────────────────────────────────────
# STEP 5: PRINT FINAL RECOMMENDATION TABLE
# ─────────────────────────────────────────────
def print_recommendations(recs, scenario_name, swap_log):
    print(f"\n{'='*70}")
    print(f"  Scenario: {scenario_name.upper()}")
    print(f"{'='*70}")
    print(f"  {'Part':<22} {'Supplier':<25} {'Score':>6} "
          f"{'Risk':>5} {'Landed$':>9} {'Total$':>10}")
    print("  " + "-"*75)

    total_val  = 0
    total_risk = 0

    for r in recs:
        icon = "🔴" if r['risk_score'] >= 60 else \
               "🟡" if r['risk_score'] >= 30 else "🟢"
        anom = " ⚠️" if r.get('is_price_anomaly') else ""
        print(f"  {icon} {r['part_no']:<22} "
              f"{r['recommended_supplier']:<25} "
              f"{r['topsis_score']:>6.4f} "
              f"{r['risk_score']:>5} "
              f"${r['landed_cost']:>8.4f} "
              f"${r['line_value']:>9.2f}{anom}")
        total_val  += r['line_value']
        total_risk += r['risk_score']

    avg_risk = total_risk / len(recs) if recs else 0
    print("  " + "-"*75)
    print(f"  {'TOTAL BOM VALUE':<54} ${total_val:>9.2f}")
    print(f"  Average risk score: {avg_risk:.1f}/100")

    if swap_log:
        print(f"\n  📋 Rebalancing swaps performed ({len(swap_log)}):")
        for s in swap_log:
            print(f"     {s}")


# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Module 5 — Recommendation Engine + Portfolio Optimizer")
    print("="*60)

    # Sample BOM
    sample_bom = pd.DataFrame([
        {'part_no': 'RES-0402-10K',   'quantity_required': 5000},
        {'part_no': 'CAP-0402-100N',  'quantity_required': 8000},
        {'part_no': 'IGBT-G4PC50W',   'quantity_required': 200},
        {'part_no': 'IC-MCU-STM32',   'quantity_required': 300},
        {'part_no': 'ASM-XFMR-LVCT', 'quantity_required': 100},
        {'part_no': 'ASM-INV-1KW',    'quantity_required': 50},
        {'part_no': 'MAG-CORE-ETD39', 'quantity_required': 500},
    ])

    print("\nStep 1: Classifying BOM...")
    classified, importance, accuracy = classify_bom(sample_bom)
    print(f"  {len(classified[classified['validation_status']=='Valid'])} valid parts classified")

    print("\nStep 2: Running 3 Pareto scenarios...")
    scenarios = generate_pareto_scenarios(classified)

    # Print all 3 scenarios
    for name, data in scenarios.items():
        print_recommendations(
            data['recommendations'], name, data['swap_log'])

    # Pareto comparison
    print(f"\n{'='*50}")
    print(f"  📊 Pareto Scenario Comparison")
    print(f"{'='*50}")
    print(f"  {'Scenario':<15} {'Total Value':>12} {'Avg Risk':>10}")
    print("  " + "-"*40)
    for name, data in scenarios.items():
        print(f"  {name:<15} ${data['total_value']:>11.2f} "
              f"{data['avg_risk']:>9.1f}/100")

    print(f"\n✅ Module 5 complete! All ML modules done.")
    print(f"   Next: Build the Streamlit dashboard.\n")