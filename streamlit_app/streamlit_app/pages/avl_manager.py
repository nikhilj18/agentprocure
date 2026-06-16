# streamlit_app/app.py
# PURPOSE: The web dashboard — 6 pages that let a buyer
#   upload a BOM and get full sourcing recommendations.
#
# Run with: streamlit run streamlit_app/app.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from modules.module1_classifier import classify_bom
from modules.module2_supplier   import build_supplier_metrics, cluster_suppliers, rank_suppliers_for_part
from modules.module3_cost       import analyze_cost_for_part
from modules.module4_risk       import build_supplier_features, train_svm_classifier, compute_risk_score
from modules.module5_optimizer  import score_bom_lines, check_constraints, rebalance, generate_pareto_scenarios

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AgentProcure",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
st.sidebar.title("🔧 AgentProcure")
st.sidebar.caption("AI-Enabled Procurement Intelligence")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["📤 Upload BOM",
     "🏷️ Classification",
     "🏆 Suppliers",
     "💰 Cost Analysis",
     "⚠️ Risk Assessment",
     "✅ Recommendations",
     "📋 AVL Manager"],
    index=0
)

st.sidebar.divider()
st.sidebar.caption("Built with Random Forest · TOPSIS · ARIMA · SVM · Claude API")

# ─────────────────────────────────────────────
# SESSION STATE — persists data across pages
# ─────────────────────────────────────────────
if 'bom_df'        not in st.session_state: st.session_state.bom_df        = None
if 'classified'    not in st.session_state: st.session_state.classified    = None
if 'importance'    not in st.session_state: st.session_state.importance    = None
if 'scenarios'     not in st.session_state: st.session_state.scenarios     = None
if 'metrics_df'    not in st.session_state: st.session_state.metrics_df    = None
if 'features_df'   not in st.session_state: st.session_state.features_df   = None
if 'svm_model'     not in st.session_state: st.session_state.svm_model     = None
if 'svm_scaler'    not in st.session_state: st.session_state.svm_scaler    = None
if 'svm_fcols'     not in st.session_state: st.session_state.svm_fcols     = None

# ─────────────────────────────────────────────
# HELPER: colour risk score
# ─────────────────────────────────────────────
def risk_colour(score):
    if score >= 60: return "🔴"
    if score >= 30: return "🟡"
    return "🟢"

# ══════════════════════════════════════════════
# PAGE 1 — UPLOAD BOM
# ══════════════════════════════════════════════
# ══════════════════════════════════════════════
# PAGE 7 — AVL MANAGER
# ══════════════════════════════════════════════
elif page == "📋 AVL Manager":
    from database.db_connect import run_query, run_insert
    from datetime import date

    st.title("📋 AVL Manager — Approved Vendor List")
    st.caption("Add, suspend, or update supplier qualifications in real time.")

    @st.cache_data(ttl=30)
    def load_avl():
        return run_query("""
            SELECT a.avl_id, a.part_no, c.description, c.component_class,
                   s.supplier_name, s.country, s.region,
                   a.qualification_status, a.qualified_date, a.supplier_id
            FROM approved_vendor_list a
            JOIN component_master c ON a.part_no = c.part_no
            JOIN supplier_master  s ON a.supplier_id = s.supplier_id
            ORDER BY a.part_no, a.qualification_status
        """)

    @st.cache_data(ttl=60)
    def load_parts():
        return run_query(
            "SELECT part_no, description, component_class "
            "FROM component_master ORDER BY part_no")

    @st.cache_data(ttl=60)
    def load_suppliers():
        return run_query(
            "SELECT supplier_id, supplier_name, country "
            "FROM supplier_master WHERE status='Active' ORDER BY supplier_name")

    avl_df      = load_avl()
    parts_df    = load_parts()
    supplier_df = load_suppliers()

    qualified = avl_df[avl_df['qualification_status'] == 'Qualified']
    suspended = avl_df[avl_df['qualification_status'] == 'Suspended']
    part_supplier_count = qualified.groupby('part_no')['supplier_id'].count()
    single_source_parts = part_supplier_count[part_supplier_count == 1]
    critical_parts      = parts_df[parts_df['component_class']=='Critical']['part_no']
    critical_single     = single_source_parts[single_source_parts.index.isin(critical_parts)]

    # Health metrics
    st.subheader("📊 AVL Health Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total AVL Entries",   len(avl_df))
    col2.metric("Qualified",           len(qualified))
    col3.metric("Suspended",           len(suspended),
                delta=f"{len(suspended)} suspended" if len(suspended) > 0 else None,
                delta_color="inverse")
    col4.metric("Single-Source Parts", len(single_source_parts),
                delta=f"{len(critical_single)} critical" if len(critical_single) > 0 else None,
                delta_color="inverse")

    if len(critical_single) > 0:
        st.warning(f"⚠️ {len(critical_single)} Critical components are single-sourced: "
                   f"{', '.join(critical_single.index.tolist()[:5])}. Add backup suppliers.")

    st.divider()

    # View AVL
    st.subheader("🔍 Current AVL Entries")
    col_f1, col_f2, col_f3 = st.columns(3)
    filter_class  = col_f1.selectbox("Filter by class",  ["All","Commodity","Critical","Custom"])
    filter_status = col_f2.selectbox("Filter by status", ["All","Qualified","Suspended","Under Review"])
    filter_search = col_f3.text_input("Search part number", placeholder="e.g. IGBT")

    display_df = avl_df.copy()
    if filter_class  != "All": display_df = display_df[display_df['component_class']==filter_class]
    if filter_status != "All": display_df = display_df[display_df['qualification_status']==filter_status]
    if filter_search:           display_df = display_df[display_df['part_no'].str.contains(filter_search, case=False)]

    def style_status(val):
        if val == 'Qualified':    return 'background-color:#d4edda;color:#155724'
        if val == 'Suspended':    return 'background-color:#f8d7da;color:#721c24'
        if val == 'Under Review': return 'background-color:#fff3cd;color:#856404'
        return ''

    show_cols = ['part_no','description','component_class',
                 'supplier_name','country','qualification_status','qualified_date']
    st.dataframe(
        display_df[show_cols].style.map(style_status, subset=['qualification_status']),
        use_container_width=True, height=300)
    st.caption(f"Showing {len(display_df)} of {len(avl_df)} entries")

    st.divider()

    # Add new qualification
    st.subheader("➕ Add New Supplier Qualification")
    with st.form("add_avl_form"):
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            selected_part = col_a1.selectbox("Select Component", parts_df['part_no'].tolist())
            current_sups  = qualified[qualified['part_no']==selected_part]['supplier_name'].tolist()
            if current_sups: st.info(f"Currently qualified: {', '.join(current_sups)}")
            else:             st.warning("No qualified suppliers — single-source risk!")
        with col_a2:
            already_ids    = avl_df[(avl_df['part_no']==selected_part) &
                                    (avl_df['qualification_status']=='Qualified')]['supplier_id'].tolist()
            available_sups = supplier_df[~supplier_df['supplier_id'].isin(already_ids)]
            if available_sups.empty:
                st.info("All suppliers already qualified.")
                new_supplier_id = None
            else:
                sup_options     = available_sups.apply(lambda r: f"{r['supplier_name']} ({r['country']})", axis=1).tolist()
                selected_label  = st.selectbox("Select New Supplier", sup_options)
                new_supplier_id = available_sups.iloc[sup_options.index(selected_label)]['supplier_id']
            qual_date = st.date_input("Qualification Date", value=date.today())

        if st.form_submit_button("✅ Add Qualification", type="primary", use_container_width=True):
            if new_supplier_id:
                try:
                    run_insert(
                        """INSERT INTO approved_vendor_list
                           (part_no, supplier_id, qualification_status, qualified_date)
                           VALUES (%s,%s,%s,%s)
                           ON CONFLICT (part_no, supplier_id)
                           DO UPDATE SET qualification_status='Qualified', qualified_date=%s""",
                        (selected_part, int(new_supplier_id), 'Qualified', qual_date, qual_date))
                    st.success(f"✅ Supplier qualified for {selected_part}!")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()

    # Update / suspend
    st.subheader("✏️ Update or Suspend Qualification")
    with st.form("update_avl_form"):
        col_u1, col_u2, col_u3 = st.columns(3)
        avl_options    = avl_df.apply(lambda r: f"{r['part_no']} — {r['supplier_name']}", axis=1).tolist()
        selected_entry = col_u1.selectbox("Select AVL Entry", avl_options)
        idx            = avl_options.index(selected_entry)
        selected_avl_id= avl_df.iloc[idx]['avl_id']
        current_status = avl_df.iloc[idx]['qualification_status']
        status_opts    = ["Qualified","Suspended","Under Review"]
        new_status     = col_u2.selectbox("New Status", status_opts,
                             index=status_opts.index(current_status)
                             if current_status in status_opts else 0)
        new_date       = col_u3.date_input("Update Date", value=date.today())

        if st.form_submit_button("💾 Update Entry", use_container_width=True):
            try:
                run_insert(
                    "UPDATE approved_vendor_list SET qualification_status=%s, qualified_date=%s WHERE avl_id=%s",
                    (new_status, new_date, int(selected_avl_id)))
                icon = "✅" if new_status=="Qualified" else "⏸️" if new_status=="Suspended" else "🔍"
                st.success(f"{icon} Updated to: {new_status}")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    if len(single_source_parts) > 0:
        st.divider()
        st.subheader("⚠️ Single-Source Risk Detail")
        single_df = qualified[qualified['part_no'].isin(single_source_parts.index)][
            ['part_no','component_class','supplier_name','country','qualification_status']].copy()
        single_df['risk'] = single_df['component_class'].map(
            {'Critical':'🔴 High','Custom':'🟡 Medium','Commodity':'🟢 Low'})
        st.dataframe(single_df, use_container_width=True, hide_index=True)
    st.title("📤 Upload Bill of Materials")
    st.markdown("Upload a CSV file with your BOM. The system will validate, "
                "classify, and generate sourcing recommendations.")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader(
            "Choose a BOM CSV file",
            type=['csv'],
            help="CSV must contain columns: part_no, quantity_required"
        )

        if uploaded:
            try:
                bom_df = pd.read_csv(uploaded)
                required_cols = {'part_no', 'quantity_required'}
                if not required_cols.issubset(bom_df.columns):
                    st.error(f"CSV must contain: {required_cols}")
                else:
                    st.session_state.bom_df = bom_df
                    st.success(f"✅ BOM loaded — {len(bom_df)} line items")
                    st.dataframe(bom_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error reading file: {e}")

    with col2:
        st.markdown("### Or use Sample BOM")
        if st.button("📋 Load Sample BOM", use_container_width=True):
            sample = pd.DataFrame([
                {'part_no': 'RES-0402-10K',   'quantity_required': 5000},
                {'part_no': 'CAP-0402-100N',  'quantity_required': 8000},
                {'part_no': 'IGBT-G4PC50W',   'quantity_required': 200},
                {'part_no': 'IC-MCU-STM32',   'quantity_required': 300},
                {'part_no': 'ASM-XFMR-LVCT', 'quantity_required': 100},
                {'part_no': 'ASM-INV-1KW',    'quantity_required': 50},
                {'part_no': 'MAG-CORE-ETD39', 'quantity_required': 500},
                {'part_no': 'MOSFET-IRF540',  'quantity_required': 400},
            ])
            st.session_state.bom_df = sample
            st.success("✅ Sample BOM loaded!")
            st.dataframe(sample, use_container_width=True)

        st.markdown("### CSV Format")
        st.code("part_no,quantity_required\nRES-0402-10K,5000\nIGBT-G4PC50W,200", language='csv')

    if st.session_state.bom_df is not None:
        st.divider()
        if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True):
            with st.spinner("Running classification, supplier scoring, cost analysis, risk assessment..."):
                try:
                    classified, importance, accuracy = classify_bom(st.session_state.bom_df)
                    st.session_state.classified = classified
                    st.session_state.importance = importance

                    metrics_df, _, _ = cluster_suppliers(build_supplier_metrics())
                    st.session_state.metrics_df = metrics_df

                    features_df = build_supplier_features()
                    svm, scaler, _, fcols = train_svm_classifier(features_df)
                    st.session_state.features_df = features_df
                    st.session_state.svm_model   = svm
                    st.session_state.svm_scaler  = scaler
                    st.session_state.svm_fcols   = fcols

                    scenarios = generate_pareto_scenarios(classified)
                    st.session_state.scenarios = scenarios

                    st.success("✅ Analysis complete! Navigate to the pages on the left.")
                except Exception as e:
                    st.error(f"Analysis error: {e}")
                    st.exception(e)

# ══════════════════════════════════════════════
# PAGE 2 — CLASSIFICATION
# ══════════════════════════════════════════════
elif page == "🏷️ Classification":
    st.title("🏷️ BOM Classification")

    if st.session_state.classified is None:
        st.warning("Please upload a BOM and run analysis first.")
        st.stop()

    df = st.session_state.classified

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    valid = df[df['validation_status'] == 'Valid']
    col1.metric("Total Parts",    len(df))
    col2.metric("Valid Parts",    len(valid))
    col3.metric("Invalid Parts",  len(df) - len(valid))
    col4.metric("RF Accuracy",    f"{df['rf_confidence'].mean():.0%}")

    st.divider()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Classification Results")
        display_df = df[['part_no','component_class','rf_confidence',
                         'sourcing_trigger','category','unit_cost_avg',
                         'validation_status']].copy()
        display_df['rf_confidence'] = display_df['rf_confidence'].apply(lambda x: f"{x:.0%}")

        def highlight_class(val):
            colours = {'Commodity':'background-color:#d4edda',
                       'Critical': 'background-color:#fff3cd',
                       'Custom':   'background-color:#cce5ff'}
            return colours.get(val, '')

        st.dataframe(
            display_df.style.map(highlight_class, subset=['component_class']),
            use_container_width=True, height=350
        )

    with col_right:
        st.subheader("Class Distribution")
        class_counts = valid['component_class'].value_counts()
        fig = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            color_discrete_map={
                'Commodity':'#28a745','Critical':'#ffc107','Custom':'#007bff'}
        )
        fig.update_layout(height=300, margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("🌲 Feature Importance — What Drives Classification")
    imp = st.session_state.importance
    fig2 = px.bar(
        imp, x='importance', y='feature', orientation='h',
        color='importance', color_continuous_scale='Greens',
        labels={'importance':'Importance Score','feature':'Feature'}
    )
    fig2.update_layout(height=300, margin=dict(t=20,b=20),
                       yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════
# PAGE 3 — SUPPLIERS
# ══════════════════════════════════════════════
elif page == "🏆 Suppliers":
    st.title("🏆 Supplier Intelligence")

    if st.session_state.metrics_df is None:
        st.warning("Please run analysis first.")
        st.stop()

    metrics = st.session_state.metrics_df

    # Supplier performance table
    st.subheader("Supplier Performance Tiers (K-Means Clustering)")
    display = metrics[['supplier_name','country','otif_rate',
                        'quality_rate','avg_lead_time_days',
                        'order_count','performance_tier','iso_certified']].copy()
    display['otif_rate']         = display['otif_rate'].apply(lambda x: f"{x:.0%}")
    display['quality_rate']      = display['quality_rate'].apply(lambda x: f"{x:.0%}")
    display['avg_lead_time_days']= display['avg_lead_time_days'].apply(lambda x: f"{x:.0f}d")
    display['iso_certified']     = display['iso_certified'].apply(lambda x: "✅" if x else "❌")
    st.dataframe(display, use_container_width=True, height=400)

    st.divider()

    # TOPSIS rankings for selected part
    st.subheader("TOPSIS Supplier Ranking")
    classified = st.session_state.classified
    if classified is not None:
        valid_parts = classified[classified['validation_status']=='Valid']['part_no'].tolist()
        selected    = st.selectbox("Select a part to see supplier ranking:", valid_parts)

        if selected:
            row = classified[classified['part_no']==selected].iloc[0]
            cls = row['component_class']
            st.caption(f"Component class: **{cls}** — weights optimised for {cls}")

            top3 = rank_suppliers_for_part(selected, cls, metrics)
            if not top3.empty:
                cols = ['supplier_name','country','topsis_score',
                        'otif_rate','quality_rate','avg_lead_time_days']
                display3 = top3[cols].copy()
                display3['otif_rate']    = display3['otif_rate'].apply(lambda x: f"{x:.0%}")
                display3['quality_rate'] = display3['quality_rate'].apply(lambda x: f"{x:.0%}")
                display3['avg_lead_time_days'] = display3['avg_lead_time_days'].apply(lambda x: f"{x:.0f}d")
                st.dataframe(display3, use_container_width=True)

                fig = px.bar(
                    top3, x='supplier_name', y='topsis_score',
                    color='topsis_score', color_continuous_scale='Blues',
                    title=f"TOPSIS Scores — {selected} ({cls})"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# PAGE 4 — COST ANALYSIS
# ══════════════════════════════════════════════
elif page == "💰 Cost Analysis":
    st.title("💰 Cost Analysis + ARIMA Forecasting")

    if st.session_state.scenarios is None:
        st.warning("Please run analysis first.")
        st.stop()

    balanced_recs = st.session_state.scenarios['balanced']['recommendations']
    df = pd.DataFrame(balanced_recs)

    # Anomaly summary
    anomalies = df[df['is_price_anomaly'] == True]
    col1, col2, col3 = st.columns(3)
    col1.metric("Total BOM Value",   f"${df['line_value'].sum():,.2f}")
    col2.metric("Price Anomalies",   len(anomalies), delta=f"{len(anomalies)} flagged")
    col3.metric("Avg Landed Cost",   f"${df['landed_cost'].mean():.4f}")

    st.divider()

    st.subheader("Landed Cost Breakdown")
    cost_display = df[['part_no','recommended_supplier','landed_cost',
                        'line_value','is_price_anomaly','anomaly_flag']].copy()
    cost_display['is_price_anomaly'] = cost_display['is_price_anomaly'].apply(
        lambda x: "⚠️ YES" if x else "✅ OK")
    cost_display['line_value']  = cost_display['line_value'].apply(lambda x: f"${x:,.2f}")
    cost_display['landed_cost'] = cost_display['landed_cost'].apply(lambda x: f"${x:.4f}")
    st.dataframe(cost_display, use_container_width=True)

    st.divider()
    st.subheader("BOM Value by Part")
    fig = px.bar(
        df.sort_values('line_value', ascending=True),
        x='line_value', y='part_no', orientation='h',
        color='component_class',
        color_discrete_map={'Commodity':'#28a745','Critical':'#ffc107','Custom':'#007bff'},
        labels={'line_value':'Total Line Value ($)','part_no':'Part Number'}
    )
    fig.update_layout(height=400, margin=dict(t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# PAGE 5 — RISK ASSESSMENT
# ══════════════════════════════════════════════
elif page == "⚠️ Risk Assessment":
    st.title("⚠️ Risk Assessment")

    if st.session_state.scenarios is None:
        st.warning("Please run analysis first.")
        st.stop()

    balanced_recs = st.session_state.scenarios['balanced']['recommendations']
    df = pd.DataFrame(balanced_recs)

    # Risk summary metrics
    high_risk = df[df['risk_score'] >= 60]
    med_risk  = df[(df['risk_score'] >= 30) & (df['risk_score'] < 60)]
    low_risk  = df[df['risk_score'] < 30]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Risk Score", f"{df['risk_score'].mean():.1f}/100")
    col2.metric("🔴 High Risk",   len(high_risk))
    col3.metric("🟡 Medium Risk", len(med_risk))
    col4.metric("🟢 Low Risk",    len(low_risk))

    st.divider()

    # Risk heatmap
    st.subheader("Risk Heatmap")
    fig = px.bar(
        df.sort_values('risk_score', ascending=True),
        x='risk_score', y='part_no', orientation='h',
        color='risk_score',
        color_continuous_scale=[[0,'green'],[0.4,'yellow'],[0.7,'red'],[1,'darkred']],
        range_color=[0, 100],
        labels={'risk_score':'Risk Score (0-100)','part_no':'Part Number'},
        text='risk_score'
    )
    fig.update_layout(height=400, margin=dict(t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Risk flags detail
    st.subheader("Risk Flags Detail")
    for _, row in df.sort_values('risk_score', ascending=False).iterrows():
        icon = risk_colour(row['risk_score'])
        with st.expander(f"{icon} {row['part_no']} — Score: {row['risk_score']}/100 "
                         f"| Supplier: {row['recommended_supplier']}"):
            for flag in row['risk_flags']:
                st.markdown(f"- `{flag}`")

# ══════════════════════════════════════════════
# PAGE 6 — RECOMMENDATIONS
# ══════════════════════════════════════════════
elif page == "✅ Recommendations":
    st.title("✅ Sourcing Recommendations")

    if st.session_state.scenarios is None:
        st.warning("Please run analysis first.")
        st.stop()

    # Scenario selector
    scenario_choice = st.radio(
        "Select scenario:",
        ["balanced", "lowest_cost", "lowest_risk"],
        horizontal=True,
        format_func=lambda x: {
            'balanced':    '⚖️ Balanced (Default)',
            'lowest_cost': '💰 Lowest Cost',
            'lowest_risk': '🛡️ Lowest Risk'
        }[x]
    )

    data = st.session_state.scenarios[scenario_choice]
    recs = data['recommendations']
    df   = pd.DataFrame(recs)

    # Summary
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total BOM Value",  f"${data['total_value']:,.2f}")
    col2.metric("Avg Risk Score",   f"{data['avg_risk']:.1f}/100")
    col3.metric("Parts Sourced",    len(recs))
    col4.metric("Swaps Performed",  len(data['swap_log']))

    st.divider()

    # Pareto comparison
    st.subheader("📊 Scenario Comparison")
    scenarios = st.session_state.scenarios
    comp_df = pd.DataFrame([
        {'Scenario': k,
         'Total Value ($)': v['total_value'],
         'Avg Risk': v['avg_risk']}
        for k, v in scenarios.items()
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.divider()

    # Main recommendation table
    st.subheader(f"Recommendation Table — {scenario_choice.replace('_',' ').title()}")
    for _, row in df.sort_values('risk_score', ascending=False).iterrows():
        icon = risk_colour(row['risk_score'])
        anom = " ⚠️ Price Anomaly" if row.get('is_price_anomaly') else ""
        with st.expander(
            f"{icon} {row['part_no']} → {row['recommended_supplier']} "
            f"| Risk: {row['risk_score']}/100 "
            f"| Landed: ${row['landed_cost']:.4f}{anom}"
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("TOPSIS Score",   f"{row['topsis_score']:.4f}")
            c2.metric("Landed Cost",    f"${row['landed_cost']:.4f}")
            c3.metric("Risk Score",     f"{row['risk_score']}/100")

            st.markdown(f"**Backup Supplier:** {row.get('backup_supplier','N/A')}")
            st.markdown("**Risk Flags:**")
            for flag in row['risk_flags']:
                st.markdown(f"  - `{flag}`")

            if row.get('swap_log'):
                st.markdown("**Rebalancing Actions:**")
                for s in row['swap_log']:
                    st.info(s)

    # Swap log
    if data['swap_log']:
        st.divider()
        st.subheader("📋 Portfolio Rebalancing Log")
        for entry in data['swap_log']:
            st.warning(entry)

    # Export
    st.divider()
    st.subheader("📥 Export")
    export_cols = ['part_no','component_class','recommended_supplier',
                   'topsis_score','landed_cost','line_value',
                   'risk_score','backup_supplier']
    export_df = df[export_cols].copy()
    csv = export_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Recommendations CSV",
        data=csv,
        file_name=f"agentprocure_{scenario_choice}.csv",
        mime='text/csv',
        use_container_width=True
    )