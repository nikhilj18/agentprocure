# streamlit_app/app_auth.py
# Run: streamlit run streamlit_app/app_auth.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import yaml
import bcrypt

st.set_page_config(
    page_title="AgentProcure",
    page_icon="🔧",
    layout="wide"
)

# ── Load credentials ──────────────────────────────────────
cfg_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'auth_config.yaml'
)
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

users = cfg['credentials']['usernames']

# ── Session state init ────────────────────────────────────
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username  = None
    st.session_state.role      = None
    st.session_state.name      = None

# ── LOGIN SCREEN ──────────────────────────────────────────
if not st.session_state.logged_in:
    st.markdown("## 🔧 AgentProcure")
    st.markdown("#### AI-Enabled Procurement Intelligence")
    st.divider()

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("### Sign In")
        uname = st.text_input("Username", placeholder="e.g. admin")
        passw = st.text_input("Password", type="password")

        if st.button("Sign In", type="primary", use_container_width=True):
            if uname in users:
                stored_hash = users[uname]['password'].encode()
                if bcrypt.checkpw(passw.encode(), stored_hash):
                    st.session_state.logged_in = True
                    st.session_state.username  = uname
                    st.session_state.role      = users[uname]['role']
                    st.session_state.name      = users[uname]['name']
                    st.rerun()
                else:
                    st.error("Incorrect password.")
            else:
                st.error("Username not found.")

        st.divider()
        st.markdown("**Demo accounts:**")
        st.markdown("- `admin` / `admin123` — Full access")
        st.markdown("- `buyer` / `buyer123` — View + Approve")
        st.markdown("- `viewer` / `viewer123` — Read only")

    st.stop()

# ── AUTHENTICATED ─────────────────────────────────────────
role      = st.session_state.role
user_name = st.session_state.name
username  = st.session_state.username

CAN_RUN_ANALYSIS = role in ('admin', 'buyer')
CAN_EDIT_AVL     = role in ('admin', 'buyer')
CAN_SUSPEND_AVL  = role == 'admin'
IS_VIEWER        = role == 'viewer'

# ── SIDEBAR ───────────────────────────────────────────────
st.sidebar.title("🔧 AgentProcure")
st.sidebar.caption("AI-Enabled Procurement Intelligence")
st.sidebar.divider()

role_badge = {'admin':'🔴 Admin','buyer':'🟡 Buyer','viewer':'🟢 Viewer'}.get(role,'⚪')
st.sidebar.markdown(f"**{user_name}**")
st.sidebar.caption(f"Role: {role_badge}")

if st.sidebar.button("Sign Out"):
    st.session_state.logged_in = False
    st.session_state.username  = None
    st.session_state.role      = None
    st.session_state.name      = None
    st.rerun()

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
st.sidebar.caption("Random Forest · TOPSIS · ARIMA · SVM · Claude API")

if IS_VIEWER:
    st.info("👁️ Read-only mode — contact Admin or Buyer to run analysis.")

# ── SESSION STATE FOR ANALYSIS ────────────────────────────
for k in ['bom_df','classified','importance','scenarios',
          'metrics_df','features_df','svm_model','svm_scaler','svm_fcols']:
    if k not in st.session_state:
        st.session_state[k] = None

def risk_colour(s):
    return "🔴" if s>=60 else "🟡" if s>=30 else "🟢"

# ── IMPORT MODULES ONCE LOGGED IN ────────────────────────
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# PAGE 1 — UPLOAD BOM
# ══════════════════════════════════════════════════════════
if page == "📤 Upload BOM":
    st.title("📤 Upload Bill of Materials")
    col1, col2 = st.columns([2,1])
    with col1:
        up = st.file_uploader("Choose BOM CSV", type=['csv'])
        if up:
            try:
                df = pd.read_csv(up)
                if not {'part_no','quantity_required'}.issubset(df.columns):
                    st.error("CSV must have: part_no, quantity_required")
                else:
                    st.session_state.bom_df = df
                    st.success(f"✅ {len(df)} parts loaded")
                    st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(str(e))
    with col2:
        st.markdown("### Sample BOM")
        if st.button("📋 Load Sample", use_container_width=True):
            st.session_state.bom_df = pd.DataFrame([
                {'part_no':'RES-0402-10K',  'quantity_required':5000},
                {'part_no':'CAP-0402-100N', 'quantity_required':8000},
                {'part_no':'IGBT-G4PC50W',  'quantity_required':200},
                {'part_no':'IC-MCU-STM32',  'quantity_required':300},
                {'part_no':'ASM-XFMR-LVCT','quantity_required':100},
                {'part_no':'ASM-INV-1KW',   'quantity_required':50},
                {'part_no':'MAG-CORE-ETD39','quantity_required':500},
            ])
            st.success("✅ Sample loaded!")
            st.dataframe(st.session_state.bom_df, use_container_width=True)

    if st.session_state.bom_df is not None:
        st.divider()
        if CAN_RUN_ANALYSIS:
            if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True):
                from modules.module1_classifier import classify_bom
                from modules.module2_supplier   import build_supplier_metrics, cluster_suppliers
                from modules.module4_risk       import build_supplier_features, train_svm_classifier
                from modules.module5_optimizer  import generate_pareto_scenarios
                with st.spinner("Running all modules..."):
                    try:
                        c, imp, acc = classify_bom(st.session_state.bom_df)
                        st.session_state.classified = c
                        st.session_state.importance = imp
                        m, _, _ = cluster_suppliers(build_supplier_metrics())
                        st.session_state.metrics_df = m
                        fd = build_supplier_features()
                        sv, sc, _, fc = train_svm_classifier(fd)
                        st.session_state.features_df = fd
                        st.session_state.svm_model   = sv
                        st.session_state.svm_scaler  = sc
                        st.session_state.svm_fcols   = fc
                        st.session_state.scenarios   = generate_pareto_scenarios(c)
                        st.success("✅ Analysis complete!")
                    except Exception as e:
                        st.error(str(e))
                        st.exception(e)
        else:
            st.warning("👁️ Read-only — ask Admin/Buyer to run analysis.")

# ══════════════════════════════════════════════════════════
# PAGE 2 — CLASSIFICATION
# ══════════════════════════════════════════════════════════
elif page == "🏷️ Classification":
    st.title("🏷️ BOM Classification")
    if st.session_state.classified is None:
        st.warning("Run analysis first.")
        st.stop()
    df    = st.session_state.classified
    valid = df[df['validation_status']=='Valid']
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Parts",   len(df))
    c2.metric("Valid Parts",   len(valid))
    c3.metric("Invalid Parts", len(df)-len(valid))
    c4.metric("RF Accuracy",   f"{df['rf_confidence'].mean():.0%}")
    st.divider()
    cl, cr = st.columns([3,2])
    with cl:
        st.subheader("Classification Results")
        d = df[['part_no','component_class','rf_confidence',
                'sourcing_trigger','category','unit_cost_avg']].copy()
        d['rf_confidence'] = d['rf_confidence'].apply(lambda x: f"{x:.0%}")
        def hl(v):
            return {'Commodity':'background-color:#d4edda',
                    'Critical':'background-color:#fff3cd',
                    'Custom':'background-color:#cce5ff'}.get(v,'')
        st.dataframe(d.style.map(hl, subset=['component_class']),
                     use_container_width=True, height=350)
    with cr:
        st.subheader("Class Distribution")
        cc  = valid['component_class'].value_counts()
        fig = px.pie(values=cc.values, names=cc.index,
                     color_discrete_map={'Commodity':'#28a745','Critical':'#ffc107','Custom':'#007bff'})
        fig.update_layout(height=300, margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.subheader("Feature Importance")
    imp  = st.session_state.importance
    fig2 = px.bar(imp, x='importance', y='feature', orientation='h',
                  color='importance', color_continuous_scale='Greens')
    fig2.update_layout(height=300, yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 3 — SUPPLIERS
# ══════════════════════════════════════════════════════════
elif page == "🏆 Suppliers":
    st.title("🏆 Supplier Intelligence")
    if st.session_state.metrics_df is None:
        st.warning("Run analysis first.")
        st.stop()
    from modules.module2_supplier import rank_suppliers_for_part
    m = st.session_state.metrics_df
    st.subheader("Supplier Tiers + ESG Scores")
    d = m[['supplier_name','country','otif_rate','quality_rate',
           'avg_lead_time_days','esg_score','performance_tier','iso_certified']].copy()
    d['otif_rate']    = d['otif_rate'].apply(lambda x: f"{x:.0%}")
    d['quality_rate'] = d['quality_rate'].apply(lambda x: f"{x:.0%}")
    d['avg_lead_time_days'] = d['avg_lead_time_days'].apply(lambda x: f"{x:.0f}d")
    d['iso_certified'] = d['iso_certified'].apply(lambda x: "✅" if x else "❌")
    st.dataframe(d, use_container_width=True, height=400)
    st.divider()
    st.subheader("TOPSIS Ranking")
    cl = st.session_state.classified
    if cl is not None:
        parts = cl[cl['validation_status']=='Valid']['part_no'].tolist()
        sel   = st.selectbox("Select part:", parts)
        if sel:
            row = cl[cl['part_no']==sel].iloc[0]
            cls = row['component_class']
            top3 = rank_suppliers_for_part(sel, cls, m)
            if not top3.empty:
                d3 = top3[['supplier_name','country','topsis_score',
                            'otif_rate','quality_rate','esg_score']].copy()
                d3['otif_rate']    = d3['otif_rate'].apply(lambda x: f"{x:.0%}")
                d3['quality_rate'] = d3['quality_rate'].apply(lambda x: f"{x:.0%}")
                st.dataframe(d3, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 4 — COST ANALYSIS
# ══════════════════════════════════════════════════════════
elif page == "💰 Cost Analysis":
    st.title("💰 Cost Analysis + ARIMA + FRED")
    if st.session_state.scenarios is None:
        st.warning("Run analysis first.")
        st.stop()
    df = pd.DataFrame(st.session_state.scenarios['balanced']['recommendations'])
    an = df[df['is_price_anomaly']==True]
    c1,c2,c3 = st.columns(3)
    c1.metric("Total BOM Value", f"${df['line_value'].sum():,.2f}")
    c2.metric("Price Anomalies", len(an))
    c3.metric("Avg Landed Cost", f"${df['landed_cost'].mean():.4f}")
    st.divider()
    cd = df[['part_no','recommended_supplier','landed_cost',
             'line_value','is_price_anomaly']].copy()
    if 'fred_calibrated' in df.columns:
        cd.insert(2, 'fred_calibrated',
                  df['fred_calibrated'].apply(lambda x: "✅ FRED" if x else "📊 PO"))
    cd['is_price_anomaly'] = cd['is_price_anomaly'].apply(lambda x: "⚠️" if x else "✅")
    cd['line_value']       = cd['line_value'].apply(lambda x: f"${x:,.2f}")
    cd['landed_cost']      = cd['landed_cost'].apply(lambda x: f"${x:.4f}")
    st.dataframe(cd, use_container_width=True)
    fig = px.bar(df.sort_values('line_value'), x='line_value', y='part_no',
                 orientation='h', color='component_class',
                 color_discrete_map={'Commodity':'#28a745','Critical':'#ffc107','Custom':'#007bff'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 5 — RISK ASSESSMENT
# ══════════════════════════════════════════════════════════
elif page == "⚠️ Risk Assessment":
    st.title("⚠️ Risk Assessment")
    if st.session_state.scenarios is None:
        st.warning("Run analysis first.")
        st.stop()
    df = pd.DataFrame(st.session_state.scenarios['balanced']['recommendations'])
    hi = df[df['risk_score']>=60]
    md = df[(df['risk_score']>=30)&(df['risk_score']<60)]
    lo = df[df['risk_score']<30]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Avg Risk", f"{df['risk_score'].mean():.1f}/100")
    c2.metric("🔴 High",  len(hi))
    c3.metric("🟡 Medium",len(md))
    c4.metric("🟢 Low",   len(lo))
    st.divider()
    fig = px.bar(df.sort_values('risk_score'), x='risk_score', y='part_no',
                 orientation='h', color='risk_score', text='risk_score',
                 color_continuous_scale=[[0,'green'],[0.4,'yellow'],[0.7,'red'],[1,'darkred']],
                 range_color=[0,100])
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    for _, row in df.sort_values('risk_score', ascending=False).iterrows():
        with st.expander(f"{risk_colour(row['risk_score'])} {row['part_no']} — "
                         f"{row['risk_score']}/100 | {row['recommended_supplier']}"):
            for flag in row['risk_flags']:
                st.markdown(f"- `{flag}`")

# ══════════════════════════════════════════════════════════
# PAGE 6 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════
elif page == "✅ Recommendations":
    st.title("✅ Sourcing Recommendations")
    if st.session_state.scenarios is None:
        st.warning("Run analysis first.")
        st.stop()
    sc   = st.radio("Scenario:", ["balanced","lowest_cost","lowest_risk"],
                    horizontal=True,
                    format_func=lambda x: {'balanced':'⚖️ Balanced',
                                           'lowest_cost':'💰 Cost','lowest_risk':'🛡️ Risk'}[x])
    data = st.session_state.scenarios[sc]
    df   = pd.DataFrame(data['recommendations'])
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("BOM Value",  f"${data['total_value']:,.2f}")
    c2.metric("Avg Risk",   f"{data['avg_risk']:.1f}/100")
    c3.metric("Parts",      len(df))
    c4.metric("Swaps",      len(data['swap_log']))
    st.divider()
    comp = pd.DataFrame([{'Scenario':k,'Value':f"${v['total_value']:,.2f}",
                          'Risk':f"{v['avg_risk']:.1f}"} for k,v in st.session_state.scenarios.items()])
    st.dataframe(comp, use_container_width=True, hide_index=True)
    st.divider()
    for _, row in df.sort_values('risk_score', ascending=False).iterrows():
        anom = " ⚠️" if row.get('is_price_anomaly') else ""
        with st.expander(f"{risk_colour(row['risk_score'])} {row['part_no']} → "
                         f"{row['recommended_supplier']} | {row['risk_score']}/100{anom}"):
            a,b,c = st.columns(3)
            a.metric("TOPSIS",   f"{row['topsis_score']:.4f}")
            b.metric("Landed $", f"${row['landed_cost']:.4f}")
            c.metric("Risk",     f"{row['risk_score']}/100")
            for flag in row['risk_flags']:
                st.markdown(f"- `{flag}`")
    if data['swap_log']:
        st.divider()
        for s in data['swap_log']:
            st.warning(s)
    csv = df[['part_no','component_class','recommended_supplier',
              'topsis_score','landed_cost','line_value','risk_score']].to_csv(index=False)
    st.download_button("⬇️ Download CSV", csv, f"recs_{sc}.csv",
                       "text/csv", use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 7 — AVL MANAGER
# ══════════════════════════════════════════════════════════
elif page == "📋 AVL Manager":
    from database.db_connect import run_query, run_insert
    from datetime import date

    st.title("📋 AVL Manager")
    if IS_VIEWER:
        st.info("👁️ Read-only mode.")

    @st.cache_data(ttl=30)
    def load_avl():
        return run_query("""
            SELECT a.avl_id, a.part_no, c.description, c.component_class,
                   s.supplier_name, s.country, a.qualification_status,
                   a.qualified_date, a.supplier_id
            FROM approved_vendor_list a
            JOIN component_master c ON a.part_no = c.part_no
            JOIN supplier_master  s ON a.supplier_id = s.supplier_id
            ORDER BY a.part_no""")

    @st.cache_data(ttl=60)
    def load_pts():
        return run_query("SELECT part_no, description, component_class FROM component_master ORDER BY part_no")

    @st.cache_data(ttl=60)
    def load_sups():
        return run_query("SELECT supplier_id, supplier_name, country FROM supplier_master WHERE status='Active' ORDER BY supplier_name")

    avl = load_avl()
    pts = load_pts()
    sup = load_sups()
    q   = avl[avl['qualification_status']=='Qualified']
    s   = avl[avl['qualification_status']=='Suspended']
    psc = q.groupby('part_no')['supplier_id'].count()
    ss  = psc[psc==1]
    cp  = pts[pts['component_class']=='Critical']['part_no']
    cs  = ss[ss.index.isin(cp)]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total",        len(avl))
    c2.metric("Qualified",    len(q))
    c3.metric("Suspended",    len(s))
    c4.metric("Single-Source",len(ss),
              delta=f"{len(cs)} critical" if len(cs)>0 else None,
              delta_color="inverse")
    if len(cs)>0:
        st.warning(f"⚠️ Critical single-sourced: {', '.join(cs.index.tolist()[:5])}")
    st.divider()

    c1,c2,c3 = st.columns(3)
    fc = c1.selectbox("Class",  ["All","Commodity","Critical","Custom"])
    fs = c2.selectbox("Status", ["All","Qualified","Suspended"])
    fq = c3.text_input("Search")
    d  = avl.copy()
    if fc!="All": d=d[d['component_class']==fc]
    if fs!="All": d=d[d['qualification_status']==fs]
    if fq:        d=d[d['part_no'].str.contains(fq,case=False)]
    st.dataframe(d[['part_no','description','component_class',
                    'supplier_name','country','qualification_status',
                    'qualified_date']], use_container_width=True, height=300)

    if CAN_EDIT_AVL:
        st.divider()
        st.subheader("➕ Add Qualification")
        with st.form("add"):
            a1,a2 = st.columns(2)
            sp    = a1.selectbox("Component", pts['part_no'].tolist())
            cur   = q[q['part_no']==sp]['supplier_name'].tolist()
            if cur: a1.info(f"Qualified: {', '.join(cur)}")
            else:   a1.warning("No suppliers — single-source risk!")
            done  = avl[(avl['part_no']==sp)&(avl['qualification_status']=='Qualified')]['supplier_id'].tolist()
            av2   = sup[~sup['supplier_id'].isin(done)]
            if av2.empty:
                a2.info("All qualified.")
                nsid = None
            else:
                opts = av2.apply(lambda r: f"{r['supplier_name']} ({r['country']})", axis=1).tolist()
                sel2 = a2.selectbox("New Supplier", opts)
                nsid = av2.iloc[opts.index(sel2)]['supplier_id']
            qd = a2.date_input("Date", value=date.today())
            if st.form_submit_button("✅ Add", type="primary", use_container_width=True) and nsid:
                try:
                    run_insert("INSERT INTO approved_vendor_list (part_no,supplier_id,qualification_status,qualified_date) VALUES (%s,%s,%s,%s) ON CONFLICT (part_no,supplier_id) DO UPDATE SET qualification_status='Qualified',qualified_date=%s",
                               (sp,int(nsid),'Qualified',qd,qd))
                    st.success("✅ Added!")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    if CAN_SUSPEND_AVL:
        st.divider()
        st.subheader("✏️ Update / Suspend")
        with st.form("upd"):
            u1,u2,u3 = st.columns(3)
            eo   = avl.apply(lambda r: f"{r['part_no']} — {r['supplier_name']}", axis=1).tolist()
            es   = u1.selectbox("Entry", eo)
            ei   = eo.index(es)
            eid  = avl.iloc[ei]['avl_id']
            ec   = avl.iloc[ei]['qualification_status']
            so   = ["Qualified","Suspended","Under Review"]
            ns   = u2.selectbox("Status", so, index=so.index(ec) if ec in so else 0)
            nd   = u3.date_input("Date", value=date.today())
            if st.form_submit_button("💾 Update", use_container_width=True):
                try:
                    run_insert("UPDATE approved_vendor_list SET qualification_status=%s,qualified_date=%s WHERE avl_id=%s",
                               (ns,nd,int(eid)))
                    st.success(f"Updated to {ns}")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    if len(ss)>0:
        st.divider()
        st.subheader("⚠️ Single-Source Risk")
        sdf = q[q['part_no'].isin(ss.index)][['part_no','component_class','supplier_name','country']].copy()
        sdf['risk'] = sdf['component_class'].map({'Critical':'🔴','Custom':'🟡','Commodity':'🟢'})
        st.dataframe(sdf, use_container_width=True, hide_index=True)