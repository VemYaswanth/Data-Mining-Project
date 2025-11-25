# app_streamlit_v5_2.py
# ----------------------------------------------------------
# Retail Intelligence Dashboard v5.2 (Clean Data Version)
# - Input: Assumes Cleaned CSV
# - Schema: Auto-detects columns silently
# - Tabs:
#     1. Patterns & Model Comparison
#     2. Temporal & Sequential
#     3. Customer Segmentation
#     4. Smart Recommender
#     5. Visual Explanations
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import time
from collections import Counter

import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Retail Intelligence Dashboard v5.2",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Retail Intelligence Dashboard v5.2")
st.caption(
    "Analytics Pipeline: Patterns → Temporal → Segmentation → Recommender.\n"
    "Ready for analysis on pre-cleaned datasets."
)

# =========================================================
# Helper Functions — Detection & Mapping
# =========================================================

def normalize_products(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()

def detect_product_column(df: pd.DataFrame):
    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if not obj_cols: return None
    scores = []
    n = len(df)
    for col in obj_cols:
        nun = df[col].nunique(dropna=True)
        if nun <= 5 or nun >= 0.95 * n: continue
        scores.append((col, nun / n))
    if not scores: return None
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0]

def detect_invoice_column(df: pd.DataFrame, product_col: str | None):
    candidates = []
    n = len(df)
    for col in df.columns:
        if col == product_col: continue
        nun = df[col].nunique(dropna=True)
        if nun <= 1 or nun >= n: continue
        avg_group = n / nun if nun > 0 else 0
        if avg_group < 1.5: continue
        candidates.append((col, avg_group))
    if not candidates: return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

def detect_customer_column(df: pd.DataFrame, used_cols: set):
    n = len(df)
    best = None
    best_score = -1
    for col in df.columns:
        if col in used_cols: continue
        nun = df[col].nunique(dropna=True)
        if nun <= 5: continue
        ratio = nun / n
        if 0.05 <= ratio <= 0.95 and ratio > best_score:
            best = col
            best_score = ratio
    return best

def detect_date_column(df: pd.DataFrame):
    best = None
    best_rate = 0
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            ok_rate = parsed.notna().mean()
            if ok_rate > 0.5 and ok_rate > best_rate:
                best = col
                best_rate = ok_rate
        except Exception:
            continue
    return best

def detect_amount_column(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols: return None
    preferred = []
    for col in num_cols:
        cname = col.lower()
        if any(key in cname for key in ["final", "total", "amount", "sales", "revenue", "net"]):
            preferred.append(col)
    if preferred: return preferred[0]
    variances = []
    for col in num_cols:
        variances.append((col, df[col].var()))
    variances.sort(key=lambda x: (x[1] if pd.notna(x[1]) else 0), reverse=True)
    return variances[0][0] if variances else None

def map_columns(df: pd.DataFrame, schema: dict):
    """
    Ensures the dataframe has the necessary mapped columns.
    If the clean dataset is missing a column (e.g. no Customer ID),
    it generates a placeholder so the code doesn't crash, but quietly.
    """
    work = df.copy()
    n = len(work)

    # Product
    if schema["product"] is None:
        work["Product_ID"] = "ITEM_" + work.index.astype(str)
        schema["product"] = "Product_ID"
    else:
        work[schema["product"]] = normalize_products(work[schema["product"]])

    # Customer
    if schema["customer"] is None:
        work["Customer_ID"] = "CUST_GEN"
        schema["customer"] = "Customer_ID"

    # Date
    if schema["date"] is None:
        work["Date_Mapped"] = pd.Timestamp("2025-01-01")
        schema["date"] = "Date_Mapped"
    else:
        work[schema["date"]] = pd.to_datetime(work[schema["date"]], errors="coerce")
        # If parsing failed completely, fallback
        if work[schema["date"]].notna().sum() == 0:
             work["Date_Mapped"] = pd.Timestamp("2025-01-01")
             schema["date"] = "Date_Mapped"

    # Amount
    if schema["amount"] is None:
        work["Amount_Mapped"] = 1.0
        schema["amount"] = "Amount_Mapped"

    # Invoice
    if schema["invoice"] is None:
        # Fallback grouping if no invoice ID exists
        work["Invoice_Mapped"] = work.index // 3 
        schema["invoice"] = "Invoice_Mapped"

    return work, schema

def build_transactions(df: pd.DataFrame, schema: dict):
    prod = schema["product"]
    inv = schema["invoice"]
    grouped = df.groupby(inv)[prod].apply(lambda s: sorted(set(s.tolist())))
    transactions = [t for t in grouped.tolist() if len(t) > 0]
    return transactions

def encode_transactions(transactions):
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_array, columns=te.columns_)

def run_mining(df_encoded: pd.DataFrame, algo: str, min_support: float, max_len: int):
    t0 = time.time()
    if algo == "Apriori":
        fi = apriori(df_encoded, min_support=min_support, use_colnames=True, max_len=max_len)
    else:
        fi = fpgrowth(df_encoded, min_support=min_support, use_colnames=True, max_len=max_len)
    elapsed = time.time() - t0
    fi = fi.sort_values("support", ascending=False)
    return fi, elapsed

def make_rules(fi: pd.DataFrame, metric: str, min_threshold: float):
    if fi is None or fi.empty: return pd.DataFrame()
    rules = association_rules(fi, metric=metric, min_threshold=min_threshold)
    rules["antecedents_str"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules["consequents_str"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
    return rules.sort_values(["lift", "confidence", "support"], ascending=False)

def plot_top_itemsets(fi: pd.DataFrame, n=20):
    if fi.empty: return None
    tmp = fi.head(n).copy()
    tmp["itemset_str"] = tmp["itemsets"].apply(lambda s: ", ".join(sorted(list(s))))
    return px.bar(tmp[::-1], x="support", y="itemset_str", orientation="h", title=f"Top {n} Frequent Itemsets")

def plot_rules_scatter(rules: pd.DataFrame):
    if rules.empty: return None
    return px.scatter(
        rules, x="support", y="confidence", size="lift",
        hover_data=["antecedents_str", "consequents_str"],
        title="Rules: Support vs Confidence (size = Lift)"
    )

def build_association_network(rules: pd.DataFrame, topn=30):
    if rules.empty: return None
    sub = rules.head(topn).copy()
    G = nx.DiGraph()
    for _, r in sub.iterrows():
        for a in r["antecedents"]:
            for c in r["consequents"]:
                G.add_edge(a, c, lift=r["lift"], confidence=r["confidence"])
    
    pos = nx.spring_layout(G, seed=42, k=0.8)
    edge_x, edge_y = [], []
    for (u, v) in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    
    node_x, node_y, labels = [], [], []
    for n, (x, y) in pos.items():
        node_x.append(x)
        node_y.append(y)
        labels.append(n)
        
    fig = go.Figure(data=[
        go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none", line=dict(width=1)),
        go.Scatter(x=node_x, y=node_y, mode="markers+text", text=labels, textposition="top center")
    ])
    fig.update_layout(title="Association Network", showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def build_rfm(df: pd.DataFrame, schema: dict):
    cust = schema["customer"]
    date = schema["date"]
    amt = schema["amount"]
    if cust not in df.columns or date not in df.columns or amt not in df.columns:
        return pd.DataFrame()
        
    work = df.dropna(subset=[cust, date])
    if work.empty: return pd.DataFrame()
    
    now = work[date].max()
    grp = work.groupby(cust).agg(
        Recency=(date, lambda s: (now - s.max()).days),
        Frequency=(date, "count"),
        Monetary=(amt, "sum")
    ).reset_index().rename(columns={cust: "customerid"})
    
    # Simple Binning
    try:
        grp["R_Score"] = pd.qcut(grp["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1])
        grp["F_Score"] = pd.qcut(grp["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        grp["M_Score"] = pd.qcut(grp["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    except:
        # Fallback if qcut fails due to low cardinality
        grp["R_Score"] = pd.cut(grp["Recency"], bins=5, labels=[5, 4, 3, 2, 1], include_lowest=True)
        grp["F_Score"] = pd.cut(grp["Frequency"], bins=5, labels=[1, 2, 3, 4, 5], include_lowest=True)
        grp["M_Score"] = pd.cut(grp["Monetary"], bins=5, labels=[1, 2, 3, 4, 5], include_lowest=True)

    grp["RFM_Score"] = grp["R_Score"].astype(int) + grp["F_Score"].astype(int) + grp["M_Score"].astype(int)
    grp["Segment"] = pd.cut(grp["RFM_Score"], bins=[0, 6, 9, 12, 15], labels=["Low", "Mid", "High", "VIP"], include_lowest=True)
    return grp

def kmeans_clusters(df_rfm: pd.DataFrame, k=3):
    try:
        feats = df_rfm[["Recency", "Frequency", "Monetary"]].fillna(0).copy()
        scaler = StandardScaler()
        X = scaler.fit_transform(feats)
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        df_rfm["KMeansCluster"] = labels
        centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=["Recency", "Frequency", "Monetary"])
        centers["Cluster"] = centers.index
        return df_rfm, centers
    except:
        return df_rfm, pd.DataFrame()

def rules_to_recommendations(rules: pd.DataFrame, base_items, enc: pd.DataFrame):
    if rules.empty or not base_items: return pd.DataFrame()
    try:
        item_support = enc.mean().sort_values(ascending=False)
        top_staples = set(item_support.head(3).index)
    except:
        top_staples = set()
    
    sub = rules[rules["antecedents"].apply(lambda s: set(base_items).issubset(s))]
    if sub.empty: return pd.DataFrame()
    
    sub_filtered = sub.copy()
    sub_filtered["consequents"] = sub_filtered["consequents"].apply(lambda s: {x for x in s if x not in top_staples})
    sub_filtered = sub_filtered[sub_filtered["consequents"].apply(lambda s: len(s) > 0)]
    
    if sub_filtered.empty: return pd.DataFrame()
    
    recs = (
        sub_filtered.explode("consequents")
        .groupby("consequents")
        .agg(
            mean_conf=("confidence", "mean"),
            mean_lift=("lift", "mean"),
            count=("confidence", "size")
        )
        .sort_values(["mean_lift", "mean_conf"], ascending=False)
        .head(10)
    )
    recs.index = recs.index.astype(str)
    return recs

# =========================================================
# 1. Upload Dataset (Cleaned)
# =========================================================

st.sidebar.header("Data Source")
uploaded = st.sidebar.file_uploader("Upload Cleaned CSV", type=["csv"])

if uploaded is None:
    st.info("👋 Welcome! Please upload your **cleaned** CSV dataset to begin analysis.")
    st.stop()

# Read CSV
df_raw = pd.read_csv(uploaded)

# =========================================================
# 2. Silent Schema Mapping
# =========================================================
# We still detect columns to be robust against different naming conventions,
# but we do not show the verbose output to the user.

product_col = detect_product_column(df_raw)
invoice_col = detect_invoice_column(df_raw, product_col)
customer_col = detect_customer_column(df_raw, {product_col, invoice_col})
date_col = detect_date_column(df_raw)
amount_col = detect_amount_column(df_raw)

initial_schema = {
    "product": product_col,
    "invoice": invoice_col,
    "customer": customer_col,
    "date": date_col,
    "amount": amount_col,
}

df_work, schema = map_columns(df_raw, initial_schema)

with st.expander("View Dataset Details & Mapped Schema"):
    st.write(f"Rows: **{len(df_work):,}** | Columns: **{len(df_work.columns):,}**")
    st.dataframe(df_work.head())
    st.json(schema)

# =========================================================
# Tabs Layout (Cleaning Tab Removed)
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Patterns",
        "🕒 Temporal",
        "👥 Segmentation",
        "🛒 Recommender",
        "📈 Pipeline",
    ]
)

# =========================================================
# TAB 1: Frequent Patterns
# =========================================================

with tab1:
    st.subheader("Frequent Patterns & Market Basket Analysis")
    
    transactions = build_transactions(df_work, schema)
    if not transactions:
        st.warning("Could not build transactions. Check your Invoice/Product columns.")
        st.stop()
        
    df_enc = encode_transactions(transactions)
    
    colA, colB = st.columns(2)
    min_support = colA.slider("Min Support", 0.001, 0.1, 0.01, step=0.001)
    max_len = colB.slider("Max Itemset Length", 2, 5, 3)
    
    # Compare Models
    fi_fp, t_fp = run_mining(df_enc, "FP-Growth", min_support, max_len)
    fi_ap, t_ap = run_mining(df_enc, "Apriori", min_support, max_len)
    
    st.markdown(f"**FP-Growth found {len(fi_fp)} itemsets** in {t_fp:.4f}s vs **Apriori** in {t_ap:.4f}s.")

    # Rules
    metric = st.selectbox("Rule Metric", ["confidence", "lift"])
    min_metric = st.slider(f"Min {metric.title()}", 0.1, 1.0, 0.3, step=0.05)
    
    rules = make_rules(fi_fp, metric, min_metric)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        if not fi_fp.empty:
            st.plotly_chart(plot_top_itemsets(fi_fp, n=15), use_container_width=True)
    with col2:
        if not rules.empty:
            st.plotly_chart(plot_rules_scatter(rules), use_container_width=True)

    if not rules.empty:
        st.markdown("### Association Network")
        st.plotly_chart(build_association_network(rules, topn=30), use_container_width=True)
        
    st.session_state["df_enc"] = df_enc
    st.session_state["rules"] = rules

# =========================================================
# TAB 2: Temporal
# =========================================================

with tab2:
    st.subheader("Temporal & Sequential Analysis")
    d_col = schema["date"]
    
    if d_col in df_work.columns:
        df_t = df_work.dropna(subset=[d_col]).copy()
        
        # Monthly
        monthly = df_t.groupby(df_t[d_col].dt.to_period("M")).size().reset_index(name="Count")
        monthly[d_col] = monthly[d_col].astype(str)
        st.line_chart(monthly.set_index(d_col))
        
        # Weekday
        weekday = df_t.groupby(df_t[d_col].dt.day_name()).size().reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        ).reset_index(name="Count")
        st.bar_chart(weekday.set_index(d_col))
    else:
        st.warning("No valid Date column found for temporal analysis.")

# =========================================================
# TAB 3: Segmentation
# =========================================================

with tab3:
    st.subheader("Customer Segmentation (RFM + K-Means)")
    
    rfm = build_rfm(df_work, schema)
    
    if rfm.empty:
        st.warning("RFM Analysis requires valid Customer, Date, and Amount columns.")
    else:
        k = st.slider("Clusters (K)", 2, 6, 3)
        rfm_km, centers = kmeans_clusters(rfm, k)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.scatter(
                rfm_km, x="Frequency", y="Monetary", color=rfm_km["KMeansCluster"].astype(str),
                hover_data=["customerid", "RFM_Score", "Segment"],
                title="Segments: Frequency vs Monetary"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.markdown("**Cluster Centers**")
            st.dataframe(centers, use_container_width=True)
            
        st.markdown("**Segment Distribution**")
        st.bar_chart(rfm_km["Segment"].value_counts())

# =========================================================
# TAB 4: Recommender
# =========================================================

with tab4:
    st.subheader("Smart Recommender")
    rules = st.session_state.get("rules", pd.DataFrame())
    df_enc = st.session_state.get("df_enc", pd.DataFrame())
    
    if rules.empty:
        st.warning("Please run Tab 1 (Patterns) first to generate rules.")
    else:
        all_items = sorted(set().union(*rules["antecedents"]).union(*rules["consequents"]))
        selected = st.multiselect("Select Basket Items", all_items)
        
        if selected:
            recs = rules_to_recommendations(rules, selected, df_enc)
            if recs.empty:
                st.info("No recommendations found (try lowering confidence/lift in Tab 1).")
            else:
                st.markdown("### Recommended Add-ons")
                st.dataframe(recs, use_container_width=True)

# =========================================================
# TAB 5: Visuals
# =========================================================

with tab5:
    st.subheader("Project Pipeline")
    
    labels = ["Cleaned Data", "Transactions", "Frequent Itemsets", "Association Rules", "Recommendations"]
    source = [0, 1, 2, 3]
    target = [1, 2, 3, 4]
    value = [10, 10, 10, 10]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels),
        link=dict(source=source, target=target, value=value)
    )])
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### Methodology
    1. **Data Ingestion**: Takes pre-cleaned CSV data.
    2. **Pattern Mining**: Uses FP-Growth to find items that co-occur.
    3. **Rule Generation**: Calculates Confidence (probability) and Lift (strength).
    4. **Recommendation**: Filters rules based on active user selection.
    """)
