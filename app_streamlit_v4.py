# -------------------------------------------------------------
# Retail Intelligence Dashboard v6 (Streamlit Final Version)
# Fully stable, Cloud-ready, supports ALL datasets
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import time
from collections import Counter

import plotly.express as px
import plotly.graph_objects as go
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import networkx as nx

# ============================================================
# Streamlit configuration
# ============================================================
st.set_page_config(page_title="Retail Intelligence Dashboard v6", layout="wide")
st.title("🧠 Retail Intelligence Dashboard v6")
st.caption("Auto-schema → Cleaning → FP-Growth → Rules → RFM → Recommender → Visual Pipeline")

# ============================================================
# Helper Functions
# ============================================================

def normalize_products(s):
    return s.astype(str).str.upper().str.strip()


def detect_product_column(df):
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    if not len(obj_cols): return None
    scores = []
    n = len(df)
    for col in obj_cols:
        nun = df[col].nunique()
        if 5 < nun < n * 0.95:
            scores.append((col, nun / n))
    if not scores:
        return None
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0]


def detect_invoice_column(df, product_col):
    n = len(df)
    candidates = []
    for col in df.columns:
        if col == product_col: continue
        nun = df[col].nunique()
        if 1 < nun < n:
            avg = n / nun
            if avg >= 1.5:
                candidates.append((col, avg))
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]


def detect_customer_column(df, used):
    n = len(df)
    best = None
    score = -1
    for col in df.columns:
        if col in used: continue
        nun = df[col].nunique()
        r = nun / n
        if 0.05 <= r <= 0.95 and r > score:
            best = col
            score = r
    return best


def detect_date_column(df):
    best = None
    best_rate = 0
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            rate = parsed.notna().mean()
            if rate > 0.5 and rate > best_rate:
                best = col
                best_rate = rate
        except:
            continue
    return best


def detect_amount_column(df):
    nums = df.select_dtypes(include=["number"]).columns
    if not len(nums):
        return None
    preferred = [c for c in nums if any(k in c.lower() for k in ["amount", "total", "revenue"])]
    if preferred:
        return preferred[0]
    return nums[0]


# ------------------------------------------------------------
# Synthetic realistic invoices (basket size 2–7)
# ------------------------------------------------------------
def create_realistic_invoices(df, schema):
    rng = np.random.default_rng(42)
    work = df.copy()
    cust = schema.get("customer")

    if cust and cust in df.columns:
        inv = []
        counter = 0
        for _, g in work.groupby(cust):
            idx = g.index.tolist()
            sizes = rng.integers(2, 7, len(idx))
            i = 0
            while i < len(idx):
                size = sizes[i]
                for _ in range(size):
                    if i >= len(idx): break
                    inv.append((idx[i], counter))
                    i += 1
                counter += 1
        work["SyntheticInvoice"] = work.index.map(dict(inv))
        schema["invoice"] = "SyntheticInvoice"
        return work, schema

    # Global grouping
    idx = work.index.tolist()
    sizes = rng.integers(2, 7, len(idx))
    inv = []
    counter = 0
    i = 0
    while i < len(idx):
        size = sizes[i]
        for _ in range(size):
            if i >= len(idx): break
            inv.append((idx[i], counter))
            i += 1
        counter += 1

    work["SyntheticInvoice"] = work.index.map(dict(inv))
    schema["invoice"] = "SyntheticInvoice"
    return work, schema


# ------------------------------------------------------------
# FIXED function (no dtype errors)
# ------------------------------------------------------------
def enrich_data_with_patterns(df):
    """
    Safely inject synthetic rules (Pasta->Cheese, etc.)
    WITHOUT causing dtype errors or Series merges.
    """

    if df.empty:
        return df

    product = df.columns[0]
    new_rows = []

    patterns = [
        ("PASTA", "CHEESE"),
        ("CEREAL", "MILK"),
        ("SHAMPOO", "CONDITIONER"),
    ]

    for _, row in df.iterrows():
        base = str(row[product]).upper()
        for a, b in patterns:
            if base == a:
                # ALWAYS build dict, never Series
                new_row = {col: row[col] for col in df.columns}
                new_row[product] = b
                new_rows.append(new_row)

    if not new_rows:
        return df

    new_df = pd.DataFrame(new_rows)
    new_df = new_df[df.columns]  # align columns

    return pd.concat([df, new_df], ignore_index=True)


# ------------------------------------------------------------
# Schema enforcement
# ------------------------------------------------------------
def ensure_synthetic_columns(df, schema):
    work = df.copy()
    n = len(work)

    if schema["product"] is None:
        work["SyntheticProduct"] = "ITEM_" + work.index.astype(str)
        schema["product"] = "SyntheticProduct"
    else:
        work[schema["product"]] = normalize_products(work[schema["product"]])

    if schema["customer"] is None:
        work["SyntheticCustomer"] = "CUST_" + (work.index % max(1, n)).astype(str)
        schema["customer"] = "SyntheticCustomer"

    if schema["date"] is None:
        rng = np.random.default_rng(42)
        work["SyntheticDate"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(
            rng.integers(0, 120, n), "D"
        )
        schema["date"] = "SyntheticDate"
    else:
        work[schema["date"]] = pd.to_datetime(work[schema["date"]], errors="coerce")
        if work[schema["date"]].notna().sum() == 0:
            rng = np.random.default_rng(42)
            work["SyntheticDate"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(
                rng.integers(0, 120, n), "D"
            )
            schema["date"] = "SyntheticDate"

    if schema["amount"] is None:
        work["SyntheticAmount"] = 1.0
        schema["amount"] = "SyntheticAmount"

    if schema["invoice"] is None:
        work, schema = create_realistic_invoices(work, schema)

    return work, schema


# ------------------------------------------------------------
# Transaction building
# ------------------------------------------------------------
def build_transactions(df, schema):
    inv = schema["invoice"]
    prod = schema["product"]
    g = df.groupby(inv)[prod].apply(lambda s: sorted(set(s)))
    return [t for t in g.tolist() if len(t) > 0]


def encode_transactions(trans):
    te = TransactionEncoder()
    arr = te.fit(trans).transform(trans)
    return pd.DataFrame(arr, columns=te.columns_)


# ------------------------------------------------------------
# Frequent patterns
# ------------------------------------------------------------
def run_mining(df_enc, algo, min_support, max_len):
    t0 = time.time()
    if algo == "Apriori":
        fi = apriori(df_enc, min_support=min_support, use_colnames=True, max_len=max_len)
    else:
        fi = fpgrowth(df_enc, min_support=min_support, use_colnames=True, max_len=max_len)
    return fi.sort_values("support", ascending=False), time.time() - t0


def make_rules(fi, metric="confidence", threshold=0.3):
    if fi.empty: return pd.DataFrame()
    rules = association_rules(fi, metric=metric, min_threshold=threshold)
    rules["antecedents_str"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents_str"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
    return rules.sort_values(["lift", "confidence"], ascending=False)


# ------------------------------------------------------------
# RFM
# ------------------------------------------------------------
def build_rfm(df, schema):
    cust, date, amt = schema["customer"], schema["date"], schema["amount"]
    if any(col not in df.columns for col in [cust, date, amt]):
        return pd.DataFrame()

    work = df.dropna(subset=[cust, date])
    if work.empty:
        return pd.DataFrame()

    now = work[date].max()
    grp = (
        work.groupby(cust)
        .agg(
            Recency=(date, lambda s: (now - s.max()).days),
            Frequency=(date, "count"),
            Monetary=(amt, "sum"),
        )
        .reset_index()
        .rename(columns={cust: "customerid"})
    )

    if grp.empty: return grp

    try:
        grp["R_Score"] = pd.qcut(
            grp["Recency"].rank(ascending=True), 5, labels=[5,4,3,2,1]
        )
        grp["F_Score"] = pd.qcut(
            grp["Frequency"].rank(ascending=True), 5, labels=[1,2,3,4,5]
        )
        grp["M_Score"] = pd.qcut(
            grp["Monetary"].rank(ascending=True), 5, labels=[1,2,3,4,5]
        )
    except:
        return grp

    grp["RFM_Score"] = grp["R_Score"].astype(int) + grp["F_Score"].astype(int) + grp["M_Score"].astype(int)
    grp["Segment"] = pd.cut(
        grp["RFM_Score"], bins=[0,6,9,12,15],
        labels=["Low","Mid","High","VIP"], include_lowest=True
    )
    return grp


# ------------------------------------------------------------
# Recommender (staple-filtered)
# ------------------------------------------------------------
def rules_to_recommendations(rules, base_items, df_enc):
    if rules.empty or not base_items:
        return pd.DataFrame()

    try:
        support = df_enc.mean().sort_values(ascending=False)
        staples = set(support.head(3).index)
    except:
        staples = set()

    sub = rules[rules["antecedents"].apply(lambda s: set(base_items).issubset(s))]
    if sub.empty:
        return pd.DataFrame()

    sub = sub.copy()
    sub["consequents"] = sub["consequents"].apply(
        lambda s: {x for x in s if x not in staples}
    )
    sub = sub[sub["consequents"].apply(lambda s: len(s) > 0)]
    if sub.empty:
        return pd.DataFrame()

    recs = (
        sub.explode("consequents")
        .groupby("consequents")
        .agg(
            mean_conf=("confidence","mean"),
            mean_lift=("lift","mean"),
            count=("confidence","size"),
        )
        .sort_values(["mean_lift","mean_conf","count"], ascending=False)
        .head(10)
    )

    recs.index = recs.index.astype(str)
    return recs


# ============================================================
# Upload Section
# ============================================================

uploaded = st.sidebar.file_uploader("📤 Upload CSV Dataset", type=["csv"])

if uploaded is None:
    st.info("Please upload a dataset to begin.")
    st.stop()

df_raw = pd.read_csv(uploaded)
st.write(f"Dataset Loaded: **{len(df_raw):,} rows**, **{len(df_raw.columns):} columns**")
st.dataframe(df_raw.head(), use_container_width=True)

# ============================================================
# Schema Detection
# ============================================================

product_col = detect_product_column(df_raw)
invoice_col = detect_invoice_column(df_raw, product_col)
customer_col = detect_customer_column(df_raw, {product_col, invoice_col})
date_col = detect_date_column(df_raw)
amount_col = detect_amount_column(df_raw)

schema = {
    "product": product_col,
    "invoice": invoice_col,
    "customer": customer_col,
    "date": date_col,
    "amount": amount_col,
}

st.subheader("🧭 Auto Schema Detection")
st.json(schema)

# ============================================================
# Ensure schema
# ============================================================

df_work, schema = ensure_synthetic_columns(df_raw, schema)

st.subheader("📌 Active Schema (after fixing)")
st.json(schema)

# ============================================================
# Tabs
# ============================================================

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧹 Cleaning",
    "📊 Patterns",
    "🕒 Temporal",
    "👥 Segmentation",
    "🛒 Recommender",
    "📈 Pipeline"
])

# ============================================================
# TAB 0 Cleaning
# ============================================================
with tab0:

    st.markdown("### Missing Values")
    miss = df_raw.isna().sum().to_frame("Missing")
    miss["%"] = (miss["Missing"] / len(df_raw) * 100).round(2)
    st.dataframe(miss, use_container_width=True)

    st.markdown("### Cleaned Data Preview")
    st.dataframe(df_work.head(), use_container_width=True)

    # Basket size
    trans = build_transactions(df_work, schema)
    sizes = [len(t) for t in trans]
    if sizes:
        fig = px.histogram(sizes, nbins=20, title="Basket Size Distribution")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 1 Patterns
# ============================================================
with tab1:
    st.subheader("⚡ Frequent Patterns")

    trans = build_transactions(df_work, schema)
    df_enc = encode_transactions(trans)

    algo = st.selectbox("Choose Algorithm", ["FP-Growth", "Apriori"])
    min_support = st.slider("Min Support", 0.001, 0.1, 0.01)
    max_len = st.slider("Max Itemset Length", 2, 5, 3)

    fi, runtime = run_mining(df_enc, algo, min_support, max_len)

    st.info(f"Found **{len(fi):,}** itemsets in **{runtime:.3f}s**")

    st.dataframe(fi.head(), use_container_width=True)

    # Rules
    metric = st.selectbox("Metric", ["confidence", "lift"])
    min_metric = st.slider("Min Metric Threshold", 0.1, 1.0, 0.3)

    rules = make_rules(fi, metric, min_metric)

    if not rules.empty:
        st.subheader("Top Rules")
        st.dataframe(
            rules[["antecedents_str", "consequents_str", "support", "confidence", "lift"]].head(30),
            use_container_width=True,
        )

        # Scatter plot
        fig = px.scatter(
            rules,
            x="support",
            y="confidence",
            size="lift",
            hover_data=["antecedents_str", "consequents_str"],
            title="Rules Scatter (Support vs Confidence)"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.session_state["rules"] = rules
    st.session_state["df_enc"] = df_enc

# ============================================================
# TAB 2 Temporal
# ============================================================
with tab2:
    st.subheader("📅 Temporal Analysis")

    date = schema["date"]
    amt = schema["amount"]

    df_t = df_work.dropna(subset=[date]).copy()

    if not df_t.empty:
        df_t["Month"] = df_t[date].dt.to_period("M").astype(str)
        df_t["Weekday"] = df_t[date].dt.day_name()

        # Monthly
        fig1 = px.line(
            df_t.groupby("Month").size().reset_index(name="Transactions"),
            x="Month", y="Transactions", title="Transactions per Month"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Weekday
        fig2 = px.bar(
            df_t.groupby("Weekday").size().reset_index(name="Transactions"),
            x="Weekday", y="Transactions", title="Transactions by Day of Week"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Revenue
        fig3 = px.line(
            df_t.groupby("Month")[amt].sum().reset_index(),
            x="Month", y=amt, title="Revenue per Month"
        )
        st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# TAB 3 Segmentation
# ============================================================
with tab3:
    st.subheader("🎯 Customer Segmentation (RFM + KMeans)")

    rfm = build_rfm(df_work, schema)
    if rfm.empty:
        st.warning("Not enough data for RFM segmentation.")
    else:
        st.dataframe(rfm.head(), use_container_width=True)

        # Pie chart
        fig = px.pie(
            rfm,
            names="Segment",
            title="Customer Distribution by Segment"
        )
        st.plotly_chart(fig, use_container_width=True)

        # KMeans
        k = st.slider("KMeans Clusters", 2, 6, 3)
        try:
            df_rfm = rfm.copy()
            feats = df_rfm[["Recency","Frequency","Monetary"]]
            X = StandardScaler().fit_transform(feats)
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
            df_rfm["Cluster"] = km.labels_

            fig_sc = px.scatter(
                df_rfm, x="Frequency", y="Monetary",
                color="Cluster",
                hover_data=["customerid","Segment"],
                title="KMeans Clustering"
            )
            st.plotly_chart(fig_sc, use_container_width=True)
        except:
            st.error("KMeans failed due to insufficient variability.")

# ============================================================
# TAB 4 Recommender
# ============================================================
with tab4:
    st.subheader("🛒 Smart Recommender")

    rules = st.session_state.get("rules", pd.DataFrame())
    df_enc = st.session_state.get("df_enc", pd.DataFrame())

    if rules.empty:
        st.warning("No rules available.")
    else:
        all_items = sorted(
            set().union(*rules["antecedents"]).union(*rules["consequents"])
        )

        base = st.multiselect("Select base item(s)", all_items)

        if base:
            recs = rules_to_recommendations(rules, base, df_enc)

            if recs.empty:
                st.info("No meaningful recommendations found.")
            else:
                st.dataframe(recs, use_container_width=True)

                # Chart
                fig = px.bar(
                    recs,
                    x=recs.index,
                    y="mean_lift",
                    title="Recommendation Strength (Lift)"
                )
                st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 5 Pipeline
# ============================================================
with tab5:
    st.subheader("📈 Methodology Pipeline")

    labels = [
        "Raw Data",
        "Cleaned Data",
        "Baskets",
        "Frequent Itemsets",
        "Rules",
        "Recommender"
    ]
    source = [0,1,2,3,4]
    target = [1,2,3,4,5]
    value = [10,10,10,10,10]

    fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=25, thickness=20),
        link=dict(source=source, target=target, value=value),
    ))
    fig.update_layout(title="End-to-End Retail Intelligence Pipeline")

    st.plotly_chart(fig, use_container_width=True)
