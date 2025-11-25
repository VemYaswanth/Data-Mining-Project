# app_streamlit_v5_1.py
# ----------------------------------------------------------
# Retail Intelligence Dashboard v5.1
# - Single dataset: Upload CSV only (no internal dataset picker)
# - Auto schema detection + synthetic fields
# - Tabs:
#     0. Data Cleaning
#     1. Patterns & Model Comparison
#     2. Temporal & Sequential
#     3. Customer Segmentation
#     4. Smart Recommender
#     5. Visual Explanations (pipeline diagrams)
# - Added comments everywhere to explain logic for academic submission
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
# Streamlit configuration (page title, icon, layout)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Retail Intelligence Dashboard v5.1",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Retail Intelligence Dashboard v5.1")
st.caption(
    "End-to-end: Cleaning → Patterns → Temporal → Segmentation → Recommender.\n"
    "Includes automatic schema detection, realistic synthetic fields, model comparison, and visual explanations."
)

# =========================================================
# Helper Functions — Data Processing & Detection
# =========================================================

def normalize_products(s: pd.Series) -> pd.Series:
    """
    Normalize product names:
    - Convert to string
    - Uppercase for consistency
    - Strip leading/trailing whitespace
    """
    return s.astype(str).str.upper().str.strip()


def detect_product_column(df: pd.DataFrame):
    """
    Guess which column is the 'product name' column.
    Heuristic:
    - Must be object/string type
    - Not too few uniques (not constant)
    - Not almost unique (not an ID)
    - We pick the text column with medium-high cardinality.
    """
    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if not obj_cols:
        return None

    scores = []
    n = len(df)
    for col in obj_cols:
        nun = df[col].nunique(dropna=True)
        if nun <= 5:
            continue  # too few distinct values, likely not product
        if nun >= 0.95 * n:
            continue  # almost unique per row, likely an ID
        scores.append((col, nun / n))  # score = uniqueness ratio

    if not scores:
        return None
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0]  # column with highest uniqueness ratio


def detect_invoice_column(df: pd.DataFrame, product_col: str | None):
    """
    Guess which column can serve as invoice/basket ID.
    Heuristic:
    - Repeated values (not unique)
    - Not the product column
    - Average group size > 1.5 (so grouping makes sense)
    """
    candidates = []
    n = len(df)
    for col in df.columns:
        if col == product_col:
            continue
        nun = df[col].nunique(dropna=True)
        if nun <= 1 or nun >= n:
            continue  # constant or unique → not useful for grouping
        avg_group = n / nun if nun > 0 else 0
        if avg_group < 1.5:
            continue
        candidates.append((col, avg_group))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def detect_customer_column(df: pd.DataFrame, used_cols: set):
    """
    Guess a 'customer-like' column:
    - High cardinality but not almost unique
    - Not already used (e.g., product or invoice)
    """
    n = len(df)
    best = None
    best_score = -1
    for col in df.columns:
        if col in used_cols:
            continue
        nun = df[col].nunique(dropna=True)
        if nun <= 5:
            continue
        ratio = nun / n  # uniqueness ratio
        # want moderate-to-high cardinality, but not ~100% unique
        if 0.05 <= ratio <= 0.95 and ratio > best_score:
            best = col
            best_score = ratio
    return best


def detect_date_column(df: pd.DataFrame):
    """
    Try parsing each column as datetime.
    Pick the one with the highest proportion of valid parsed dates.
    """
    best = None
    best_rate = 0
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            ok_rate = parsed.notna().mean()  # percentage of valid dates
            if ok_rate > 0.5 and ok_rate > best_rate:
                best = col
                best_rate = ok_rate
        except Exception:
            continue
    return best


def detect_amount_column(df: pd.DataFrame):
    """
    Guess an 'amount' / 'total' column:
    - Numeric type
    - Prefer names containing amount-like keywords
    - Otherwise, pick numeric column with largest variance.
    """
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:
        return None

    preferred = []
    for col in num_cols:
        cname = col.lower()
        if any(key in cname for key in ["final", "total", "amount", "sales", "revenue", "net"]):
            preferred.append(col)
    if preferred:
        return preferred[0]

    variances = []
    for col in num_cols:
        variances.append((col, df[col].var()))
    variances.sort(key=lambda x: (x[1] if pd.notna(x[1]) else 0), reverse=True)
    return variances[0][0] if variances else None


def create_realistic_invoices(work: pd.DataFrame, schema: dict):
    """
    Create realistic synthetic invoices (basket IDs) when no invoice column exists.

    Case 1: If a customer column exists:
        - For each customer, group their rows into variable basket sizes (2–7 items).
    Case 2: No customer column:
        - Globally group rows into variable-size baskets (2–7 items).

    This avoids unrealistic grouping like index//3 and gives
    more realistic shopping carts.
    """
    rng = np.random.default_rng(42)
    n = len(work)

    cust = schema.get("customer")
    if cust and cust in work.columns:
        # Per-customer grouping
        inv_ids = []
        invoice_counter = 0
        for _, group in work.groupby(cust):
            idx = group.index.tolist()
            size_vec = rng.integers(2, 7, size=len(group))  # random basket size 2–7
            i = 0
            while i < len(idx):
                size = size_vec[i]
                for _ in range(size):
                    if i >= len(idx):
                        break
                    inv_ids.append((idx[i], invoice_counter))
                    i += 1
                invoice_counter += 1
        uid_map = dict(inv_ids)
        work["SyntheticInvoice"] = work.index.map(uid_map)
        schema["invoice"] = "SyntheticInvoice"
        return work, schema

    # No customer column → global grouping
    inv_ids = []
    idx = work.index.tolist()
    size_vec = rng.integers(2, 7, size=n)
    invoice_counter = 0
    i = 0
    while i < n:
        size = size_vec[i]
        for _ in range(size):
            if i >= n:
                break
            inv_ids.append((idx[i], invoice_counter))
            i += 1
        invoice_counter += 1

    uid_map = dict(inv_ids)
    work["SyntheticInvoice"] = work.index.map(uid_map)
    schema["invoice"] = "SyntheticInvoice"
    return work, schema


def ensure_synthetic_columns(df: pd.DataFrame, schema: dict):
    """
    Ensure the dataset has usable:
    - product column
    - customer column
    - date column
    - amount column
    - invoice column

    If any is missing, we create a synthetic version so that
    the pipeline NEVER breaks.
    """
    work = df.copy()
    n = len(work)

    # Product: synthetic item label if missing
    if schema["product"] is None:
        work["SyntheticProduct"] = "ITEM_" + work.index.astype(str)
        schema["product"] = "SyntheticProduct"
        st.warning("No product-like column detected → using SyntheticProduct.")
    else:
        work[schema["product"]] = normalize_products(work[schema["product"]])

    # Customer: synthetic if missing
    if schema["customer"] is None:
        work["SyntheticCustomer"] = "CUST_" + (work.index % max(n, 1)).astype(str)
        schema["customer"] = "SyntheticCustomer"
        st.info("No customer-like column detected → SyntheticCustomer created.")

    # Date: synthetic date range if missing or invalid
    if schema["date"] is None:
        rng = np.random.default_rng(42)
        work["SyntheticDate"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(
            rng.integers(0, 120, size=n), unit="D"
        )
        schema["date"] = "SyntheticDate"
        st.info("No date-like column detected → SyntheticDate created.")
    else:
        work[schema["date"]] = pd.to_datetime(
            work[schema["date"]], errors="coerce", infer_datetime_format=True
        )
        if work[schema["date"]].notna().sum() == 0:
            rng = np.random.default_rng(42)
            work["SyntheticDate"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(
                rng.integers(0, 120, size=n), unit="D"
            )
            schema["date"] = "SyntheticDate"
            st.info("Mapped date column had no valid dates → SyntheticDate created.")

    # Amount: synthetic flat amount if missing
    if schema["amount"] is None:
        work["SyntheticAmount"] = 1.0
        schema["amount"] = "SyntheticAmount"
        st.info("No numeric 'amount' column detected → SyntheticAmount = 1 used.")

    # Invoice: realistic synthetic baskets if missing
    if schema["invoice"] is None:
        work, schema = create_realistic_invoices(work, schema)
        st.info("Synthetic realistic invoices created (basket sizes 2–7).")

    return work, schema


def build_transactions(df: pd.DataFrame, schema: dict):
    """
    Convert the line-item dataset into basket transactions.

    - Group by invoice ID
    - Collect all products in each invoice into a list
    """
    prod = schema["product"]
    inv = schema["invoice"]
    grouped = df.groupby(inv)[prod].apply(lambda s: sorted(set(s.tolist())))
    transactions = [t for t in grouped.tolist() if len(t) > 0]
    return transactions


def encode_transactions(transactions):
    """
    Convert a list of transactions into a one-hot encoded DataFrame.
    Each column = item, each row = basket, True = item present.
    """
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)
    return df_encoded


def run_mining(df_encoded: pd.DataFrame, algo: str, min_support: float, max_len: int):
    """
    Run FP-Growth or Apriori to mine frequent itemsets.
    Returns:
    - frequent itemsets sorted by support
    - runtime in seconds
    """
    t0 = time.time()
    if algo == "Apriori":
        fi = apriori(df_encoded, min_support=min_support, use_colnames=True, max_len=max_len)
    else:
        fi = fpgrowth(df_encoded, min_support=min_support, use_colnames=True, max_len=max_len)
    elapsed = time.time() - t0
    fi = fi.sort_values("support", ascending=False)
    return fi, elapsed


def make_rules(fi: pd.DataFrame, metric: str, min_threshold: float):
    """
    Generate association rules from frequent itemsets.
    Adds readable string versions of antecedents and consequents.
    """
    if fi is None or fi.empty:
        return pd.DataFrame()

    rules = association_rules(fi, metric=metric, min_threshold=min_threshold)
    rules["antecedents_str"] = rules["antecedents"].apply(
        lambda s: ", ".join(sorted(list(s)))
    )
    rules["consequents_str"] = rules["consequents"].apply(
        lambda s: ", ".join(sorted(list(s)))
    )
    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False)
    return rules


def plot_top_itemsets(fi: pd.DataFrame, n=20):
    """
    Horizontal bar chart for top itemsets by support.
    """
    if fi.empty:
        return None
    tmp = fi.head(n).copy()
    tmp["itemset_str"] = tmp["itemsets"].apply(lambda s: ", ".join(sorted(list(s))))
    fig = px.bar(
        tmp[::-1],
        x="support",
        y="itemset_str",
        orientation="h",
        title=f"Top {n} Frequent Itemsets",
    )
    return fig


def plot_rules_scatter(rules: pd.DataFrame):
    """
    Scatterplot of rules: support vs confidence, size ~ lift.
    Helps compare strength of rules visually.
    """
    if rules.empty:
        return None
    fig = px.scatter(
        rules,
        x="support",
        y="confidence",
        size="lift",
        hover_data=["antecedents_str", "consequents_str"],
        title="Rules: Support vs Confidence (size = Lift)",
    )
    return fig


def build_association_network(rules: pd.DataFrame, topn=30):
    """
    Build a network graph:
    - Nodes: items
    - Directed edges: association rules (A → B)
    Only uses top N rules by order in the rules DataFrame.
    """
    if rules.empty:
        return None
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

    fig = go.Figure(
        data=[
            go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none", line=dict(width=1)),
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=labels,
                textposition="top center",
            ),
        ]
    )
    fig.update_layout(
        title="Association Network (Top Rules by Lift)",
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def missing_summary(df: pd.DataFrame):
    """
    Table summarizing missing values per column (count + percentage).
    """
    total = df.isna().sum()
    pct = (total / len(df) * 100).round(2) if len(df) else 0
    out = pd.DataFrame({"Missing": total, "Missing_%": pct})
    return out.sort_values("Missing", ascending=False)


def build_rfm(df: pd.DataFrame, schema: dict):
    """
    Build Recency-Frequency-Monetary table per customer.

    Recency  = days since last purchase
    Frequency = number of transactions
    Monetary  = total amount spent
    """
    cust = schema["customer"]
    date = schema["date"]
    amt = schema["amount"]
    if cust not in df.columns or date not in df.columns or amt not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work = work.dropna(subset=[cust, date])
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
    if grp.empty:
        return grp

    # Convert R/F/M into 1–5 scores via quantiles or bins
    try:
        grp["R_Score"] = pd.qcut(
            grp["Recency"].rank(method="first", ascending=True),
            5,
            labels=[5, 4, 3, 2, 1],
        )  # smaller Recency = more recent = higher score
        grp["F_Score"] = pd.qcut(
            grp["Frequency"].rank(method="first", ascending=True),
            5,
            labels=[1, 2, 3, 4, 5],
        )
        grp["M_Score"] = pd.qcut(
            grp["Monetary"].rank(method="first", ascending=True),
            5,
            labels=[1, 2, 3, 4, 5],
        )
    except Exception:
        grp["R_Score"] = pd.cut(
            grp["Recency"], bins=5, labels=[5, 4, 3, 2, 1], include_lowest=True
        )
        grp["F_Score"] = pd.cut(
            grp["Frequency"], bins=5, labels=[1, 2, 3, 4, 5], include_lowest=True
        )
        grp["M_Score"] = pd.cut(
            grp["Monetary"], bins=5, labels=[1, 2, 3, 4, 5], include_lowest=True
        )

    grp["RFM_Score"] = (
        grp["R_Score"].astype(int)
        + grp["F_Score"].astype(int)
        + grp["M_Score"].astype(int)
    )
    grp["Segment"] = pd.cut(
        grp["RFM_Score"],
        bins=[0, 6, 9, 12, 15],
        labels=["Low", "Mid", "High", "VIP"],
        include_lowest=True,
    )
    return grp


def kmeans_clusters(df_rfm: pd.DataFrame, k=3):
    """
    Run K-Means clustering on R, F, M to create numerical clusters
    on top of the RFM segmentation.
    """
    try:
        feats = df_rfm[["Recency", "Frequency", "Monetary"]].fillna(0).copy()
        scaler = StandardScaler()
        X = scaler.fit_transform(feats)
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        df_rfm = df_rfm.copy()
        df_rfm["KMeansCluster"] = labels
        centers = pd.DataFrame(
            scaler.inverse_transform(km.cluster_centers_),
            columns=["Recency", "Frequency", "Monetary"],
        )
        centers["Cluster"] = centers.index
        return df_rfm, centers
    except Exception:
        return df_rfm, pd.DataFrame()


def rules_to_recommendations(rules: pd.DataFrame, base_items, enc: pd.DataFrame):
    """
    Core recommendation logic:

    1. Compute global item support from encoded matrix (enc).
    2. Identify top global "staples" (e.g., 3 most frequent items).
    3. Filter rules whose antecedents contain all selected base items.
    4. Remove staples from consequents (so we avoid trivial recs like BREAD).
    5. Aggregate remaining consequents:
       - mean_conf = average confidence across rules
       - mean_lift = average lift
       - count     = number of rules supporting that consequent

    Returned sorted DataFrame is the final recommended add-ons.
    """
    if rules.empty or not base_items:
        return pd.DataFrame()

    # 1) Global item support to find staples
    try:
        item_support = enc.mean().sort_values(ascending=False)
        top_staples = set(item_support.head(3).index)  # top-3 globally frequent items
    except Exception:
        top_staples = set()

    # 2) Select rules whose antecedents contain all selected base items
    sub = rules[rules["antecedents"].apply(lambda s: set(base_items).issubset(s))]
    if sub.empty:
        return pd.DataFrame()

    # 3) Remove staples from consequents
    sub_filtered = sub.copy()
    sub_filtered["consequents"] = sub_filtered["consequents"].apply(
        lambda s: {x for x in s if x not in top_staples}
    )
    sub_filtered = sub_filtered[sub_filtered["consequents"].apply(lambda s: len(s) > 0)]
    if sub_filtered.empty:
        return pd.DataFrame()

    # 4) Aggregate consequential items as recommendations
    recs = (
        sub_filtered.explode("consequents")
        .groupby("consequents")
        .agg(
            mean_conf=("confidence", "mean"),
            mean_lift=("lift", "mean"),
            count=("confidence", "size"),
        )
        .sort_values(["mean_lift", "mean_conf", "count"], ascending=False)
        .head(10)
    )
    recs.index = recs.index.astype(str)
    return recs


# =========================================================
# 1. Upload Dataset — Only Upload, No Internal Selection
# =========================================================

st.sidebar.header("Upload Dataset")
uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded is None:
    st.info("Please upload a CSV dataset to begin.")
    st.stop()

# Read CSV into DataFrame
df_raw = pd.read_csv(uploaded)

st.write(f"Rows: **{len(df_raw):,}**, Columns: **{len(df_raw.columns):,}**")
st.dataframe(df_raw.head(20), use_container_width=True)

# =========================================================
# 2. Auto Schema Detection + Synthetic Fixes
# =========================================================

st.subheader("Auto Schema Detection")

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

st.markdown("**Detected schema (before synthetic fixes):**")
st.json(schema)

# Ensure all required fields exist (synthetic if needed)
df_work, schema = ensure_synthetic_columns(df_raw, schema)
st.markdown("**Active schema (after synthetic fixes):**")
st.json(schema)

# =========================================================
# Tabs Layout (including visual explanations tab)
# =========================================================

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🧹 Data Cleaning",
        "📊 Patterns & Model Comparison",
        "🕒 Temporal & Sequential",
        "👥 Customer Segmentation",
        "🛒 Smart Recommender",
        "📈 Visual Explanations",
    ]
)

# =========================================================
# TAB 0: Data Cleaning & Basic Graphs
# =========================================================

with tab0:
    st.subheader("Data Cleaning & Overview")

    # Show missing values per column
    st.markdown("### Missing Value Summary")
    miss = missing_summary(df_raw)
    st.dataframe(miss, use_container_width=True)

    st.markdown("### Cleaned Working Data (after auto schema & synthetics)")
    st.dataframe(df_work.head(20), use_container_width=True)

    # Graph: distribution of basket sizes (how many items per invoice)
    st.markdown("### Basket Size Distribution (Items per Invoice)")
    transactions_preview = build_transactions(df_work, schema)
    basket_sizes = [len(t) for t in transactions_preview]
    if basket_sizes:
        size_series = pd.Series(basket_sizes, name="BasketSize")
        fig_sizes = px.histogram(
            size_series,
            x="BasketSize",
            nbins=20,
            title="Distribution of Basket Sizes",
        )
        st.plotly_chart(fig_sizes, use_container_width=True)
    else:
        st.info("No baskets available yet (unexpected after synthetic fixes).")

    # Allow downloading cleaned dataset
    csv_bytes = df_work.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download cleaned_dataset.csv",
        data=csv_bytes,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )

# =========================================================
# TAB 1: Frequent Patterns & Model Comparison
# =========================================================

with tab1:
    st.subheader("Frequent Patterns & Model Comparison (FP-Growth vs Apriori)")

    transactions = build_transactions(df_work, schema)
    st.write(f"Built **{len(transactions):,}** basket transactions.")

    if not transactions:
        st.warning("No transactions found even after synthetic fixes.")
        st.stop()

    # One-hot encode baskets
    df_enc = encode_transactions(transactions)
    st.write(f"Unique items in baskets: **{df_enc.shape[1]:,}**")

    # User controls for support, itemset length
    colA, colB = st.columns(2)
    min_support = colA.slider("Min Support", 0.001, 0.1, 0.01, step=0.001)
    max_len = colB.slider("Max Itemset Length", 2, 5, 3)

    # Model Comparison: FP-Growth vs Apriori
    st.markdown("### Model Comparison")
    fi_fp, t_fp = run_mining(df_enc, "FP-Growth", min_support, max_len)
    fi_ap, t_ap = run_mining(df_enc, "Apriori", min_support, max_len)

    comp = pd.DataFrame(
        {
            "Algorithm": ["FP-Growth", "Apriori"],
            "Runtime (s)": [round(t_fp, 3), round(t_ap, 3)],
            "Frequent Itemsets": [len(fi_fp), len(fi_ap)],
        }
    )
    st.dataframe(comp, use_container_width=True)

    # Graph: model comparison as bar chart
    fig_comp = px.bar(
        comp,
        x="Algorithm",
        y="Frequent Itemsets",
        color="Algorithm",
        title="Frequent Itemsets Found: FP-Growth vs Apriori",
        text="Frequent Itemsets",
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Build rules from FP-Growth (faster, more scalable)
    metric = st.selectbox("Rule Metric", ["confidence", "lift"])
    min_metric = st.slider(f"Min {metric.title()}", 0.1, 1.0, 0.3, step=0.05)
    topn = st.slider("Top N Itemsets to display", 10, 50, 20, step=5)

    rules = make_rules(fi_fp, metric, min_metric)
    st.info(f"Rules generated (from FP-Growth): **{len(rules):,}**")

    # Chart: Top frequent itemsets
    if not fi_fp.empty:
        fig_top = plot_top_itemsets(fi_fp, n=topn)
        if fig_top:
            st.plotly_chart(fig_top, use_container_width=True)

    # Chart: scatter of rules
    if not rules.empty:
        fig_rules = plot_rules_scatter(rules)
        if fig_rules:
            st.plotly_chart(fig_rules, use_container_width=True)

        st.markdown("### Top Association Rules")
        st.dataframe(
            rules[["antecedents_str", "consequents_str", "support", "confidence", "lift"]].head(30),
            use_container_width=True,
        )

        # Network graph of item associations
        fig_net = build_association_network(rules, topn=30)
        if fig_net:
            st.plotly_chart(fig_net, use_container_width=True)

    # Store for recommender tab
    st.session_state["df_enc"] = df_enc
    st.session_state["rules"] = rules

# =========================================================
# TAB 2: Temporal & Sequential Analysis
# =========================================================

with tab2:
    st.subheader("Temporal & Sequential Analysis")

    date_col = schema["date"]
    amt_col = schema["amount"]
    prod_col = schema["product"]
    cust_col = schema["customer"]

    if date_col in df_work.columns:
        df_t = df_work.copy()
        df_t = df_t.dropna(subset=[date_col])
        if not df_t.empty:
            # Monthly trend: number of transactions per month
            monthly = (
                df_t.groupby(df_t[date_col].dt.to_period("M"))
                .size()
                .reset_index(name="tx_count")
            )
            monthly[date_col] = monthly[date_col].astype(str)
            if not monthly.empty:
                st.markdown("### Transactions per Month")
                st.line_chart(monthly.set_index(date_col))

            # Weekday trend: number of transactions by day-of-week
            weekday = (
                df_t.groupby(df_t[date_col].dt.day_name())
                .size()
                .reset_index(name="tx_count")
            )
            if not weekday.empty:
                weekday = weekday.sort_values("tx_count", ascending=False)
                weekday = weekday.rename(columns={weekday.columns[0]: "Day"})
                st.markdown("### Transactions by Day of Week")
                st.bar_chart(weekday.set_index("Day"))

            # Monthly revenue trend if amount column exists
            if amt_col in df_t.columns:
                amt_monthly = (
                    df_t.groupby(df_t[date_col].dt.to_period("M"))[amt_col]
                    .sum()
                    .reset_index()
                )
                amt_monthly[date_col] = amt_monthly[date_col].astype(str)
                if not amt_monthly.empty:
                    st.markdown("### Revenue per Month")
                    st.line_chart(amt_monthly.set_index(date_col))

            # Sequential analysis: A → B transitions at customer level
            if cust_col in df_t.columns and prod_col in df_t.columns:
                seq_data = (
                    df_t.dropna(subset=[prod_col])
                    .sort_values([cust_col, date_col])
                    .copy()
                )
                seq_data[prod_col] = normalize_products(seq_data[prod_col])
                seq_counts = Counter()
                grouped = seq_data.groupby(cust_col)[prod_col].apply(list)
                for seq in grouped:
                    for i in range(len(seq) - 1):
                        seq_counts[(seq[i], seq[i + 1])] += 1
                if seq_counts:
                    seq_df = pd.DataFrame(
                        [(a, b, c) for (a, b), c in seq_counts.items()],
                        columns=["From", "To", "Count"],
                    ).sort_values("Count", ascending=False)
                    st.markdown("### Top Sequential Transitions (A → B)")
                    st.dataframe(seq_df.head(20), use_container_width=True)
        else:
            st.info("No valid dates after parsing for temporal analysis.")
    else:
        st.info("No usable date column available (real or synthetic).")

# =========================================================
# TAB 3: Customer Segmentation (RFM + K-Means)
# =========================================================

with tab3:
    st.subheader("Customer Segmentation (RFM + K-Means)")

    rfm = build_rfm(df_work, schema)
    if rfm.empty:
        st.info(
            "RFM requires customer, date, and amount columns (real or synthetic). "
            "These were not sufficient in the current dataset."
        )
    else:
        st.markdown("### RFM Summary (Top 50 customers)")
        st.dataframe(rfm.head(50), use_container_width=True)

        # K-Means clustering on R, F, M
        k = st.slider("Number of K-Means clusters", 2, 6, 3)
        rfm_km, centers = kmeans_clusters(rfm, k=k)

        if not centers.empty:
            st.markdown("### Cluster Centers (approximate original scale)")
            st.dataframe(centers, use_container_width=True)

        # Scatter plot of customers colored by KMeansCluster
        try:
            fig_seg = px.scatter(
                rfm_km,
                x="Frequency",
                y="Monetary",
                color=rfm_km.get("KMeansCluster", pd.Series([0] * len(rfm_km))).astype(str),
                hover_data=["customerid", "Recency", "RFM_Score", "Segment"],
                title="Customer Segmentation Scatter (Frequency vs Monetary)",
            )
            st.plotly_chart(fig_seg, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not plot segmentation scatter: {e}")

        # Segment-level summary table
        seg_summary = (
            rfm_km.groupby("Segment", dropna=False)
            .agg(
                Customers=("customerid", "count"),
                AvgSpend=("Monetary", "mean"),
                AvgFreq=("Frequency", "mean"),
                AvgRecency=("Recency", "mean"),
            )
            .round(2)
        )
        st.markdown("### Segment Summary")
        st.dataframe(seg_summary, use_container_width=True)

        # Pie chart of customer counts per Segment
        seg_counts = rfm_km["Segment"].value_counts(dropna=False).reset_index()
        seg_counts.columns = ["Segment", "Customers"]
        fig_pie = px.pie(
            seg_counts,
            names="Segment",
            values="Customers",
            title="Customer Distribution by Segment",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# =========================================================
# TAB 4: Smart Recommender (Staple-Filtered)
# =========================================================

with tab4:
    st.subheader("Smart Recommender (Staple-Filtered)")

    rules = st.session_state.get("rules", pd.DataFrame())
    df_enc = st.session_state.get("df_enc", pd.DataFrame())
    prod_col = schema["product"]
    amt_col = schema["amount"]
    inv_col = schema["invoice"]

    if rules.empty or df_enc.empty:
        st.warning(
            "No rules or encoded baskets available. "
            "Please ensure Tab 1 has run with at least some itemsets and rules."
        )
    else:
        # Candidate list of items for the user to choose from
        try:
            all_items = sorted(
                set().union(*rules["antecedents"]).union(*rules["consequents"])
            )
        except Exception:
            all_items = sorted(df_enc.columns.tolist())

        selected = st.multiselect("Select base product(s) for recommendations", all_items)

        if selected:
            # Core recommendation call
            recs = rules_to_recommendations(rules, selected, df_enc)
            if recs.empty:
                st.info(
                    "No non-staple recommendations found for this selection. "
                    "Try a different product or adjust support/metric thresholds in Tab 1."
                )
            else:
                st.markdown("### Top Recommended Add-ons (after filtering staples)")
                st.dataframe(recs, use_container_width=True)

                # Graph: mean lift bar chart for recommended items
                st.markdown("#### Recommendation Strength (Mean Lift)")
                st.bar_chart(recs["mean_lift"])

                # Estimate revenue impact where rule holds
                if (
                    inv_col in df_work.columns
                    and prod_col in df_work.columns
                    and amt_col in df_work.columns
                ):
                    df_seg = df_work.copy()
                    df_seg[prod_col] = normalize_products(df_seg[prod_col])
                    by_inv = df_seg.groupby(inv_col).agg(
                        items=(prod_col, lambda s: set(s.tolist())),
                        revenue=(amt_col, "sum"),
                    )
                    impacts = []
                    for item in recs.index:
                        mask = by_inv["items"].apply(
                            lambda s: set(selected).issubset(s) and (item in s)
                        )
                        impacts.append(
                            {
                                "item": item,
                                "support_in_invoices": int(mask.sum()),
                                "revenue_sum": by_inv.loc[mask, "revenue"].sum(),
                            }
                        )
                    impact_df = pd.DataFrame(impacts).sort_values(
                        "revenue_sum", ascending=False
                    )
                    st.markdown("#### Estimated Revenue Impact (where rules hold)")
                    st.dataframe(impact_df, use_container_width=True)

                # Textual explanation for viva / lecturer
                st.markdown("### Explanation of How Recommendations are Computed")
                st.write(
                    "- First, we mine association rules from frequent itemsets (Tab 1).\n"
                    "- When you select a base product (or multiple products), "
                    "we filter rules whose antecedents contain all of those products.\n"
                    "- Next, we compute global item support and remove the top-3 most "
                    "frequent 'staple' items from recommendations (e.g., BREAD, MILK).\n"
                    "- The remaining consequents are aggregated to compute mean confidence, "
                    "mean lift, and the number of rules supporting each.\n"
                    "- The final list, sorted by lift and confidence, is what you see above "
                    "as the recommended add-on products."
                )

# =========================================================
# TAB 5: Visual Explanations (Pipeline Diagrams)
# =========================================================

with tab5:
    st.subheader("Visual Explanations & Pipeline")

    st.markdown("### Overall Data Mining Pipeline")
    st.write(
        "This Sankey diagram shows the high-level flow of the project:\n"
        "Raw Data → Cleaning → Baskets → Frequent Itemsets → Association Rules → Recommender."
    )

    # Simple Sankey diagram of the pipeline
    labels = [
        "Raw Data",
        "Cleaned Data",
        "Baskets / Invoices",
        "Frequent Itemsets",
        "Association Rules",
        "Smart Recommender",
    ]
    # We connect each stage sequentially with unit flow
    source = [0, 1, 2, 3, 4]  # index positions in labels
    target = [1, 2, 3, 4, 5]
    value = [10, 10, 10, 10, 10]

    sankey_fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(pad=20, thickness=20, line=dict(color="black", width=0.5), label=labels),
                link=dict(source=source, target=target, value=value),
            )
        ]
    )
    sankey_fig.update_layout(title_text="End-to-End Pipeline", font_size=12)
    st.plotly_chart(sankey_fig, use_container_width=True)

    st.markdown("### How the Recommender Works (Conceptual Steps)")
    st.markdown(
        """
1. **Frequent Itemsets:**  
   - We discover which items frequently appear together in the same basket.

2. **Association Rules:**  
   - From these sets, we create rules of the form  
     _{Antecedents} → {Consequents}_  
   - Each rule has support, confidence, and lift.

3. **User Selection:**  
   - In the Smart Recommender tab, the user picks one or more base products.

4. **Rule Filtering:**  
   - We filter rules whose **antecedents** contain all of the selected products.

5. **Staple Filtering:**  
   - We compute global item support and remove the top-3 most common items 
     (e.g., BREAD, MILK) from **consequents**, so the tool doesn't just recommend trivial staples.

6. **Aggregation:**  
   - For each remaining consequent item, we calculate:
        - Average confidence across rules it appears in  
        - Average lift  
        - Number of supporting rules  

7. **Ranking & Display:**  
   - We sort by lift and confidence and show the Top-N items as recommended add-ons.
        """
    )

    st.markdown("### RFM & Segmentation Logic (Summary)")
    st.markdown(
        """
- **Recency (R):** Days since the customer's last purchase (lower is better).
- **Frequency (F):** How many transactions the customer made in the period.
- **Monetary (M):** Total amount spent.

We convert R, F, M into scores from 1–5 and sum them into an RFM_Score.
Based on this, we define segments:
- Low, Mid, High, VIP

Then we also run **K-Means** clustering on R, F, M to find numerical clusters, which can be
compared against the rule-based RFM segments.
        """
    )

    st.info(
        "Use this tab during your presentation to explain the methodology visually to your lecturer.\n"
        "You can scroll through each step and narrate the story of the project."
    )