import streamlit as st
import pandas as pd
import numpy as np
import time
from collections import Counter

import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Universal Product Recommender v5", page_icon="🧠", layout="wide")
st.title("🧠 Universal Product Recommender v5 (Zero-Fail)")
st.caption("Upload ANY CSV → auto-detects schema → mines rules → suggests products.")


# ------------------------- Utilities -------------------------
def normalize_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def detect_product_column(df: pd.DataFrame) -> str:
    """Pick a text-like column with many unique values but also repeats."""
    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if not obj_cols:
        # no text columns → synth product later
        return None

    scores = []
    n = len(df)
    for col in obj_cols:
        nun = df[col].nunique(dropna=True)
        # ignore almost-unique IDs and almost-constant columns
        if nun <= 5:
            continue
        if nun >= 0.95 * n:
            continue
        # score: more unique + more non-null
        scores.append((col, nun / n))

    if not scores:
        return None
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0]


def detect_invoice_column(df: pd.DataFrame, product_col: str | None) -> str | None:
    """Pick a column with repeated values that can group lines into baskets."""
    candidates = []
    n = len(df)
    for col in df.columns:
        if col == product_col:
            continue
        nun = df[col].nunique(dropna=True)
        if nun <= 1 or nun >= n:  # constant or all unique
            continue
        # average group size
        avg_group = n / nun if nun > 0 else 0
        if avg_group < 1.5:
            continue
        candidates.append((col, avg_group))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def detect_customer_column(df: pd.DataFrame, used_cols: set) -> str | None:
    """Pick a high-cardinality column not already used."""
    n = len(df)
    best = None
    best_score = -1
    for col in df.columns:
        if col in used_cols:
            continue
        nun = df[col].nunique(dropna=True)
        if nun <= 5:
            continue
        ratio = nun / n
        # want moderately high cardinality, not almost unique
        if 0.05 <= ratio <= 0.9 and ratio > best_score:
            best = col
            best_score = ratio
    return best


def detect_date_column(df: pd.DataFrame) -> str | None:
    """Try to parse each column as date; pick best."""
    best = None
    best_rate = 0
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            ok = parsed.notna().mean()
            if ok > 0.5 and ok > best_rate:
                best = col
                best_rate = ok
        except Exception:
            continue
    return best


def detect_amount_column(df: pd.DataFrame) -> str | None:
    """Try to pick a numeric 'amount' column by name + variance."""
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:
        return None

    # name heuristics first
    preferred = []
    for col in num_cols:
        cname = col.lower()
        if any(key in cname for key in ["final", "total", "amount", "sales", "revenue", "net"]):
            preferred.append(col)
    if preferred:
        return preferred[0]

    # fallback: highest variance numeric
    variances = []
    for col in num_cols:
        variances.append((col, df[col].var()))
    variances.sort(key=lambda x: (x[1] if pd.notna(x[1]) else 0), reverse=True)
    return variances[0][0] if variances else None


def ensure_synthetic_columns(df: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, dict]:
    """Fill missing product/invoice/customer/date with synthetic columns so nothing fails."""
    work = df.copy()
    n = len(work)

    # Product
    if schema["product"] is None:
        work["SyntheticProduct"] = "ITEM_" + work.index.astype(str)
        schema["product"] = "SyntheticProduct"
        st.warning("No obvious product column found → using SyntheticProduct.")
    else:
        work[schema["product"]] = normalize_text(work[schema["product"]])

    # Invoice / basket
    if schema["invoice"] is None:
        work["SyntheticInvoice"] = "INV_" + (work.index // 3).astype(str)
        schema["invoice"] = "SyntheticInvoice"
        st.info("No invoice-like column found → grouping rows in chunks of 3 as SyntheticInvoice.")

    # Customer
    if schema["customer"] is None:
        work["SyntheticCustomer"] = "CUST_" + (work.index % max(n, 1)).astype(str)
        schema["customer"] = "SyntheticCustomer"
        st.info("No customer-like column found → SyntheticCustomer created.")

    # Date
    if schema["date"] is None:
        rng = np.random.default_rng(42)
        work["SyntheticDate"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(
            rng.integers(0, 120, size=n),
            unit="D",
        )
        schema["date"] = "SyntheticDate"
        st.info("No usable date column found → SyntheticDate created.")
    else:
        work[schema["date"]] = pd.to_datetime(work[schema["date"]], errors="coerce")
        if work[schema["date"]].notna().sum() == 0:
            rng = np.random.default_rng(42)
            work["SyntheticDate"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(
                rng.integers(0, 120, size=n),
                unit="D",
            )
            schema["date"] = "SyntheticDate"
            st.info("Mapped date column had no valid dates → SyntheticDate created.")

    # Amount (optional)
    if schema["amount"] is None:
        # use 1 for all if nothing numeric → segmentation still possible but meaningless
        work["SyntheticAmount"] = 1.0
        schema["amount"] = "SyntheticAmount"
        st.info("No suitable numeric 'amount' column found → SyntheticAmount = 1 used.")
    return work, schema


def build_transactions(df: pd.DataFrame, schema: dict) -> list[list[str]]:
    g = df.groupby(schema["invoice"])[schema["product"]].apply(lambda s: sorted(set(s.tolist())))
    tx = [t for t in g.tolist() if len(t) > 0]
    return tx


def mine_rules_from_transactions(transactions: list[list[str]]):
    if not transactions:
        return pd.DataFrame(), pd.DataFrame(), 0.0

    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    t0 = time.time()
    fi = fpgrowth(df_encoded, min_support=0.01, use_colnames=True, max_len=3)
    elapsed = time.time() - t0

    if fi.empty:
        return df_encoded, pd.DataFrame(), elapsed

    rules = association_rules(fi, metric="confidence", min_threshold=0.3)
    if rules.empty:
        return df_encoded, pd.DataFrame(), elapsed

    rules["antecedents_str"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules["consequents_str"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False)

    return df_encoded, rules, elapsed


def build_rfm(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
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

    try:
        grp["R_Score"] = pd.qcut(grp["Recency"].rank(method="first", ascending=True), 5, labels=[5, 4, 3, 2, 1])
        grp["F_Score"] = pd.qcut(grp["Frequency"].rank(method="first", ascending=True), 5, labels=[1, 2, 3, 4, 5])
        grp["M_Score"] = pd.qcut(grp["Monetary"].rank(method="first", ascending=True), 5, labels=[1, 2, 3, 4, 5])
    except Exception:
        grp["R_Score"] = pd.cut(grp["Recency"], bins=5, labels=[5, 4, 3, 2, 1], include_lowest=True)
        grp["F_Score"] = pd.cut(grp["Frequency"], bins=5, labels=[1, 2, 3, 4, 5], include_lowest=True)
        grp["M_Score"] = pd.cut(grp["Monetary"], bins=5, labels=[1, 2, 3, 4, 5], include_lowest=True)

    grp["RFM_Score"] = grp["R_Score"].astype(int) + grp["F_Score"].astype(int) + grp["M_Score"].astype(int)
    grp["Segment"] = pd.cut(
        grp["RFM_Score"],
        bins=[0, 6, 9, 12, 15],
        labels=["Low", "Mid", "High", "VIP"],
        include_lowest=True,
    )
    return grp


def rules_to_recommendations(rules: pd.DataFrame, base_items: list[str], topn: int = 10) -> pd.DataFrame:
    if rules.empty or not base_items:
        return pd.DataFrame()

    sub = rules[rules["antecedents"].apply(lambda s: set(base_items).issubset(s))]
    if sub.empty:
        return pd.DataFrame()

    recs = (
        sub.explode("consequents")
        .groupby("consequents")
        .agg(
            mean_conf=("confidence", "mean"),
            mean_lift=("lift", "mean"),
            count=("confidence", "size"),
        )
        .sort_values(["mean_lift", "mean_conf", "count"], ascending=False)
        .head(topn)
    )
    recs.index = recs.index.astype(str)
    return recs


def plot_top_itemsets(fi: pd.DataFrame, n: int = 20):
    if fi.empty:
        return None
    tmp = fi.head(n).copy()
    tmp["itemset_str"] = tmp["itemsets"].apply(lambda s: ", ".join(sorted(list(s))))
    fig = px.bar(tmp[::-1], x="support", y="itemset_str", orientation="h", title=f"Top {n} Frequent Itemsets")
    return fig


def plot_rules_scatter(rules: pd.DataFrame):
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


def build_association_network(rules: pd.DataFrame, topn: int = 30):
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
            go.Scatter(x=node_x, y=node_y, mode="markers+text", text=labels, textposition="top center"),
        ]
    )
    fig.update_layout(
        title="Association Network (Top Rules by Lift)",
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ------------------------- App logic -------------------------
st.subheader("Step 1: Upload any CSV")

uploaded = st.file_uploader("Upload a dataset (.csv)", type=["csv"])
if uploaded is None:
    st.info("Waiting for a CSV file…")
    st.stop()

df_raw = pd.read_csv(uploaded)
st.write(f"Rows: **{len(df_raw):,}**, Columns: **{len(df_raw.columns):,}**")
st.dataframe(df_raw.head(20), use_container_width=True)

# Auto-detect schema
st.subheader("Step 2: Auto-detected schema")

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

st.json(schema)

df_work, schema = ensure_synthetic_columns(df_raw, schema)

st.markdown("**After synthetic fixes, active schema is:**")
st.json(schema)

# Build transactions and mine rules
st.subheader("Step 3: Market Basket Mining & Product Suggestions")

transactions = build_transactions(df_work, schema)
st.write(f"Built **{len(transactions):,}** transactions (baskets).")

if not transactions:
    st.error("Unexpected: no transactions built even after synthetic columns. (This should basically never happen.)")
    st.stop()

df_encoded, rules, elapsed = mine_rules_from_transactions(transactions)

if rules.empty:
    st.warning("No strong rules found at default thresholds. (Dataset may be tiny or extremely sparse.)")
else:
    st.success(f"Mined **{len(rules):,}** association rules in {elapsed:.2f}s.")

    # Show top rules
    st.markdown("### Top Association Rules")
    st.dataframe(
        rules[["antecedents_str", "consequents_str", "support", "confidence", "lift"]].head(30),
        use_container_width=True,
    )

    # Scatter plot
    fig_rules = plot_rules_scatter(rules)
    if fig_rules:
        st.plotly_chart(fig_rules, use_container_width=True)

    # Network
    fig_net = build_association_network(rules, topn=30)
    if fig_net:
        st.plotly_chart(fig_net, use_container_width=True)

    # Simple recommender UI (minimal)
    st.markdown("### Product-Based Suggestions")

    # Suggest from frequent items or from rules
    try:
        # items appearing in rules
        all_items = sorted(
            set().union(*rules["antecedents"]).union(*rules["consequents"])
        )
    except Exception:
        all_items = sorted(df_encoded.columns.tolist())

    if all_items:
        base_item = st.selectbox("Pick a product to get suggested add-ons:", all_items)
        if base_item:
            recs = rules_to_recommendations(rules, [base_item], topn=10)
            if recs.empty:
                st.info("No direct cross-sell rules for that product. Try another product.")
            else:
                st.write(f"**Recommended add-ons when customers buy `{base_item}`:**")
                st.dataframe(recs, use_container_width=True)
                st.bar_chart(recs["mean_lift"])
    else:
        st.info("No items available to recommend from. (Check if dataset is too small.)")

# Optional: basic temporal + RFM overview (automatic, no controls)
st.subheader("Step 4: Auto Temporal & Customer Insights (optional)")

df_temp = df_work.copy()
date_col = schema["date"]
amt_col = schema["amount"]
cust_col = schema["customer"]
prod_col = schema["product"]

# Temporal overview
if date_col in df_temp.columns:
    df_temp = df_temp.dropna(subset=[date_col])
    if not df_temp.empty:
        # Monthly trend
        monthly = df_temp.groupby(df_temp[date_col].dt.to_period("M")).size().reset_index(name="tx_count")
        monthly[date_col] = monthly[date_col].astype(str)
        if not monthly.empty:
            st.markdown("**Transactions per Month (auto)**")
            st.line_chart(monthly.set_index(date_col))

        # Weekday
        weekday = df_temp.groupby(df_temp[date_col].dt.day_name()).size().reset_index(name="tx_count")
        if not weekday.empty:
            weekday = weekday.sort_values("tx_count", ascending=False)
            weekday = weekday.rename(columns={weekday.columns[0]: "Day"})
            st.markdown("**Transactions by Day of Week (auto)**")
            st.bar_chart(weekday.set_index("Day"))

        # Revenue trend if amount
        if amt_col in df_temp.columns:
            amt_monthly = (
                df_temp.groupby(df_temp[date_col].dt.to_period("M"))[amt_col]
                .sum()
                .reset_index()
            )
            amt_monthly[date_col] = amt_monthly[date_col].astype(str)
            if not amt_monthly.empty:
                st.markdown("**Revenue per Month (auto)**")
                st.line_chart(amt_monthly.set_index(date_col))

# Simple RFM summary
rfm = build_rfm(df_work, schema)
if not rfm.empty:
    st.markdown("**Top 20 Customers by Monetary (auto RFM)**")
    st.dataframe(
        rfm.sort_values("Monetary", ascending=False).head(20),
        use_container_width=True,
    )
