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
# Streamlit config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Retail Intelligence Dashboard v4.2",
    page_icon="🧠",
    layout="wide",
)
st.title("🧠 Retail Intelligence Dashboard v4.2")
st.caption("Field mapping + Cleaning → Association Rules → Temporal → Segmentation → Recommender")

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = df.isna().sum()
    pct = (total / len(df) * 100).round(2) if len(df) else 0
    out = pd.DataFrame({"Missing": total, "Missing_%": pct})
    return out.sort_values("Missing", ascending=False)


def normalize_products(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def ensure_invoice_column(df: pd.DataFrame, schema: dict):
    """Ensure there is an invoice column; create SyntheticInvoice if needed."""
    inv = schema.get("invoice")
    cust = schema.get("customer")

    work = df.copy()

    if inv and inv in work.columns:
        return work, inv

    # Create synthetic invoice
    if cust and cust in work.columns:
        work["_inv_idx"] = work.groupby(cust).cumcount() // 3 + 1000
    else:
        work["_inv_idx"] = range(1000, 1000 + len(work))

    if "SyntheticInvoice" not in work.columns:
        work["SyntheticInvoice"] = "INV" + work["_inv_idx"].astype(str)

    st.warning("⚠️ No invoice column mapped — synthetic invoices created as 'SyntheticInvoice'.")
    return work, "SyntheticInvoice"


def ensure_date_column(df: pd.DataFrame, schema: dict):
    """Ensure there is a usable date column; create SyntheticDate if needed."""
    date_col = schema.get("date")
    work = df.copy()

    if date_col and date_col in work.columns:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        return work, date_col

    if "SyntheticDate" not in work.columns:
        rng = np.random.default_rng(42)
        work["SyntheticDate"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(
            rng.integers(0, 120, size=len(work)), unit="D"
        )
        st.info("ℹ️ No date column mapped — synthetic dates created as 'SyntheticDate'.")

    return work, "SyntheticDate"


def build_transactions(df: pd.DataFrame, schema: dict) -> list:
    """Build basket transactions list based on mapped columns + synthetic invoice/date if needed."""
    prod_col = schema.get("product")
    if not prod_col or prod_col not in df.columns:
        return []

    work, inv_col = ensure_invoice_column(df, schema)
    work = work.dropna(subset=[prod_col]).copy()
    work[prod_col] = normalize_products(work[prod_col])

    tx = work.groupby(inv_col)[prod_col].apply(lambda s: sorted(set(s.tolist()))).tolist()
    return [t for t in tx if len(t) > 0]


def encode_transactions(transactions: list) -> pd.DataFrame:
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
    return fi.sort_values("support", ascending=False), elapsed


def make_rules(fi: pd.DataFrame, metric: str, min_threshold: float) -> pd.DataFrame:
    if fi is None or fi.empty:
        return pd.DataFrame()
    rules = association_rules(fi, metric=metric, min_threshold=min_threshold)
    rules["antecedents_str"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules["consequents_str"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
    return rules.sort_values(["lift", "confidence", "support"], ascending=False)


def top_itemsets_chart(fi: pd.DataFrame, n=20):
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


def network_from_rules(rules: pd.DataFrame, topn=30):
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


def build_rfm(df: pd.DataFrame, schema: dict):
    cust_col = schema.get("customer")
    amt_col = schema.get("finalamount")

    if not cust_col or not amt_col:
        return pd.DataFrame()
    if cust_col not in df.columns or amt_col not in df.columns:
        return pd.DataFrame()

    work, date_col = ensure_date_column(df, schema)
    work = work[~work[date_col].isna()].copy()
    if work.empty:
        return pd.DataFrame()

    now = work[date_col].max()
    grp = (
        work.groupby(cust_col)
        .agg(
            Recency=(date_col, lambda s: (now - s.max()).days),
            Frequency=(date_col, "count"),
            Monetary=(amt_col, "sum"),
        )
        .reset_index()
        .rename(columns={cust_col: "customerid"})
    )

    if grp.empty:
        return grp

    try:
        grp["R_Score"] = pd.qcut(
            grp["Recency"].rank(method="first", ascending=True),
            5,
            labels=[5, 4, 3, 2, 1],
        )
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


def get_work_df():
    return st.session_state.get("df_clean", st.session_state["df_raw"])


# ---------------------------------------------------------
# Data source + field mapping
# ---------------------------------------------------------
st.sidebar.header("1️⃣ Upload Dataset")
uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV file to continue.")
    st.stop()

df_raw = pd.read_csv(uploaded)
st.session_state["df_raw"] = df_raw

st.subheader("Raw Data Preview")
st.dataframe(df_raw.head(20), use_container_width=True)

cols = df_raw.columns.tolist()

st.sidebar.header("2️⃣ Map Dataset Fields (Schema)")
product_col = st.sidebar.selectbox("Product Column (required)", cols)
invoice_col = st.sidebar.selectbox("Invoice Column (optional)", ["(none)"] + cols)
customer_col = st.sidebar.selectbox("Customer Column (optional)", ["(none)"] + cols)
date_col = st.sidebar.selectbox("Date Column (optional)", ["(none)"] + cols)
quantity_col = st.sidebar.selectbox("Quantity Column (optional)", ["(none)"] + cols)
finalamount_col = st.sidebar.selectbox("Final Amount / Net Sales Column (optional)", ["(none)"] + cols)
unitprice_col = st.sidebar.selectbox("Unit Price Column (optional)", ["(none)"] + cols)

schema = {
    "product": product_col,
    "invoice": None if invoice_col == "(none)" else invoice_col,
    "customer": None if customer_col == "(none)" else customer_col,
    "date": None if date_col == "(none)" else date_col,
    "quantity": None if quantity_col == "(none)" else quantity_col,
    "finalamount": None if finalamount_col == "(none)" else finalamount_col,
    "unitprice": None if unitprice_col == "(none)" else unitprice_col,
}

st.markdown("### Active Schema Mapping")
st.json(schema)

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab0, tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🧹 Data Cleaning & Prep",
        "📊 Dataset & Rules",
        "🕒 Temporal Insights",
        "👥 Customer Segmentation",
        "🛍️ Smart Recommender",
    ]
)

# ---------------------------------------------------------
# TAB 0: Data Cleaning & Prep
# ---------------------------------------------------------
with tab0:
    st.subheader("Data Cleaning & Preparation")

    st.markdown("#### Missing Value Summary")
    miss = missing_summary(df_raw)
    st.dataframe(miss, use_container_width=True)

    prod_col = schema["product"]
    q_col = schema["quantity"]
    d_col = schema["date"]
    inv_col = schema["invoice"]

    c1, c2, c3 = st.columns(3)
    drop_null_product = c1.checkbox("Drop rows with missing product", value=True)
    drop_null_date = c2.checkbox("Drop rows with missing date (if mapped)", value=False)
    remove_bad_qty = c3.checkbox("Remove rows with Quantity ≤ 0 (if mapped)", value=True)

    c4, c5 = st.columns(2)
    normalize_names = c4.checkbox("Standardize product names (uppercase/strip)", value=True)
    drop_dup_invoice_item = c5.checkbox(
        "Drop duplicate (Invoice, Product) rows (if invoice mapped)", value=True
    )

    if st.button("Run Cleaning"):
        df_clean = df_raw.copy()

        if drop_null_product and prod_col in df_clean.columns:
            df_clean = df_clean.dropna(subset=[prod_col])

        if drop_null_date and d_col and d_col in df_clean.columns:
            df_clean = df_clean.dropna(subset=[d_col])

        if remove_bad_qty and q_col and q_col in df_clean.columns:
            df_clean[q_col] = pd.to_numeric(df_clean[q_col], errors="coerce")
            df_clean = df_clean[df_clean[q_col] > 0]

        if normalize_names and prod_col in df_clean.columns:
            df_clean[prod_col] = normalize_products(df_clean[prod_col])

        if d_col and d_col in df_clean.columns:
            df_clean[d_col] = pd.to_datetime(df_clean[d_col], errors="coerce")
            df_clean["month"] = df_clean[d_col].dt.month_name()
            df_clean["dayofweek"] = df_clean[d_col].dt.day_name()

        if drop_dup_invoice_item and inv_col and inv_col in df_clean.columns and prod_col in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=[inv_col, prod_col])

        st.session_state["df_clean"] = df_clean

        st.success(f"Cleaning complete: {len(df_raw):,} → {len(df_clean):,} rows")
        st.markdown("#### Cleaned Data Preview")
        st.dataframe(df_clean.head(20), use_container_width=True)

        csv_bytes = df_clean.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download cleaned_dataset.csv",
            data=csv_bytes,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
        )

    if "df_clean" in st.session_state:
        st.info("Using the cleaned dataset for the other tabs. Re-run cleaning to update it.")


# ---------------------------------------------------------
# TAB 1: Dataset & Rules
# ---------------------------------------------------------
with tab1:
    df = get_work_df()
    st.subheader("Dataset & Association Rule Mining")

    st.markdown("#### Working Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    transactions = build_transactions(df, schema)
    st.write(f"Transactions built: **{len(transactions):,}**")
    if not transactions:
        st.error("Could not build transactions. Check product mapping; invoice will be synthetic if missing.")
        st.stop()

    with st.spinner("Encoding baskets..."):
        df_encoded = encode_transactions(transactions)

    st.write(f"Unique items: **{df_encoded.shape[1]:,}**")

    colA, colB, colC = st.columns(3)
    algo = colA.selectbox("Algorithm", ["FP-Growth", "Apriori"])
    min_support = colB.slider("Min Support", 0.001, 0.1, 0.01, step=0.001)
    max_len = colC.slider("Max Itemset Length", 2, 5, 3)

    metric = st.selectbox("Rule Metric", ["confidence", "lift"])
    min_metric = st.slider(f"Min {metric.title()}", 0.1, 1.0, 0.3, step=0.05)
    topn = st.slider("Top N to display", 10, 100, 30, step=5)

    with st.spinner(f"Running {algo}..."):
        fi, elapsed = run_mining(df_encoded, algo, min_support, max_len)

    st.success(f"Completed in {elapsed:.2f}s | Frequent itemsets: {len(fi):,}")
    rules = make_rules(fi, metric, min_metric)
    st.info(f"Rules generated: **{len(rules):,}**")

    if not fi.empty:
        fig_top = top_itemsets_chart(fi, n=topn)
        if fig_top:
            st.plotly_chart(fig_top, use_container_width=True)

    if not rules.empty:
        fig_rules = plot_rules_scatter(rules)
        if fig_rules:
            st.plotly_chart(fig_rules, use_container_width=True)

        st.markdown("#### Association Rules (Top)")
        st.dataframe(
            rules[["antecedents_str", "consequents_str", "support", "confidence", "lift"]].head(topn),
            use_container_width=True,
        )

        net = network_from_rules(rules, topn=topn)
        if net:
            st.plotly_chart(net, use_container_width=True)

    # store for other tabs if needed
    st.session_state["rules_global"] = rules


# ---------------------------------------------------------
# TAB 2: Temporal Insights
# ---------------------------------------------------------
with tab2:
    df = get_work_df()
    st.subheader("Temporal & Sequential Insights")

    df_date, date_col_use = ensure_date_column(df, schema)
    amt_col = schema.get("finalamount")

    if date_col_use:
        work = df_date.copy()
        work = work[~work[date_col_use].isna()].copy()

        if not work.empty:
            # Transactions per month
            monthly = (
                work.groupby(work[date_col_use].dt.to_period("M"))
                .size()
                .reset_index(name="tx_count")
            )
            monthly[date_col_use] = monthly[date_col_use].astype(str)
            st.markdown("#### Transactions per Month")
            if not monthly.empty:
                st.line_chart(monthly.set_index(date_col_use))

            # Day of week
            weekday = (
                work.groupby(work[date_col_use].dt.day_name())
                .size()
                .reset_index(name="tx_count")
            )
            weekday = weekday.sort_values("tx_count", ascending=False)
            st.markdown("#### Transactions by Day of Week")
            if not weekday.empty:
                weekday = weekday.rename(columns={weekday.columns[0]: "Day"})
                st.bar_chart(weekday.set_index("Day"))

            # Revenue trend
            if amt_col and amt_col in work.columns:
                amt_monthly = (
                    work.groupby(work[date_col_use].dt.to_period("M"))[amt_col]
                    .sum()
                    .reset_index()
                )
                amt_monthly[date_col_use] = amt_monthly[date_col_use].astype(str)
                st.markdown("#### Revenue per Month")
                if not amt_monthly.empty:
                    st.line_chart(amt_monthly.set_index(date_col_use))

            # Sequential bigrams
            cust_col = schema.get("customer")
            prod_col = schema.get("product")
            if cust_col and prod_col and cust_col in work.columns and prod_col in work.columns:
                seq_data = work.dropna(subset=[prod_col]).sort_values([cust_col, date_col_use])
                seq_data[prod_col] = normalize_products(seq_data[prod_col])
                seq2_counts = Counter()
                grouped = seq_data.groupby(cust_col)[prod_col].apply(list)
                for seq in grouped:
                    for i in range(len(seq) - 1):
                        seq2_counts[(seq[i], seq[i + 1])] += 1
                if seq2_counts:
                    seq2 = pd.DataFrame(
                        [(a, b, c) for (a, b), c in seq2_counts.items()],
                        columns=["From", "To", "Count"],
                    ).sort_values("Count", ascending=False).head(20)
                    st.markdown("#### Top Sequential Bigrams (A → B)")
                    st.dataframe(seq2, use_container_width=True)
        else:
            st.info("No valid dates available after cleaning/parsing.")
    else:
        st.info("No usable date column (real or synthetic) available for temporal analysis.")


# ---------------------------------------------------------
# TAB 3: Customer Segmentation
# ---------------------------------------------------------
with tab3:
    df = get_work_df()
    st.subheader("Customer Segmentation (RFM & K-Means)")

    rfm = build_rfm(df, schema)
    if rfm.empty:
        st.info(
            "RFM requires mapped customer column and final amount column. "
            "Synthetic date is used if no real date is mapped."
        )
    else:
        st.markdown("#### RFM Summary (Top 50 customers)")
        st.dataframe(rfm.head(50), use_container_width=True)

        k = st.slider("Number of K-Means clusters", 2, 6, 3)
        rfm_km, centers = kmeans_clusters(rfm, k=k)

        if not centers.empty:
            st.markdown("#### Cluster Centers (approx. original scale)")
            st.dataframe(centers, use_container_width=True)

        try:
            fig = px.scatter(
                rfm_km,
                x="Frequency",
                y="Monetary",
                color=rfm_km.get("KMeansCluster", pd.Series([0] * len(rfm_km))).astype(str),
                hover_data=["customerid", "Recency", "RFM_Score", "Segment"],
                title="Customer Segmentation Scatter",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render cluster scatter: {e}")

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
        st.markdown("#### Segment Summary")
        st.dataframe(seg_summary, use_container_width=True)


# ---------------------------------------------------------
# TAB 4: Smart Recommender
# ---------------------------------------------------------
with tab4:
    df = get_work_df()
    st.subheader("Smart Recommender Dashboard")

    df_date, date_col_use = ensure_date_column(df, schema)
    prod_col = schema.get("product")
    amt_col = schema.get("finalamount")

    # Date filter
    if date_col_use and date_col_use in df_date.columns:
        min_d, max_d = df_date[date_col_use].min(), df_date[date_col_use].max()
        if pd.isna(min_d) or pd.isna(max_d):
            df2 = df_date
        else:
            start, end = st.date_input(
                "Filter by Date Range",
                value=(min_d.date(), max_d.date()),
            )
            mask = (df_date[date_col_use] >= pd.to_datetime(start)) & (
                df_date[date_col_use] <= pd.to_datetime(end)
            )
            df2 = df_date.loc[mask].copy()
    else:
        df2 = df_date

    # Build RFM for possible segment filter
    rfm_seg = build_rfm(df2, schema)
    seg_options = ["(none)"]
    if not rfm_seg.empty:
        seg_options += ["Low", "Mid", "High", "VIP"]
    picked_seg = st.selectbox("Filter by RFM Segment (optional)", seg_options)

    # Build transactions and rules on filtered df2
    transactions2 = build_transactions(df2, schema)
    if transactions2:
        df_enc2 = encode_transactions(transactions2)
        fi2, _ = run_mining(df_enc2, "FP-Growth", 0.01, 3)
        rules2 = make_rules(fi2, "confidence", 0.3)
    else:
        rules2 = pd.DataFrame()

    if rules2.empty:
        st.warning("No rules available. Try relaxing support/confidence in Tab 1 or check mapping.")
    else:
        # Item picker
        try:
            all_items = sorted(
                set().union(*rules2["antecedents"]).union(*rules2["consequents"])
            )
        except Exception:
            all_items = sorted(
                pd.unique(
                    df2.get(prod_col, pd.Series([], dtype=str))
                    .astype(str)
                    .str.upper()
                )
            )
        selected = st.multiselect("Select base product(s)", all_items)

        if selected:
            # Optional segment filter
            if (
                picked_seg != "(none)"
                and not rfm_seg.empty
                and schema.get("customer")
                and schema["customer"] in df2.columns
            ):
                seg_customers = set(
                    rfm_seg[rfm_seg["Segment"] == picked_seg]["customerid"]
                )
                df_seg = df2[df2[schema["customer"]].isin(seg_customers)].copy()
            else:
                df_seg = df2

            # Filter rules where selected ⊆ antecedents
            sub = rules2[rules2["antecedents"].apply(lambda s: set(selected).issubset(s))]
            if sub.empty:
                st.info(
                    "No direct rules for selected base product(s). Try a single base product "
                    "or adjust mining parameters in Tab 1."
                )
            else:
                recs = (
                    sub.explode("consequents")
                    .groupby("consequents")
                    .agg(
                        mean_conf=("confidence", "mean"),
                        mean_lift=("lift", "mean"),
                        count=("confidence", "size"),
                    )
                    .sort_values(["mean_lift", "mean_conf", "count"], ascending=False)
                    .head(10)
                )
                st.markdown("#### Top Recommended Add-ons")
                st.dataframe(recs, use_container_width=True)
                st.bar_chart(recs["mean_lift"])

                # Revenue impact where rule holds
                inv_df, inv_col_use = ensure_invoice_column(df_seg, schema)
                if (
                    inv_col_use
                    and prod_col
                    and prod_col in inv_df.columns
                    and amt_col
                    and amt_col in inv_df.columns
                ):
                    inv_df[prod_col] = normalize_products(inv_df[prod_col])
                    by_inv = inv_df.groupby(inv_col_use).agg(
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
                    st.markdown("#### Estimated Revenue Impact (where rule holds)")
                    st.dataframe(impact_df, use_container_width=True)

                st.markdown("#### Explanation")
                best = list(recs.index[:3])
                st.write(
                    f"When customers buy **{', '.join(selected)}**, "
                    f"they often also purchase **{', '.join(best)}**. "
                    "High lift indicates these add-ons co-occur more frequently than random chance."
                )
