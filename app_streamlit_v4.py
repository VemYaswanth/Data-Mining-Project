
# app_streamlit_v4.py
# Omnichannel Retail Intelligence Dashboard (v4)
# Tabs: Data Cleaning | Dataset & Rules | Temporal Insights | Customer Segmentation | Smart Recommender

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

st.set_page_config(page_title="Omnichannel Retail Intelligence Dashboard", page_icon="🧹", layout="wide")
st.title("🧹 Omnichannel Retail Intelligence Dashboard (v4)")
st.caption("End-to-end: Cleaning → Association Rules → Temporal → Segmentation → Recommender")

# --------------------------------
# Helpers
# --------------------------------
@st.cache_data(show_spinner=False)
def load_csv_safely(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def detect_schema(df: pd.DataFrame) -> dict:
    cols = {c.lower(): c for c in df.columns}
    schema = {
        "product": None, "invoice": None, "customer": None, "date": None,
        "finalamount": None, "unitprice": None, "quantity": None,
        "channel": None, "store": None, "aisle": None, "month": None, "dayofweek": None
    }
    # product
    for cand in ["description", "productname", "product_name", "item"]:
        if cand in cols: schema["product"] = cols[cand]; break
    # invoice
    for cand in ["invoiceno", "invoice", "basket_id", "transaction_id"]:
        if cand in cols: schema["invoice"] = cols[cand]; break
    # customer/date
    for cand in ["customerid", "customer_id", "custid"]:
        if cand in cols: schema["customer"] = cols[cand]; break
    for cand in ["transactiondate", "invoice_date", "date", "orderdate"]:
        if cand in cols: schema["date"] = cols[cand]; break
    # numeric
    for cand in ["finalamount", "final_amount", "totalamount", "total", "amount"]:
        if cand in cols: schema["finalamount"] = cols[cand]; break
    for cand in ["unitprice", "price", "unit_price"]:
        if cand in cols: schema["unitprice"] = cols[cand]; break
    for cand in ["quantity", "qty"]:
        if cand in cols: schema["quantity"] = cols[cand]; break
    # dims
    if "channel" in cols: schema["channel"] = cols["channel"]
    for cand in ["storename", "store_name", "country"]:
        if cand in cols: schema["store"] = cols[cand]; break
    if "aisle" in cols: schema["aisle"] = cols["aisle"]
    if "month" in cols: schema["month"] = cols["month"]
    for cand in ["dayofweek", "day_of_week"]:
        if cand in cols: schema["dayofweek"] = cols[cand]; break
    return schema

def normalize_products(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()

def ensure_datetime(df: pd.DataFrame, col: str):
    try:
        return pd.to_datetime(df[col])
    except Exception:
        return pd.to_datetime(df[col], errors="coerce")

def build_transactions(df: pd.DataFrame, schema: dict) -> list:
    prod = schema["product"]
    if not prod: return []
    work = df.dropna(subset=[prod]).copy()
    work[prod] = normalize_products(work[prod])
    if schema["invoice"] and schema["invoice"] in work.columns:
        key = schema["invoice"]
        tx = work.groupby(key)[prod].apply(lambda s: sorted(set(s.tolist()))).tolist()
    elif schema["customer"] and schema["date"] and schema["customer"] in work.columns and schema["date"] in work.columns:
        try:
            work["_date"] = pd.to_datetime(work[schema["date"]]).dt.date
        except Exception:
            work["_date"] = work[schema["date"]].astype(str)
        key = ["_date", schema["customer"]]
        work["_tx"] = work[key].astype(str).agg("|".join, axis=1)
        tx = work.groupby("_tx")[prod].apply(lambda s: sorted(set(s.tolist()))).tolist()
    else:
        tx = [[x] for x in work[prod].tolist()]
    return [t for t in tx if len(t)>0]

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
    if fi is None or fi.empty: return pd.DataFrame()
    rules = association_rules(fi, metric=metric, min_threshold=min_threshold)
    rules["antecedents_str"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules["consequents_str"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
    return rules.sort_values(["lift","confidence","support"], ascending=False)

def plot_rules_scatter(rules: pd.DataFrame):
    if rules.empty: return None
    return px.scatter(rules, x="support", y="confidence", size="lift",
                      hover_data=["antecedents_str","consequents_str"],
                      title="Rules: Support vs Confidence (size=Lift)")

def top_itemsets_chart(fi: pd.DataFrame, n=20):
    if fi.empty: return None
    tmp = fi.head(n).copy()
    tmp["itemset_str"] = tmp["itemsets"].apply(lambda s: ", ".join(sorted(list(s))))
    return px.bar(tmp[::-1], x="support", y="itemset_str", orientation="h", title=f"Top {n} Frequent Itemsets")

def network_from_rules(rules: pd.DataFrame, topn=30):
    if rules.empty: return None
    sub = rules.head(topn).copy()
    G = nx.DiGraph()
    for _, r in sub.iterrows():
        for a in r["antecedents"]:
            for c in r["consequents"]:
                G.add_edge(a, c, lift=r["lift"], confidence=r["confidence"])
    pos = nx.spring_layout(G, seed=42, k=0.8)
    edge_x, edge_y = [], []
    for (u,v) in G.edges():
        x0,y0 = pos[u]; x1,y1 = pos[v]
        edge_x += [x0,x1,None]; edge_y += [y0,y1,None]
    node_x, node_y, labels = [], [], []
    for n,(x,y) in pos.items():
        node_x.append(x); node_y.append(y); labels.append(n)
    fig = go.Figure(data=[
        go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none", line=dict(width=1)),
        go.Scatter(x=node_x, y=node_y, mode="markers+text", text=labels, textposition="top center")
    ])
    fig.update_layout(title="Association Network (Top Rules by Lift)", showlegend=False, margin=dict(l=10,r=10,t=40,b=10))
    return fig

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = df.isna().sum()
    pct = (total / len(df) * 100).round(2) if len(df) else 0
    out = pd.DataFrame({"Missing": total, "Missing_%": pct})
    return out.sort_values("Missing", ascending=False)

def build_rfm(df: pd.DataFrame, schema: dict):
    cust = schema["customer"]; date = schema["date"]; amt = schema["finalamount"]
    if not (cust and date and amt): return pd.DataFrame()
    if cust not in df.columns or date not in df.columns or amt not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work[date] = ensure_datetime(work, date)
    work = work[~work[date].isna()].copy()
    if work.empty: return pd.DataFrame()
    now = work[date].max()
    grp = work.groupby(cust).agg(
        Recency=(date, lambda s: (now - s.max()).days),
        Frequency=(date, 'count'),
        Monetary=(amt, 'sum')
    ).reset_index().rename(columns={cust:"customerid"})
    # scoring (1-5)
    try:
        grp["R_Score"] = pd.qcut(grp["Recency"].rank(method="first", ascending=True), 5, labels=[5,4,3,2,1])
        grp["F_Score"] = pd.qcut(grp["Frequency"].rank(method="first", ascending=True), 5, labels=[1,2,3,4,5])
        grp["M_Score"] = pd.qcut(grp["Monetary"].rank(method="first", ascending=True), 5, labels=[1,2,3,4,5])
    except Exception:
        grp["R_Score"] = pd.cut(grp["Recency"], bins=5, labels=[5,4,3,2,1], include_lowest=True)
        grp["F_Score"] = pd.cut(grp["Frequency"], bins=5, labels=[1,2,3,4,5], include_lowest=True)
        grp["M_Score"] = pd.cut(grp["Monetary"], bins=5, labels=[1,2,3,4,5], include_lowest=True)
    grp["RFM_Score"] = grp["R_Score"].astype(int) + grp["F_Score"].astype(int) + grp["M_Score"].astype(int)
    grp["Segment"] = pd.cut(grp["RFM_Score"], bins=[0,6,9,12,15], labels=["Low","Mid","High","VIP"], include_lowest=True)
    return grp

def kmeans_clusters(df_rfm: pd.DataFrame, k=3):
    try:
        feats = df_rfm[["Recency","Frequency","Monetary"]].fillna(0).copy()
        scaler = StandardScaler()
        X = scaler.fit_transform(feats)
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        df_rfm = df_rfm.copy()
        df_rfm["KMeansCluster"] = labels
        centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=["Recency","Frequency","Monetary"])
        centers["Cluster"] = centers.index
        return df_rfm, centers
    except Exception:
        return df_rfm, pd.DataFrame()

# --------------------------------
# Sidebar: Data Source
# --------------------------------
st.sidebar.header("Data Source")
choice = st.sidebar.selectbox("Choose dataset", ["Upload CSV", "chain_cleaned.csv", "grocery_cleaned.csv"])
upl = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if choice != "Upload CSV" and upl is None:
    df_raw = load_csv_safely("/mnt/data/" + choice)
else:
    df_raw = pd.read_csv(upl) if upl is not None else pd.DataFrame()

if df_raw.empty:
    st.warning("Please upload or select a dataset.")
    st.stop()

schema = detect_schema(df_raw)

# Tab layout
tab0, tab1, tab2, tab3, tab4 = st.tabs(["🧹 Data Cleaning & Prep", "📊 Dataset & Rules", "🕒 Temporal Insights", "👥 Customer Segmentation", "🛍️ Smart Recommender"])

# --------------------------------
# TAB 0: Data Cleaning & Preparation
# --------------------------------
with tab0:
    st.subheader("Preview Raw Dataset")
    st.dataframe(df_raw.head(30), use_container_width=True)

    st.markdown("### Missing Value Summary")
    miss = missing_summary(df_raw)
    st.dataframe(miss, use_container_width=True)

    # Cleaning controls
    st.markdown("### Cleaning Options")
    prod_col = schema["product"]
    date_col = schema["date"]
    qty_col = schema["quantity"]
    inv_col = schema["invoice"]

    c1, c2, c3 = st.columns(3)
    drop_null_product = c1.checkbox("Drop rows with missing product", value=True)
    drop_null_date = c2.checkbox("Drop rows with missing date (if present)", value=False)
    remove_bad_qty = c3.checkbox("Remove rows with Quantity <= 0 (if present)", value=True)

    c4, c5 = st.columns(2)
    normalize_names = c4.checkbox("Standardize product names (uppercase/strip)", value=True)
    drop_dup_invoice_item = c5.checkbox("Drop duplicate (Invoice, Product) rows", value=True)

    # Run cleaning
    st.markdown("### Run Cleaning")
    if st.button("Clean Data"):
        df_clean = df_raw.copy()

        if drop_null_product and prod_col and prod_col in df_clean.columns:
            df_clean = df_clean.dropna(subset=[prod_col])

        if drop_null_date and date_col and date_col in df_clean.columns:
            df_clean = df_clean.dropna(subset=[date_col])

        if remove_bad_qty and qty_col and qty_col in df_clean.columns:
            df_clean[qty_col] = pd.to_numeric(df_clean[qty_col], errors="coerce")
            df_clean = df_clean[df_clean[qty_col] > 0]

        if normalize_names and prod_col and prod_col in df_clean.columns:
            df_clean[prod_col] = normalize_products(df_clean[prod_col])

        if date_col and date_col in df_clean.columns:
            df_clean[date_col] = ensure_datetime(df_clean, date_col)
            df_clean["month"] = df_clean[date_col].dt.month_name()
            df_clean["dayofweek"] = df_clean[date_col].dt.day_name()

        if drop_dup_invoice_item and inv_col and inv_col in df_clean.columns and prod_col and prod_col in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=[inv_col, prod_col])

        st.success(f"Cleaning complete: {len(df_raw):,} → {len(df_clean):,} rows")
        st.markdown("**Preview (Cleaned)**")
        st.dataframe(df_clean.head(30), use_container_width=True)

        st.markdown("### Download Cleaned CSV")
        csv_bytes = df_clean.to_csv(index=False).encode("utf-8")
        st.download_button("Download cleaned_dataset.csv", data=csv_bytes, file_name="cleaned_dataset.csv", mime="text/csv")

        st.session_state["df_clean"] = df_clean

    if "df_clean" in st.session_state:
        st.info("Using the cleaned dataset for all other tabs. You can re-run cleaning anytime.")

def get_df_work():
    return st.session_state.get("df_clean", df_raw)

# --------------------------------
# TAB 1: Dataset & Rules
# --------------------------------
with tab1:
    df = get_df_work()
    st.subheader("Dataset Overview & Association Mining")
    st.write("**Detected schema:**", schema)
    st.dataframe(df.head(30), use_container_width=True)

    transactions = build_transactions(df, schema)
    st.write(f"Transactions built: **{len(transactions):,}**")
    if not transactions:
        st.error("Could not build transactions. Check product/invoice/customer/date columns.")
        st.stop()

    with st.spinner("Encoding baskets..."):
        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)
    st.write(f"Unique items: **{df_encoded.shape[1]:,}**")

    colA, colB, colC = st.columns(3)
    algo = colA.selectbox("Algorithm", ["FP-Growth", "Apriori"])
    min_support = colB.slider("Min Support", 0.001, 0.1, 0.01, step=0.001)
    max_len = colC.slider("Max Itemset Length", 2, 5, 3)

    metric = st.selectbox("Rule Metric", ["confidence", "lift"])
    min_metric = st.slider(f"Min {metric.title()}", 0.1, 1.0, 0.3, step=0.05)
    topn = st.slider("Top N to display", 10, 100, 30, step=5)

    with st.spinner(f"Running {algo}..."):
        fi = pd.DataFrame(); elapsed = 0.0
        try:
            fi, elapsed = run_mining(df_encoded, algo, min_support, max_len)
        except Exception as e:
            st.error(f"Mining error: {e}")
    st.success(f"Completed in {elapsed:.2f}s | Frequent itemsets: {len(fi):,}")

    rules = make_rules(fi, metric, min_metric)
    st.info(f"Rules generated: **{len(rules):,}**")

    if not fi.empty:
        fig_top = top_itemsets_chart(fi, n=topn)
        if fig_top: st.plotly_chart(fig_top, use_container_width=True)

    if not rules.empty:
        fig_rules = plot_rules_scatter(rules)
        if fig_rules: st.plotly_chart(fig_rules, use_container_width=True)

        st.dataframe(rules[["antecedents_str","consequents_str","support","confidence","lift"]].head(topn), use_container_width=True)

        net = network_from_rules(rules, topn=topn)
        if net: st.plotly_chart(net, use_container_width=True)

# --------------------------------
# TAB 2: Temporal Insights
# --------------------------------
with tab2:
    df = get_df_work()
    st.subheader("Temporal & Sequential Insights")
    date_col = schema["date"]
    amt_col = schema["finalamount"]
    if date_col and date_col in df.columns:
        work = df.copy()
        work[date_col] = ensure_datetime(work, date_col)
        work = work[~work[date_col].isna()].copy()
        if not work.empty:
            monthly = work.groupby(work[date_col].dt.to_period("M")).size().reset_index(name="tx_count")
            monthly[date_col] = monthly[date_col].astype(str)
            st.markdown("**Transactions per Month**")
            if not monthly.empty:
                st.line_chart(monthly.set_index(date_col))

            weekday = work.groupby(work[date_col].dt.day_name()).size().reset_index(name="tx_count")
            weekday = weekday.sort_values("tx_count", ascending=False)
            st.markdown("**Transactions by Day of Week**")
            if not weekday.empty:
                weekday = weekday.rename(columns={weekday.columns[0]:"Day"})
                st.bar_chart(weekday.set_index("Day"))

            if amt_col and amt_col in work.columns:
                amt_monthly = work.groupby(work[date_col].dt.to_period("M"))[amt_col].sum().reset_index()
                amt_monthly[date_col] = amt_monthly[date_col].astype(str)
                st.markdown("**Revenue per Month**")
                if not amt_monthly.empty:
                    st.line_chart(amt_monthly.set_index(date_col))

            cust_col = schema["customer"]
            prod_col = schema["product"]
            if cust_col and prod_col and cust_col in work.columns and prod_col in work.columns:
                seq_data = work.dropna(subset=[prod_col]).sort_values([cust_col, date_col])
                seq_data[prod_col] = normalize_products(seq_data[prod_col])
                seq2_counts = Counter()
                grouped = seq_data.groupby(cust_col)[prod_col].apply(list)
                for seq in grouped:
                    for i in range(len(seq)-1):
                        seq2_counts[(seq[i], seq[i+1])] += 1
                if seq2_counts:
                    seq2 = pd.DataFrame([(a,b,c) for (a,b),c in seq2_counts.items()], columns=["From","To","Count"]).sort_values("Count", ascending=False).head(20)
                    st.markdown("**Top Sequential Bigrams (A → B)**")
                    st.dataframe(seq2, use_container_width=True)
        else:
            st.info("Date parsing produced no valid rows for temporal analysis.")
    else:
        st.info("No date column detected; temporal insights unavailable for this dataset.")

# --------------------------------
# TAB 3: Customer Segmentation
# --------------------------------
with tab3:
    df = get_df_work()
    st.subheader("Customer Segmentation (RFM & K-Means)")
    rfm = build_rfm(df, schema)
    if rfm.empty:
        st.info("Need customer, date, and final amount columns for RFM segmentation.")
    else:
        st.markdown("**RFM Summary (Top 50)**")
        st.dataframe(rfm.head(50), use_container_width=True)

        k = st.slider("Number of K-Means clusters", 2, 6, 3)
        rfm_km, centers = kmeans_clusters(rfm, k=k)
        if not centers.empty:
            st.markdown("**Cluster Centers (approximate original scale)**")
            st.dataframe(centers, use_container_width=True)

        try:
            fig = px.scatter(rfm_km, x="Frequency", y="Monetary",
                             color=rfm_km.get("KMeansCluster", pd.Series([0]*len(rfm_km))).astype(str),
                             hover_data=["customerid","Recency","RFM_Score","Segment"],
                             title="Customer Segmentation Scatter")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render cluster scatter: {e}")

        seg_summary = rfm_km.groupby("Segment", dropna=False).agg(Customers=("customerid","count"),
                                                    AvgSpend=("Monetary","mean"),
                                                    AvgFreq=("Frequency","mean"),
                                                    AvgRecency=("Recency","mean")).round(2)
        st.markdown("**Segment Summary**")
        st.dataframe(seg_summary, use_container_width=True)

# --------------------------------
# TAB 4: Smart Recommender
# --------------------------------
with tab4:
    df = get_df_work()
    st.subheader("Smart Recommender Dashboard")
    st.write("Use mined rules to recommend add-on products. Filter by time window and (optionally) segment.")

    date_col = schema["date"]
    prod_col = schema["product"]
    if date_col and prod_col and date_col in df.columns and prod_col in df.columns:
        df2 = df.copy()
        df2[date_col] = ensure_datetime(df2, date_col)
        min_d, max_d = df2[date_col].min(), df2[date_col].max()
        if pd.isna(min_d) or pd.isna(max_d):
            df2 = df2
        else:
            start, end = st.date_input("Date range", value=(min_d.date(), max_d.date()))
            mask = (df2[date_col] >= pd.to_datetime(start)) & (df2[date_col] <= pd.to_datetime(end))
            df2 = df2.loc[mask].copy()
    else:
        df2 = df.copy()

    seg_options = ["(none)"]
    rfm_cache = build_rfm(df2, schema) if not df2.empty else pd.DataFrame()
    if not rfm_cache.empty:
        seg_options += ["Low","Mid","High","VIP"]
    picked_seg = st.selectbox("Filter by RFM Segment (optional)", seg_options)

    try:
        transactions2 = build_transactions(df2, schema)
        enc2 = encode_transactions(transactions2) if transactions2 else pd.DataFrame()
        fi2, _ = run_mining(enc2, "FP-Growth", 0.01, 3) if not enc2.empty else (pd.DataFrame(), 0)
        rules2 = make_rules(fi2, "confidence", 0.3) if not fi2.empty else pd.DataFrame()
    except Exception:
        rules2 = pd.DataFrame()

    if rules2.empty:
        st.warning("No rules available. Clean the data in Tab 0 and/or adjust mining thresholds in Tab 1.")
    else:
        try:
            all_items = sorted(set().union(*rules2['antecedents']).union(*rules2['consequents']))
        except Exception:
            all_items = sorted(pd.unique(df2.get(prod_col, pd.Series([], dtype=str)).astype(str).str.upper()))
        selected = st.multiselect("Select base product(s)", all_items)

        if selected:
            if picked_seg != "(none)" and not rfm_cache.empty and schema["customer"] and schema["customer"] in df2.columns:
                seg_customers = set(rfm_cache[rfm_cache["Segment"]==picked_seg]["customerid"])
                df_seg = df2[df2[schema["customer"]].isin(seg_customers)].copy()
            else:
                df_seg = df2

            sub = rules2[rules2["antecedents"].apply(lambda s: set(selected).issubset(s))]
            if sub.empty:
                st.info("No direct rules for selected base product(s). Try selecting one item or lowering thresholds in Tab 1.")
            else:
                recs = (sub.explode("consequents")
                          .groupby("consequents")
                          .agg(mean_conf=("confidence","mean"), mean_lift=("lift","mean"), count=("confidence","size"))
                          .sort_values(["mean_lift","mean_conf","count"], ascending=False)
                          .head(10))
                st.markdown("**Top Recommended Add-ons**")
                st.dataframe(recs, use_container_width=True)
                st.bar_chart(recs["mean_lift"])

                inv_col = schema["invoice"]
                if prod_col in df_seg.columns and schema["finalamount"] in df_seg.columns and inv_col and inv_col in df_seg.columns:
                    df_seg[prod_col] = normalize_products(df_seg[prod_col])
                    by_inv = df_seg.groupby(inv_col).agg(
                        items=(prod_col, lambda s: set(s.tolist())),
                        revenue=(schema["finalamount"], "sum")
                    )
                    impacts = []
                    for item in recs.index:
                        mask = by_inv["items"].apply(lambda s: set(selected).issubset(s) and (item in s))
                        impacts.append({"item": item, "support_in_invoices": int(mask.sum()), "revenue_sum": by_inv.loc[mask, "revenue"].sum()})
                    impact_df = pd.DataFrame(impacts).sort_values("revenue_sum", ascending=False)
                    st.markdown("**Estimated Revenue Impact (where rule holds)**")
                    st.dataframe(impact_df, use_container_width=True)

                st.markdown("**Explanation**")
                best = list(recs.index[:3])
                st.write(f"When customers buy **{', '.join(selected)}**, they often also purchase **{', '.join(best)}**. High lift indicates these add-ons co-occur more frequently than random chance.")
