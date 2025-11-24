# ----------------------------------------------------------
# Retail Intelligence Dashboard v5.1
# With full comments and no "choose dataset" option
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

# ----------------------------------------------------------
# Streamlit Config
# ----------------------------------------------------------
st.set_page_config(
    page_title="Retail Intelligence Dashboard v5.1",
    page_icon="🛒",
    layout="wide",
)
st.title("🧠 Retail Intelligence Dashboard v5.1")
st.caption("Data Cleaning → Patterns → Temporal → Segmentation → Recommender → Visual Explanations")


# ==========================================================
# Helper Functions — Data Processing & Detection
# ==========================================================

def normalize_products(s: pd.Series) -> pd.Series:
    """Normalize product names."""
    return s.astype(str).str.upper().str.strip()


def detect_product_column(df):
    """Find best text column resembling product name."""
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    if not len(obj_cols): return None
    scores = []
    n = len(df)
    for col in obj_cols:
        nun = df[col].nunique()
        if nun <= 5 or nun >= 0.95*n: 
            continue
        scores.append((col, nun/n))
    if not scores: return None
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0]


def detect_invoice_column(df, product_col):
    """Find column that groups items into baskets."""
    candidates = []
    n = len(df)
    for col in df.columns:
        if col == product_col: continue
        nun = df[col].nunique()
        if nun <= 1 or nun >= n: continue
        avg_group = n / nun
        if avg_group >= 1.5:
            candidates.append((col, avg_group))
    if not candidates: return None
    return sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]


def detect_customer_column(df, used):
    """Find customer-like column."""
    n = len(df)
    best, bestscore = None, -1
    for col in df.columns:
        if col in used: continue
        nun = df[col].nunique()
        ratio = nun/n
        if 0.05 <= ratio <= 0.95 and ratio > bestscore:
            best, bestscore = col, ratio
    return best


def detect_date_column(df):
    """Find best date column."""
    best, best_rate = None, 0
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            rate = parsed.notna().mean()
            if rate > 0.5 and rate > best_rate:
                best, best_rate = col, rate
        except:
            continue
    return best


def detect_amount_column(df):
    """Find amount-like column."""
    nums = df.select_dtypes(include=["number"]).columns
    if not len(nums): return None
    prefer = [c for c in nums if any(k in c.lower() for k in ["amount","total","sales","revenue"])]
    if prefer: return prefer[0]
    return nums[0]


def create_realistic_invoices(df, schema):
    """Generate synthetic realistic invoices (basket size 2–7)."""
    rng = np.random.default_rng(42)
    work = df.copy()

    if schema["customer"] in df.columns:
        inv_ids = []
        counter = 0
        for _, g in work.groupby(schema["customer"]):
            idx = g.index.tolist()
            sizes = rng.integers(2,7,len(idx))
            i=0
            while i < len(idx):
                size = sizes[i]
                for _ in range(size):
                    if i>=len(idx): break
                    inv_ids.append((idx[i], counter))
                    i+=1
                counter+=1
        work["SyntheticInvoice"] = work.index.map(dict(inv_ids))
        schema["invoice"]="SyntheticInvoice"
        return work, schema

    # No customer column → global
    idx = work.index.tolist()
    sizes = rng.integers(2,7,len(idx))
    inv_ids=[]
    counter=0
    i=0
    while i < len(idx):
        size=sizes[i]
        for _ in range(size):
            if i>=len(idx): break
            inv_ids.append((idx[i],counter))
            i+=1
        counter+=1
    work["SyntheticInvoice"] = work.index.map(dict(inv_ids))
    schema["invoice"]="SyntheticInvoice"
    return work, schema


def ensure_synthetic_columns(df, schema):
    """Guarantee presence of product/date/customer/invoice/amount."""
    work = df.copy()
    n = len(work)

    # Product
    if schema["product"] is None:
        work["SyntheticProduct"]="ITEM_"+work.index.astype(str)
        schema["product"]="SyntheticProduct"
    else:
        work[schema["product"]] = normalize_products(work[schema["product"]])

    # Customer
    if schema["customer"] is None:
        work["SyntheticCustomer"] = "CUST_"+(work.index%max(n,1)).astype(str)
        schema["customer"]="SyntheticCustomer"

    # Date
    if schema["date"] is None:
        rng=np.random.default_rng(42)
        work["SyntheticDate"]=pd.Timestamp("2025-01-01")+pd.to_timedelta(rng.integers(0,120,n),"D")
        schema["date"]="SyntheticDate"
    else:
        work[schema["date"]] = pd.to_datetime(work[schema["date"]],errors="coerce")
        if work[schema["date"]].notna().sum()==0:
            rng=np.random.default_rng(42)
            work["SyntheticDate"]=pd.Timestamp("2025-01-01")+pd.to_timedelta(rng.integers(0,120,n),"D")
            schema["date"]="SyntheticDate"

    # Amount
    if schema["amount"] is None:
        work["SyntheticAmount"]=1.0
        schema["amount"]="SyntheticAmount"

    # Invoice
    if schema["invoice"] is None:
        work, schema = create_realistic_invoices(work,schema)

    return work, schema


def build_transactions(df, schema):
    """Make basket list from invoice→product."""
    g = df.groupby(schema["invoice"])[schema["product"]].apply(lambda s: sorted(set(s)))
    return [t for t in g.tolist() if len(t)>0]


def encode_transactions(transactions):
    """One-hot encode baskets."""
    te=TransactionEncoder()
    arr=te.fit(transactions).transform(transactions)
    return pd.DataFrame(arr, columns=te.columns_)


def run_mining(df_enc, algo, min_support, max_len):
    """Run FP-Growth or Apriori."""
    t0=time.time()
    if algo=="Apriori":
        fi=apriori(df_enc,min_support=min_support,use_colnames=True,max_len=max_len)
    else:
        fi=fpgrowth(df_enc,min_support=min_support,use_colnames=True,max_len=max_len)
    return fi.sort_values("support",ascending=False), time.time()-t0


def make_rules(fi, metric, threshold):
    """Make association rules."""
    if fi is None or fi.empty: return pd.DataFrame()
    rules=association_rules(fi,metric=metric,min_threshold=threshold)
    rules["antecedents_str"]=rules["antecedents"].apply(lambda s:", ".join(sorted(s)))
    rules["consequents_str"]=rules["consequents"].apply(lambda s:", ".join(sorted(s)))
    return rules.sort_values(["lift","confidence","support"],ascending=False)
def plot_top_itemsets(fi, n=20):
    """Horizontal bar chart for top itemsets by support."""
    if fi.empty:
        return None
    tmp = fi.head(n).copy()
    tmp["itemset_str"] = tmp["itemsets"].apply(lambda s: ", ".join(sorted(s)))
    fig = px.bar(
        tmp[::-1],
        x="support",
        y="itemset_str",
        orientation="h",
        title=f"Top {n} Frequent Itemsets",
    )
    return fig


def plot_rules_scatter(rules):
    """Scatter plot of rules: support vs confidence, bubble size = lift."""
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


def build_association_network(rules, topn=30):
    """Network graph of top-N rules (A→B edges)."""
    if rules.empty:
        return None
    sub = rules.head(topn)
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
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                hoverinfo="none",
                line=dict(width=1),
            ),
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


def missing_summary(df):
    """Missing values per column (count + percentage)."""
    total = df.isna().sum()
    pct = (total / len(df) * 100).round(2) if len(df) else 0
    out = pd.DataFrame({"Missing": total, "Missing_%": pct})
    return out.sort_values("Missing", ascending=False)


def build_rfm(df, schema):
    """
    Build Recency-Frequency-Monetary per customer.
    Recency  = days since last purchase
    Frequency = number of transactions
    Monetary  = total amount spent
    """
    cust, date, amt = schema["customer"], schema["date"], schema["amount"]
    if cust not in df.columns or date not in df.columns or amt not in df.columns:
        return pd.DataFrame()

    work = df.copy().dropna(subset=[cust, date])
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

    # Convert R/F/M to scores 1–5 (quantiles or bins)
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


def kmeans_clusters(df_rfm, k=3):
    """Run K-Means on [Recency, Frequency, Monetary]."""
    try:
        feats = df_rfm[["Recency", "Frequency", "Monetary"]].fillna(0)
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


def rules_to_recommendations(rules, base_items, enc):
    """
    Core recommendation logic:

    1. Compute global item support from encoded matrix (enc).
    2. Identify top-3 global 'staple' items.
    3. Filter rules whose antecedents contain all base_items.
    4. Remove staples from consequents.
    5. Aggregate remaining consequents (mean_conf, mean_lift, count).
    """
    if rules.empty or not base_items:
        return pd.DataFrame()

    # 1) Global item support → staples
    try:
        item_support = enc.mean().sort_values(ascending=False)
        top_staples = set(item_support.head(3).index)
    except Exception:
        top_staples = set()

    # 2) Filter rules by antecedent
    sub = rules[rules["antecedents"].apply(lambda s: set(base_items).issubset(s))]
    if sub.empty:
        return pd.DataFrame()

    # 3) Remove staple consequents
    sub_filtered = sub.copy()
    sub_filtered["consequents"] = sub_filtered["consequents"].apply(
        lambda s: {x for x in s if x not in top_staples}
    )
    sub_filtered = sub_filtered[sub_filtered["consequents"].apply(lambda s: len(s) > 0)]
    if sub_filtered.empty:
        return pd.DataFrame()

    # 4) Aggregate remaining consequents as recs
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


# ==========================================================
# 1. Upload Dataset (only upload, no dataset picker)
# ==========================================================
st.sidebar.header("Upload Dataset")
uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded is None:
    st.info("Please upload a CSV dataset to begin.")
    st.stop()

df_raw = pd.read_csv(uploaded)
st.write(f"Rows: **{len(df_raw):,}**, Columns: **{len(df_raw.columns):,}**")
st.dataframe(df_raw.head(20), use_container_width=True)

# ==========================================================
# 2. Auto Schema Detection + Synthetic Fixes
# ==========================================================
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

df_work, schema = ensure_synthetic_columns(df_raw, schema)

st.markdown("**Active schema (after synthetic fixes):**")
st.json(schema)

# ==========================================================
# Tabs Definition
# ==========================================================
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

# ==========================================================
# TAB 0: Data Cleaning & Basket Size Graph
# ==========================================================
with tab0:
    st.subheader("Data Cleaning & Overview")

    st.markdown("### Missing Value Summary")
    miss = missing_summary(df_raw)
    st.dataframe(miss, use_container_width=True)

    st.markdown("### Cleaned Working Data (after auto schema & synthetics)")
    st.dataframe(df_work.head(20), use_container_width=True)

    # Basket size distribution
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

    # Download cleaned data
    csv_bytes = df_work.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download cleaned_dataset.csv",
        data=csv_bytes,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )

# ==========================================================
# TAB 1: Patterns & Model Comparison
# ==========================================================
with tab1:
    st.subheader("Frequent Patterns & Model Comparison")

    transactions = build_transactions(df_work, schema)
    st.write(f"Built **{len(transactions):,}** basket transactions.")

    if not transactions:
        st.warning("No transactions found even after synthetic fixes.")
        st.stop()

    df_enc = encode_transactions(transactions)
    st.write(f"Unique items in baskets: **{df_enc.shape[1]:,}**")

    colA, colB = st.columns(2)
    min_support = colA.slider("Min Support", 0.001, 0.1, 0.01, step=0.001)
    max_len = colB.slider("Max Itemset Length", 2, 5, 3)

    # Compare FP-Growth vs Apriori
    st.markdown("### Model Comparison: FP-Growth vs Apriori")
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

    # Graph comparing number of itemsets
    fig_comp = px.bar(
        comp,
        x="Algorithm",
        y="Frequent Itemsets",
        color="Algorithm",
        title="Frequent Itemsets Found: FP-Growth vs Apriori",
        text="Frequent Itemsets",
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    metric = st.selectbox("Rule Metric", ["confidence", "lift"])
    min_metric = st.slider(f"Min {metric.title()}", 0.1, 1.0, 0.3, step=0.05)
    topn = st.slider("Top N Itemsets to display", 10, 50, 20, step=5)

    rules = make_rules(fi_fp, metric, min_metric)
    st.info(f"Rules generated (FP-Growth-based): **{len(rules):,}**")

    # Top itemsets chart
    if not fi_fp.empty:
        fig_top = plot_top_itemsets(fi_fp, n=topn)
        if fig_top:
            st.plotly_chart(fig_top, use_container_width=True)

    # Rules scatter + network
    if not rules.empty:
        fig_rules_sc = plot_rules_scatter(rules)
        if fig_rules_sc:
            st.plotly_chart(fig_rules_sc, use_container_width=True)

        st.markdown("### Top Association Rules")
        st.dataframe(
            rules[["antecedents_str", "consequents_str", "support", "confidence", "lift"]].head(30),
            use_container_width=True,
        )

        fig_net = build_association_network(rules, topn=30)
        if fig_net:
            st.plotly_chart(fig_net, use_container_width=True)

    # Store for recommender
    st.session_state["df_enc"] = df_enc
    st.session_state["rules"] = rules

# ==========================================================
# TAB 2: Temporal & Sequential
# ==========================================================
with tab2:
    st.subheader("Temporal & Sequential Analysis")

    date_col = schema["date"]
    amt_col = schema["amount"]
    prod_col = schema["product"]
    cust_col = schema["customer"]

    if date_col in df_work.columns:
        df_t = df_work.dropna(subset=[date_col]).copy()
        if not df_t.empty:
            # Monthly transactions
            monthly = (
                df_t.groupby(df_t[date_col].dt.to_period("M"))
                .size()
                .reset_index(name="tx_count")
            )
            monthly[date_col] = monthly[date_col].astype(str)
            if not monthly.empty:
                st.markdown("### Transactions per Month")
                st.line_chart(monthly.set_index(date_col))

            # Day-of-week
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

            # Revenue per month
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

            # Sequential A→B transitions
            if cust_col in df_t.columns and prod_col in df_t.columns:
                seq_data = (
                    df_t.dropna(subset=[prod_col])
                    .sort_values([cust_col, date_col])
                    .copy()
                )
                seq_data[prod_col] = normalize_products(seq_data[prod_col])
                seq_counts = Counter()
                for _, group in seq_data.groupby(cust_col)[prod_col]:
                    seq = list(group)
                    for i in range(len(seq) - 1):
                        seq_counts[(seq[i], seq[i+1])] += 1
                if seq_counts:
                    seq_df = pd.DataFrame(
                        [(a,b,c) for (a,b),c in seq_counts.items()],
                        columns=["From","To","Count"]
                    ).sort_values("Count",ascending=False)
                    st.markdown("### Top Sequential Transitions (A → B)")
                    st.dataframe(seq_df.head(20), use_container_width=True)
        else:
            st.info("No valid dates after parsing.")
    else:
        st.info("No usable date column for temporal analysis.")

# ==========================================================
# TAB 3: Customer Segmentation
# ==========================================================
with tab3:
    st.subheader("Customer Segmentation (RFM + K-Means)")

    rfm = build_rfm(df_work, schema)
    if rfm.empty:
        st.info(
            "RFM requires usable customer, date, and amount columns "
            "(they may be synthetic). Not enough info in this dataset."
        )
    else:
        st.markdown("### RFM Summary (Top 50 customers)")
        st.dataframe(rfm.head(50), use_container_width=True)

        k = st.slider("Number of K-Means clusters", 2, 6, 3)
        rfm_km, centers = kmeans_clusters(rfm, k=k)

        if not centers.empty:
            st.markdown("### Cluster Centers (approximate original scale)")
            st.dataframe(centers, use_container_width=True)

        try:
            fig_seg = px.scatter(
                rfm_km,
                x="Frequency",
                y="Monetary",
                color=rfm_km.get("KMeansCluster", pd.Series([0]*len(rfm_km))).astype(str),
                hover_data=["customerid","Recency","RFM_Score","Segment"],
                title="Customer Segmentation: Frequency vs Monetary",
            )
            st.plotly_chart(fig_seg, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not plot segmentation scatter: {e}")

        seg_summary = (
            rfm_km.groupby("Segment", dropna=False)
            .agg(
                Customers=("customerid","count"),
                AvgSpend=("Monetary","mean"),
                AvgFreq=("Frequency","mean"),
                AvgRecency=("Recency","mean"),
            ).round(2)
        )
        st.markdown("### Segment Summary")
        st.dataframe(seg_summary, use_container_width=True)

        seg_counts = rfm_km["Segment"].value_counts(dropna=False).reset_index()
        seg_counts.columns=["Segment","Customers"]
        fig_pie = px.pie(
            seg_counts,
            names="Segment",
            values="Customers",
            title="Customer Distribution by Segment",
        )
        st.plotly_chart(fig_pie, use_container_width=True)
# ==========================================================
# TAB 4: Smart Recommender (Staple-Filtered)
# ==========================================================
with tab4:
    st.subheader("Smart Recommender (Staple-Filtered)")

    # Get rules & encoded baskets from Tab 1
    rules = st.session_state.get("rules", pd.DataFrame())
    df_enc = st.session_state.get("df_enc", pd.DataFrame())

    prod_col = schema["product"]
    amt_col = schema["amount"]
    inv_col = schema["invoice"]

    if rules.empty or df_enc.empty:
        st.warning(
            "No rules or encoded baskets available. "
            "Please ensure Tab 1 has run and found some frequent itemsets and rules."
        )
    else:
        # Build candidate item list from rules
        try:
            all_items = sorted(
                set().union(*rules["antecedents"]).union(*rules["consequents"])
            )
        except Exception:
            all_items = sorted(df_enc.columns.tolist())

        selected = st.multiselect(
            "Select base product(s) for recommendations",
            all_items,
            help="Choose one or more items that the customer is buying. "
                 "The system will recommend additional add-on products."
        )

        if selected:
            recs = rules_to_recommendations(rules, selected, df_enc)

            if recs.empty:
                st.info(
                    "No non-staple recommendations found for this selection. "
                    "Try a different product or relax thresholds in Tab 1."
                )
            else:
                st.markdown("### Top Recommended Add-ons (after filtering staples)")
                st.dataframe(recs, use_container_width=True)

                # Bar chart of mean lift (strength of recommendations)
                st.markdown("#### Recommendation Strength (Mean Lift)")
                st.bar_chart(recs["mean_lift"])

                # Estimate revenue impact if invoice and amount exist
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

                # Clear explanation for viva / lecturer
                st.markdown("### Explanation: How the Recommender Works")
                st.write(
                    "- We first mine **association rules** from baskets in Tab 1.\n"
                    "- Each rule has the form `{Antecedents} → {Consequents}` with support, confidence, and lift.\n"
                    "- When you select base product(s), we keep only rules whose **antecedents** contain all those items.\n"
                    "- We compute **global item support** and identify the top-3 most common 'staple' items.\n"
                    "- These staples are removed from the rule consequents to avoid trivial suggestions like BREAD or MILK.\n"
                    "- For each remaining consequent item, we aggregate information:\n"
                    "   * **mean_conf** = average confidence across all rules suggesting that item\n"
                    "   * **mean_lift** = average lift (how much better than random)\n"
                    "   * **count** = number of rules backing that item\n"
                    "- Finally, we sort by lift, confidence, and rule count to display the **top recommended add-ons**."
                )

# ==========================================================
# TAB 5: Visual Explanations (Pipeline & Methodology)
# ==========================================================
with tab5:
    st.subheader("Visual Explanations & Pipeline")

    st.markdown("### Overall Data Mining Pipeline")
    st.write(
        "This diagram summarizes the full flow of the project:\n"
        "Raw Data → Cleaning & Schema Detection → Baskets → Frequent Itemsets → "
        "Association Rules → Smart Recommender."
    )

    # Sankey diagram for pipeline
    labels = [
        "Raw Data",
        "Cleaned & Typed Data",
        "Baskets / Invoices",
        "Frequent Itemsets",
        "Association Rules",
        "Smart Recommender",
    ]
    source = [0, 1, 2, 3, 4]  # from
    target = [1, 2, 3, 4, 5]  # to
    value = [10, 10, 10, 10, 10]

    sankey_fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=20,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                ),
                link=dict(source=source, target=target, value=value),
            )
        ]
    )
    sankey_fig.update_layout(title_text="End-to-End Data Mining Pipeline", font_size=12)
    st.plotly_chart(sankey_fig, use_container_width=True)

    st.markdown("### Step-by-Step Recommender Logic (Conceptual)")
    st.markdown(
        """
1. **Frequent Itemsets**  
   - We find which items frequently occur together in baskets using FP-Growth / Apriori.

2. **Association Rules**  
   - From these itemsets, we build rules of the form  
     **{A, B} → {C}**, with:
       - Support: how often the full pattern appears  
       - Confidence: how often C appears when A,B are present  
       - Lift: how much more likely C is compared to random

3. **User Selection**  
   - In the Smart Recommender tab, the user picks one or more base products (e.g. EGGS, PASTA).

4. **Rule Filtering**  
   - We keep only rules whose **antecedents** contain all selected items.

5. **Staple Filtering**  
   - We compute global item popularity in the whole dataset.  
   - The top-3 most frequent items (staples like BREAD or MILK) are removed from consequents, so recommendations are not trivial.

6. **Aggregation**  
   - For each remaining recommended item (consequent), we aggregate across all matching rules:  
       - Mean confidence  
       - Mean lift  
       - Rule count  

7. **Ranking & Display**  
   - Items are sorted primarily by **lift** (strength of association), then confidence and count.  
   - The top items are displayed as the recommended add-on products.
        """
    )

    st.markdown("### RFM & Segmentation (Logic Overview)")
    st.markdown(
        """
- **Recency (R)**: Days since the customer's last purchase. Lower = more recent.
- **Frequency (F)**: Number of transactions in the period.
- **Monetary (M)**: Total spending in the period.

We transform R, F, M into scores from 1–5 and sum them to obtain an **RFM_Score**.  
Using this, we segment customers into four groups:
- **VIP** (highest RFM_Score)  
- **High**  
- **Mid**  
- **Low**

We then apply **K-Means clustering** on (R, F, M) to get a numerical clustering that can be compared with the rule-based RFM segments.  
This combination of rule-based and model-based segmentation strengthens the business interpretation.
        """
    )

    st.info(
        "Use this tab in your presentation to explain the method visually and logically. "
        "You can walk your lecturer through each step of the pipeline and how it leads "
        "to actionable recommendations."
    )
