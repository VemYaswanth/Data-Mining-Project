
# app_streamlit_v2.py
# Enhanced Streamlit App with Product Recommender tab

import streamlit as st
import pandas as pd
import numpy as np
import time
from itertools import combinations
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

st.set_page_config(page_title="Omnichannel Market Basket Analysis", page_icon="🛒", layout="wide")

st.title("🛒 Omnichannel Market Basket Analysis & Recommender")
st.caption("Apriori & FP-Growth based interactive web app with product recommendations.")

# ------------------------------
# Utility Functions
# ------------------------------
@st.cache_data(show_spinner=False)
def load_csv_safely(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def detect_schema(df: pd.DataFrame) -> dict:
    cols = {c.lower(): c for c in df.columns}
    schema = {"product_col": None, "invoice_col": None, "cust_col": None, "date_col": None}
    for cand in ["description", "productname", "product_name", "item"]:
        if cand in cols: schema["product_col"] = cols[cand]; break
    for cand in ["invoiceno", "invoice", "basket_id", "transaction_id"]:
        if cand in cols: schema["invoice_col"] = cols[cand]; break
    for cand in ["customerid", "customer_id"]: 
        if cand in cols: schema["cust_col"] = cols[cand]; break
    for cand in ["transactiondate", "date", "orderdate"]: 
        if cand in cols: schema["date_col"] = cols[cand]; break
    return schema

def build_transactions(df, schema):
    prod = schema["product_col"]
    if prod is None: return []
    work = df.dropna(subset=[prod]).copy()
    work[prod] = work[prod].astype(str).str.upper().str.strip()
    if schema["invoice_col"]:
        transactions = work.groupby(schema["invoice_col"])[prod].apply(list).tolist()
    elif schema["cust_col"] and schema["date_col"]:
        work["_tx"] = work[schema["cust_col"]].astype(str) + "_" + pd.to_datetime(work[schema["date_col"]], errors='coerce').dt.date.astype(str)
        transactions = work.groupby("_tx")[prod].apply(list).tolist()
    else:
        transactions = [[x] for x in work[prod]]
    return transactions

def encode_transactions(transactions):
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_array, columns=te.columns_)

def run_mining(df_encoded, algo, min_support, max_len):
    t0 = time.time()
    if algo == "Apriori":
        fi = apriori(df_encoded, min_support=min_support, use_colnames=True, max_len=max_len)
    else:
        fi = fpgrowth(df_encoded, min_support=min_support, use_colnames=True, max_len=max_len)
    return fi, time.time()-t0

def make_rules(fi, metric, min_threshold):
    if fi.empty: return pd.DataFrame()
    rules = association_rules(fi, metric=metric, min_threshold=min_threshold)
    rules["antecedents_str"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules["consequents_str"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
    return rules

# ------------------------------
# Sidebar - Data Loading
# ------------------------------
st.sidebar.header("Data & Settings")
sample_choice = st.sidebar.selectbox("Choose dataset", ["Upload CSV", "chain_cleaned.csv", "grocery_cleaned.csv"])
uploaded = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

if sample_choice != "Upload CSV" and uploaded is None:
    df = load_csv_safely("/mnt/data/" + sample_choice)
else:
    df = pd.read_csv(uploaded) if uploaded else pd.DataFrame()

if df.empty:
    st.warning("Please upload or select a dataset to begin.")
    st.stop()

schema = detect_schema(df)

# Tabs for analysis and recommender
tab1, tab2, tab3 = st.tabs(["📊 Dataset & Rules", "🛒 Product Recommender", "📈 Comparison"])

# ------------------------------
# TAB 1: Dataset & Rule Mining
# ------------------------------
with tab1:
    st.subheader("Dataset Overview")
    st.write("**Detected schema:**", schema)
    st.dataframe(df.head(30))

    transactions = build_transactions(df, schema)
    st.write(f"Transactions built: {len(transactions)}")

    if not transactions:
        st.error("No transactions could be built.")
        st.stop()

    df_encoded = encode_transactions(transactions)
    st.write(f"Unique items: {df_encoded.shape[1]}")

    algo = st.selectbox("Algorithm", ["FP-Growth", "Apriori"])
    min_support = st.slider("Min Support", 0.001, 0.1, 0.01, step=0.001)
    max_len = st.slider("Max Itemset Length", 2, 5, 3)
    metric = st.selectbox("Rule Metric", ["confidence", "lift"])
    min_metric = st.slider(f"Min {metric.title()}", 0.1, 1.0, 0.3, step=0.05)

    with st.spinner(f"Running {algo}..."):
        fi, elapsed = run_mining(df_encoded, algo, min_support, max_len)
    st.success(f"Mining completed in {elapsed:.2f}s with {len(fi)} itemsets.")

    rules = make_rules(fi, metric, min_metric)
    st.write(f"Generated {len(rules)} rules.")

    if not rules.empty:
        st.dataframe(rules[["antecedents_str", "consequents_str", "support", "confidence", "lift"]].head(30))

        fig = px.scatter(rules, x="support", y="confidence", size="lift", hover_data=["antecedents_str", "consequents_str"], title="Support vs Confidence (size = Lift)")
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# TAB 2: Product Recommender
# ------------------------------
with tab2:
    st.subheader("🛍️ Smart Product Recommender")
    st.write("Select a product or combination of products to get recommendations based on mined rules.")

    if 'rules' not in locals() or rules.empty:
        st.warning("Please generate rules in the first tab before using the recommender.")
    else:
        all_items = sorted(set().union(*rules['antecedents']).union(*rules['consequents']))
        selected = st.multiselect("Select product(s)", all_items)

        if selected:
            subset = rules[rules['antecedents'].apply(lambda s: set(selected).issubset(s))]
            if subset.empty:
                st.info("No rules found for the selected product(s). Try a broader selection.")
            else:
                recs = (subset.explode('consequents')
                            .groupby('consequents')
                            .agg({'confidence':'mean','lift':'mean'})
                            .sort_values(['lift','confidence'], ascending=False)
                            .head(10))
                st.subheader("Top Recommended Products")
                st.dataframe(recs)
                st.bar_chart(recs['lift'])

                st.markdown("### 🔍 Explanation")
                best = recs.head(3).index.tolist()
                if best:
                    st.write(f"Customers who buy **{', '.join(selected)}** often also buy **{', '.join(best)}**, with high lift and confidence values indicating strong associations.")

# ------------------------------
# TAB 3: Comparison
# ------------------------------
with tab3:
    st.subheader("Model Comparison (FP-Growth vs Apriori)")
    results = []
    for algo_name in ["FP-Growth", "Apriori"]:
        fi_tmp, t_tmp = run_mining(df_encoded, algo_name, min_support, max_len)
        rules_tmp = make_rules(fi_tmp, metric, min_metric)
        results.append({"Algorithm": algo_name, "Runtime (s)": round(t_tmp, 2), "Itemsets": len(fi_tmp), "Rules": len(rules_tmp)})
    st.dataframe(pd.DataFrame(results))
