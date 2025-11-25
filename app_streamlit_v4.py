import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# ML & Mining Libraries
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page Config
st.set_page_config(page_title="Retail Intelligence Dashboard", page_icon="📈", layout="wide")

# --------------------------------
# 1. Helper Functions
# --------------------------------

@st.cache_data
def generate_demo_data(n_rows=2000):
    """Generates a synthetic retail dataset for testing purposes."""
    np.random.seed(42)
    products = {'PROD_A': 'Wireless Mouse', 'PROD_B': 'Mechanical Keyboard', 'PROD_C': 'USB-C Cable', 
                'PROD_D': 'Monitor Stand', 'PROD_E': 'Webcam HD', 'PROD_F': 'Gaming Headset'}
    prod_codes = list(products.keys())
    cust_ids = [f"CUST_{i:03d}" for i in range(1, 51)]
    
    data = []
    invoice_id = 1000
    start_date = datetime.now() - timedelta(days=180)
    
    for _ in range(n_rows // 3):
        invoice_id += 1
        date = start_date + timedelta(days=np.random.randint(0, 180))
        customer = np.random.choice(cust_ids)
        basket_size = np.random.randint(1, 6)
        basket_items = np.random.choice(prod_codes, basket_size, replace=False)
        
        for p_code in basket_items:
            qty = np.random.randint(1, 4)
            price = np.random.uniform(10, 100)
            data.append({
                "InvoiceNo": str(invoice_id),
                "product_name": products[p_code],
                "Quantity": qty,
                "transaction_date": date,
                "final_amount": round(qty * price, 2),
                "customer_id": customer
            })
            
    return pd.DataFrame(data)

def detect_schema(df: pd.DataFrame) -> dict:
    """Maps raw column names to logical schema, updated for snake_case."""
    cols = {c.lower(): c for c in df.columns}
    schema = {
        "product": None, "invoice": None, "customer": None, "date": None,
        "amount": None
    }
    
    # Product
    for cand in ["product_name", "productname", "description", "item"]:
        if cand in cols: schema["product"] = cols[cand]; break
        
    # Invoice (Optional - can use Customer+Date if missing)
    for cand in ["invoiceno", "invoice", "transaction_id", "invoice_id"]:
        if cand in cols: schema["invoice"] = cols[cand]; break
        
    # Customer
    for cand in ["customer_id", "customerid", "custid", "id"]:
        if cand in cols: schema["customer"] = cols[cand]; break
        
    # Date
    for cand in ["transaction_date", "transactiondate", "invoicedate", "date"]:
        if cand in cols: schema["date"] = cols[cand]; break
        
    # Amount
    for cand in ["final_amount", "finalamount", "total_amount", "totalamount", "sales"]:
        if cand in cols: schema["amount"] = cols[cand]; break
        
    return schema

def preprocess_data(df, schema):
    """Cleans data and creates a surrogate Invoice ID if missing."""
    df = df.copy()
    
    # 1. Date Handling
    if schema["date"]:
        df[schema["date"]] = pd.to_datetime(df[schema["date"]], errors='coerce')
    
    # 2. Invoice Creation (Crucial for datasets like yours)
    # If no unique invoice ID exists, we group by (Customer + Date)
    if not schema["invoice"] and schema["customer"] and schema["date"]:
        st.caption("ℹ️ No Invoice ID found. Creating baskets based on Customer + Date.")
        df['_invoice_id'] = df[schema["customer"]].astype(str) + "_" + df[schema["date"]].dt.date.astype(str)
        schema["invoice"] = '_invoice_id'
        
    return df, schema

def build_transactions(df: pd.DataFrame, schema: dict) -> list:
    prod = schema["product"]
    inv = schema["invoice"]
    
    if not prod or not inv:
        return []
    
    # Clean products
    df = df.dropna(subset=[prod])
    df[prod] = df[prod].astype(str).str.strip()
    
    # Group
    transactions = df.groupby(inv)[prod].apply(
        lambda x: list(set(x))
    ).tolist()
    
    return transactions

def run_mining_pipeline(transactions, min_support, min_metric, metric_name="lift"):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    
    if frequent_itemsets.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    rules = association_rules(frequent_itemsets, metric=metric_name, min_threshold=min_metric)
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ', '.join(list(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: ', '.join(list(x)))
    
    return frequent_itemsets, rules

def calculate_rfm(df, schema):
    cust = schema["customer"]
    date_col = schema["date"]
    amt = schema["amount"]
    inv = schema["invoice"]
    
    if not all([cust, date_col, amt]):
        return pd.DataFrame()
    
    now = df[date_col].max() + timedelta(days=1)
    
    rfm = df.groupby(cust).agg({
        date_col: lambda x: (now - x.max()).days,
        inv: 'nunique',
        amt: 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    # Filter out negative monetary values (returns/errors)
    rfm = rfm[rfm['Monetary'] > 0]
    
    return rfm

# --------------------------------
# 2. Main App Logic
# --------------------------------
st.title("📈 Retail Intelligence Dashboard")

# Sidebar
with st.sidebar:
    st.header("Data Source")
    source_opt = st.radio("Choose Source", ["Upload CSV", "Use Demo Data"])
    
    df = pd.DataFrame()
    if source_opt == "Use Demo Data":
        df = generate_demo_data()
    else:
        upl = st.file_uploader("Upload CSV", type=['csv'])
        if upl:
            df = pd.read_csv(upl)

    st.divider()
    st.subheader("⚙️ Analysis Params")
    min_sup = st.slider("Min Support", 0.001, 0.2, 0.01, format="%.3f")
    metric = st.selectbox("Metric", ["lift", "confidence"])
    thresh = st.slider("Threshold", 0.1, 5.0, 1.0)

if df.empty:
    st.info("Awaiting data...")
    st.stop()

# Auto-detect and Preprocess
raw_schema = detect_schema(df)
df_clean, schema = preprocess_data(df, raw_schema)

# Validation
if not schema["product"]:
    st.error(f"Could not identify a Product column. Found: {list(df.columns)}")
    st.stop()

# Tabs
t1, t2, t3 = st.tabs(["🛍️ Market Basket", "👥 Segmentation", "🤖 Recommender"])

# --- TAB 1: Rules ---
with t1:
    transactions = build_transactions(df_clean, schema)
    st.metric("Total Baskets", len(transactions))
    
    with st.spinner("Mining..."):
        fi, rules = run_mining_pipeline(transactions, min_sup, thresh, metric)
        
    if not rules.empty:
        st.dataframe(rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False).head(10), use_container_width=True)
        
        # Network Viz
        top_rules = rules.sort_values("lift", ascending=False).head(20)
        G = nx.DiGraph()
        for _, r in top_rules.iterrows():
            G.add_edge(r['antecedents_str'], r['consequents_str'], weight=r['lift'])
            
        pos = nx.spring_layout(G, k=0.8, seed=42)
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
            
        fig = go.Figure([
            go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888')),
            go.Scatter(x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()],
                       mode='markers+text', text=list(G.nodes()), textposition="top center",
                       marker=dict(size=10, color='lightblue'))
        ])
        fig.update_layout(showlegend=False, margin=dict(t=0,b=0,l=0,r=0), xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No rules found. Try lowering Support/Threshold.")

# --- TAB 2: RFM ---
with t2:
    rfm = calculate_rfm(df_clean, schema)
    if not rfm.empty:
        scaler = StandardScaler()
        X = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])
        k = st.slider("Clusters", 2, 6, 3)
        rfm['Cluster'] = KMeans(n_clusters=k, random_state=42).fit_predict(X)
        
        c1, c2 = st.columns([1,2])
        c1.write(rfm.groupby('Cluster')[['Recency','Frequency','Monetary']].mean().round(1))
        
        fig = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary', color='Cluster', opacity=0.6)
        c2.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Insufficient data for RFM (Need Customer ID + Date + Amount)")

# --- TAB 3: Recommender ---
with t3:
    all_items = sorted(list(set([i for t in transactions for i in t])))
    cart = st.multiselect("Select items in cart", all_items)
    
    if cart and not rules.empty:
        recs = []
        for _, r in rules.iterrows():
            if set(r['antecedents']).issubset(set(cart)):
                add_ons = set(r['consequents']) - set(cart)
                if add_ons:
                    recs.append({"Item": ', '.join(add_ons), "Confidence": r['confidence'], "Lift": r['lift']})
        
        if recs:
            st.success("Recommendations generated!")
            st.dataframe(pd.DataFrame(recs).sort_values("Lift", ascending=False).drop_duplicates("Item"), use_container_width=True)
        else:
            st.info("No specific recommendations for this combination.")
