import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# ML & Mining Libraries
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Retail Intelligence Dashboard", page_icon="📈", layout="wide")

# --------------------------------
# 1. Helper Functions
# --------------------------------

@st.cache_data
def generate_demo_data(n_rows=2000):
    """Generates synthetic data with meaningful patterns."""
    np.random.seed(42)
    products = {
        'PROD_A': 'Wireless Mouse', 'PROD_B': 'Mechanical Keyboard', 
        'PROD_C': 'USB-C Cable', 'PROD_D': 'Monitor Stand', 
        'PROD_E': 'Webcam HD', 'PROD_F': 'Gaming Headset'
    }
    prod_codes = list(products.keys())
    cust_ids = [f"CUST_{i:03d}" for i in range(1, 51)]
    
    data = []
    invoice_id = 1000
    start_date = datetime.now() - timedelta(days=180)
    
    for _ in range(n_rows // 3):
        invoice_id += 1
        date = start_date + timedelta(days=np.random.randint(0, 180))
        customer = np.random.choice(cust_ids)
        
        # Enforce multi-item baskets for demo
        basket_size = np.random.randint(2, 6) 
        basket_items = np.random.choice(prod_codes, basket_size, replace=False)
        
        for p_code in basket_items:
            qty = np.random.randint(1, 4)
            price = np.random.uniform(10, 100)
            data.append({
                "invoice_id": str(invoice_id),
                "product_name": products[p_code],
                "quantity": qty,
                "transaction_date": date,
                "final_amount": round(qty * price, 2),
                "customer_id": customer
            })
            
    return pd.DataFrame(data)

def detect_schema(df: pd.DataFrame) -> dict:
    """Maps raw column names to a standard schema."""
    cols = {c.lower(): c for c in df.columns}
    schema = {
        "product": None, "invoice": None, "customer": None, "date": None, "amount": None
    }
    
    # Flexible mapping
    for cand in ["product_name", "product", "description", "item"]:
        if cand in cols: schema["product"] = cols[cand]; break
        
    for cand in ["invoice_id", "invoiceno", "invoice", "transaction_id"]:
        if cand in cols: schema["invoice"] = cols[cand]; break
        
    for cand in ["customer_id", "customerid", "custid", "id"]:
        if cand in cols: schema["customer"] = cols[cand]; break
        
    for cand in ["transaction_date", "date", "invoicedate"]:
        if cand in cols: schema["date"] = cols[cand]; break
        
    for cand in ["final_amount", "total_amount", "amount", "sales"]:
        if cand in cols: schema["amount"] = cols[cand]; break
        
    return schema

def build_transactions(df, schema, grouping_strategy="invoice"):
    """
    Builds list of baskets based on user-selected grouping strategy.
    Strategies: 'invoice' (Invoice/Date), 'customer' (All time history)
    """
    prod = schema["product"]
    inv = schema["invoice"]
    cust = schema["customer"]
    date_col = schema["date"]
    
    # 1. Identify the Grouping Key
    group_col = None
    
    if grouping_strategy == "customer":
        if not cust: return [], "Missing Customer ID column"
        group_col = cust
        
    else: # Default: Invoice or Date
        if inv:
            group_col = inv
        elif cust and date_col:
            # Create Surrogate Invoice ID if missing
            df['_invoice_id'] = df[cust].astype(str) + "_" + pd.to_datetime(df[date_col]).dt.date.astype(str)
            group_col = '_invoice_id'
        else:
            return [], "Need Invoice ID or (Customer + Date) to group transactions."

    # 2. Build Baskets
    # Drop items with no product name
    work_df = df.dropna(subset=[prod]).copy()
    work_df[prod] = work_df[prod].astype(str).str.strip()
    
    # Filter out single-item baskets? No, keep them for support calculation, 
    # but they won't generate rules A->B on their own.
    
    transactions = work_df.groupby(group_col)[prod].apply(
        lambda x: list(set(x)) # Unique items per basket
    ).tolist()
    
    avg_len = np.mean([len(t) for t in transactions]) if transactions else 0
    
    return transactions, avg_len

def run_mining_pipeline(transactions, min_support, min_metric, metric_name="lift"):
    if not transactions:
        return pd.DataFrame(), pd.DataFrame()

    te = TransactionEncoder()
    try:
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        # FP-Growth
        frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
        
        if frequent_itemsets.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Rules
        rules = association_rules(frequent_itemsets, metric=metric_name, min_threshold=min_metric)
        
        if not rules.empty:
            rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ', '.join(list(x)))
            rules["consequents_str"] = rules["consequents"].apply(lambda x: ', '.join(list(x)))
            
        return frequent_itemsets, rules
        
    except Exception as e:
        st.error(f"Mining Error: {e}")
        return pd.DataFrame(), pd.DataFrame()

def calculate_rfm(df, schema):
    cust = schema["customer"]
    date_col = schema["date"]
    amt = schema["amount"]
    
    if not all([cust, date_col, amt]):
        return pd.DataFrame()
    
    # Ensure Date
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    now = df[date_col].max() + timedelta(days=1)
    
    # Calculate RFM
    # If no invoice col, use date as proxy for frequency
    freq_col = schema["invoice"] if schema["invoice"] else date_col
    freq_agg = 'nunique' if schema["invoice"] else 'count'

    rfm = df.groupby(cust).agg({
        date_col: lambda x: (now - x.max()).days,
        freq_col: freq_agg,
        amt: 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    rfm = rfm[rfm['Monetary'] > 0] # Filter returns
    return rfm

# --------------------------------
# 2. Main App Logic
# --------------------------------
st.title("📈 Retail Intelligence Dashboard")

# Sidebar
with st.sidebar:
    st.header("1. Data Source")
    source_opt = st.radio("Choose Source", ["Use Demo Data", "Upload CSV"], index=1)
    
    df = pd.DataFrame()
    if source_opt == "Use Demo Data":
        df = generate_demo_data()
    else:
        upl = st.file_uploader("Upload CSV", type=['csv'])
        if upl:
            df = pd.read_csv(upl)

    st.divider()
    st.header("2. Analysis Params")
    
    # New: Basket Definition Strategy
    st.markdown("**Basket Definition**")
    basket_mode = st.selectbox(
        "How to group items?",
        ["Same Invoice/Date", "Same Customer (All History)"],
        index=0,
        help="If your data has 1 item per row/date, switch to 'Same Customer' to see what customers buy over their lifetime."
    )
    grouping_key = "customer" if "Customer" in basket_mode else "invoice"

    st.markdown("**Mining Thresholds**")
    # Lowered minimum support to 0.001
    min_sup = st.slider("Min Support", 0.001, 0.2, 0.01, format="%.3f")
    metric = st.selectbox("Metric", ["lift", "confidence"])
    thresh = st.slider("Metric Threshold", 0.1, 5.0, 1.0)

if df.empty:
    st.info("👈 Upload a CSV or select Demo Data to start.")
    st.stop()

# Preprocess
schema = detect_schema(df)
if not schema["product"]:
    st.error("Could not find a 'Product' column. Please check your CSV.")
    st.stop()

# Tabs
t1, t2, t3 = st.tabs(["🛍️ Market Basket", "👥 Segmentation", "🤖 Recommender"])

# --- TAB 1: Market Basket ---
with t1:
    st.subheader("Association Rules")
    
    # Build Transactions
    transactions, avg_basket_size = build_transactions(df, schema, grouping_key)
    
    col1, col2 = st.columns(2)
    col1.metric("Total Baskets", len(transactions))
    col2.metric("Avg Items/Basket", f"{avg_basket_size:.2f}")

    # Warning for sparse data
    if avg_basket_size < 1.1:
        st.warning(
            f"⚠️ **Low Basket Size ({avg_basket_size:.2f}) detected.**\n\n"
            "Almost all your transactions have only 1 item. Association rules (A → B) cannot be found."
            "\n👉 **Action:** Switch 'Basket Definition' in the sidebar to **'Same Customer (All History)'**."
        )

    with st.spinner("Mining patterns..."):
        fi, rules = run_mining_pipeline(transactions, min_sup, thresh, metric)

    if not rules.empty:
        st.success(f"Found {len(rules)} rules!")
        st.dataframe(
            rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']]
            .sort_values('lift', ascending=False)
            .head(10), 
            use_container_width=True
        )
        
        # Network Viz
        try:
            G = nx.DiGraph()
            # Limit to top 30 to prevent lag
            top_rules = rules.sort_values("lift", ascending=False).head(30)
            
            for _, r in top_rules.iterrows():
                G.add_edge(r['antecedents_str'], r['consequents_str'], weight=r['lift'])
                
            pos = nx.spring_layout(G, k=0.5, seed=42)
            
            edge_x, edge_y = [], []
            for u, v in G.edges():
                x0, y0 = pos[u]; x1, y1 = pos[v]
                edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
                
            node_x = [pos[n][0] for n in G.nodes()]
            node_y = [pos[n][1] for n in G.nodes()]
            
            fig = go.Figure([
                go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'), hoverinfo='none'),
                go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()), 
                           textposition="top center", marker=dict(size=10, color='royalblue'))
            ])
            fig.update_layout(showlegend=False, margin=dict(t=10,b=10,l=10,r=10), 
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Network viz unavailable: {e}")
            
    else:
        st.warning("No rules found. Try lowering 'Min Support' or changing 'Basket Definition'.")

# --- TAB 2: RFM ---
with t2:
    st.subheader("Customer Segmentation")
    rfm = calculate_rfm(df, schema)
    
    if not rfm.empty:
        # Normalize
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
        
        k = st.slider("Number of Segments (K)", 2, 6, 3)
        rfm['Cluster'] = KMeans(n_clusters=k, random_state=42).fit_predict(rfm_scaled)
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.write("### Segment Stats")
            st.dataframe(rfm.groupby('Cluster')[['Recency','Frequency','Monetary']].mean().round(1))
        
        with c2:
            fig = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary', color='Cluster', opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Cannot calculate RFM. Ensure Customer, Date, and Amount columns exist.")

# --- TAB 3: Recommender ---
with t3:
    st.subheader("Smart Recommender")
    
    all_items = sorted(list(set([i for t in transactions for i in t])))
    cart = st.multiselect("Simulate Cart (Select Items):", all_items)
    
    if cart and not rules.empty:
        recs = []
        for _, r in rules.iterrows():
            # If cart contains the antecedent
            if set(r['antecedents']).issubset(set(cart)):
                add_ons = set(r['consequents']) - set(cart)
                if add_ons:
                    recs.append({
                        "Recommendation": ', '.join(add_ons),
                        "Confidence": r['confidence'],
                        "Lift": r['lift'],
                        "Reason": f"Because you picked {', '.join(r['antecedents'])}"
                    })
        
        if recs:
            st.success(f"Found {len(recs)} suggestions!")
            st.dataframe(pd.DataFrame(recs).sort_values("Lift", ascending=False).drop_duplicates("Recommendation"), use_container_width=True)
        else:
            st.info("No associations found for these specific items.")
    elif not rules.empty:
        st.info("Select items above to see what goes with them.")
    else:
        st.warning("No rules available (Generate them in Tab 1 first).")
