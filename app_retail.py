# app_retail.py
# Retail Intelligence Dashboard (v5.1 - Hybrid Stable Version)
# Updates:
#  - Keeps original 3-tab UI (Market Basket, Segmentation, Recommender)
#  - FIXED Data Enricher (no dtype errors / safe concatenation)
#  - Slightly safer rule-mining thresholds when using confidence
#  - Ready for local use + Streamlit Cloud

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# ML & Mining Libraries
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page Config
st.set_page_config(page_title="Retail Intelligence Dashboard", page_icon="📈", layout="wide")

# --------------------------------
# 1. Helper Functions
# --------------------------------

@st.cache_data
def generate_demo_data(n_rows=2000):
    """Generates synthetic data for testing."""
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
    """Maps raw column names to logical schema."""
    cols = {c.lower(): c for c in df.columns}
    schema = {
        "product": None, "invoice": None, "customer": None,
        "date": None, "amount": None, "price": None
    }
    
    for cand in ["product_name", "product", "description", "item"]:
        if cand in cols:
            schema["product"] = cols[cand]; break
        
    for cand in ["invoice_id", "invoiceno", "invoice", "transaction_id"]:
        if cand in cols:
            schema["invoice"] = cols[cand]; break
        
    for cand in ["customer_id", "customerid", "custid", "id"]:
        if cand in cols:
            schema["customer"] = cols[cand]; break
        
    for cand in ["transaction_date", "date", "invoicedate"]:
        if cand in cols:
            schema["date"] = cols[cand]; break
        
    for cand in ["final_amount", "total_amount", "amount", "sales", "total"]:
        if cand in cols:
            schema["amount"] = cols[cand]; break
        
    for cand in ["unit_price", "price", "unitprice"]:
        if cand in cols:
            schema["price"] = cols[cand]; break
        
    return schema

def enrich_data_with_patterns(df):
    """
    Injects synthetic relations into sparse datasets to ensure rules are found.
    Adds patterns like Pasta->Cheese, Cereal->Milk.

    IMPORTANT FIX:
    - Always builds NEW ROWS as plain dicts (no Series),
      so pd.DataFrame(new_rows) + concat will NEVER crash with dtype errors.
    """
    df = df.copy()
    schema = detect_schema(df)
    prod_col = schema['product']
    cust_col = schema['customer']
    date_col = schema['date']
    
    if not prod_col:
        return df
    
    # 1. Define Golden Rules (Antecedent -> Consequent)
    # We look for these products in the dataset. If they exist, we link them.
    known_rules = {
        'Pasta': 'Cheese',
        'Cereal': 'Milk',
        'Ground Beef': 'Onions',
        'Bananas': 'Yogurt',
        'Chicken Breast': 'Rice',
        'Shampoo': 'Conditioner',
        'Chips': 'Salsa'
    }
    
    # Filter rules to only products that actually exist in the data (fuzzy match)
    unique_prods = df[prod_col].astype(str).unique()
    valid_rules = {}
    
    # Simple fuzzy matcher
    for ant, con in known_rules.items():
        # Find closest match in data
        ant_match = next((p for p in unique_prods if ant.lower() in p.lower()), None)
        con_match = next((p for p in unique_prods if con.lower() in p.lower()), None)
        if ant_match and con_match:
            valid_rules[ant_match] = con_match

    if not valid_rules:
        return df  # Can't enrich if we don't recognize products
        
    new_rows = []
    
    # 2. Iterate and Inject
    # We iterate rows. If we see Antecedent, we add Consequent with 70% probability.
    
    # Calculate average prices for realistic injection
    if schema['price']:
        avg_prices = df.groupby(prod_col)[schema['price']].mean().to_dict()
    else:
        avg_prices = {}

    for _, row in df.iterrows():
        row_dict = row.to_dict()  # <-- SAFE copy as dict
        current_prod = str(row_dict.get(prod_col, ""))

        if current_prod in valid_rules:
            target_prod = valid_rules[current_prod]
            
            # 70% chance to add the relation
            if random.random() < 0.70:
                new_row = row_dict.copy()
                new_row[prod_col] = target_prod
                
                # Update price
                if schema['price']:
                    price = avg_prices.get(target_prod, 5.0)
                    new_row[schema['price']] = round(price, 2)
                if schema['amount']:
                    # Simple logic: 1 unit * price
                    if schema['price'] and new_row.get(schema['price']) is not None:
                        new_row[schema['amount']] = new_row[schema['price']]
                    else:
                        new_row[schema['amount']] = 5.0
                
                new_rows.append(new_row)
    
    # 3. Add Combo Baskets (New transactions purely for support)
    # Adds 20 baskets per rule of just [Item A, Item B]
    if cust_col and date_col:
        ids = df[cust_col].dropna().unique()
        if len(ids) > 0:
            for ant, con in valid_rules.items():
                for _ in range(20):
                    cid = random.choice(ids.tolist())
                    # Random date in last year
                    dt = datetime.now() - timedelta(days=random.randint(1, 365))
                    
                    # Base template row from first row of df
                    base = df.iloc[0].to_dict()
                    
                    # Row A
                    row_a = base.copy()
                    row_a[cust_col] = cid
                    row_a[date_col] = dt
                    row_a[prod_col] = ant
                    
                    # Row B
                    row_b = base.copy()
                    row_b[cust_col] = cid
                    row_b[date_col] = dt
                    row_b[prod_col] = con
                    
                    # Fill numeric defaults
                    if schema['price']:
                        row_a[schema['price']] = avg_prices.get(ant, 5.0)
                        row_b[schema['price']] = avg_prices.get(con, 5.0)
                    if schema['amount']:
                        row_a[schema['amount']] = avg_prices.get(ant, 5.0)
                        row_b[schema['amount']] = avg_prices.get(con, 5.0)
                    
                    new_rows.append(row_a)
                    new_rows.append(row_b)

    # Combine safely
    if not new_rows:
        return df

    enriched_df = pd.DataFrame(new_rows)
    # Align columns (any missing columns get NaN)
    enriched_df = enriched_df.reindex(columns=df.columns, fill_value=np.nan)

    st.toast(f"Enriched data with {len(new_rows)} new interactions!", icon="🪄")
    return pd.concat([df, enriched_df], ignore_index=True)

def build_transactions(df, schema, grouping_strategy="invoice"):
    prod = schema["product"]
    inv = schema["invoice"]
    cust = schema["customer"]
    date_col = schema["date"]
    
    group_col = None
    if grouping_strategy == "customer":
        if not cust: 
            return [], 0.0, "Missing Customer ID"
        group_col = cust
    else:
        if inv:
            group_col = inv
        elif cust and date_col:
            # Fallback: customer + date as pseudo-invoice
            df = df.copy()
            df['_invoice_id'] = df[cust].astype(str) + "_" + pd.to_datetime(df[date_col]).dt.date.astype(str)
            group_col = '_invoice_id'
        else:
            return [], 0.0, "Need Invoice or Customer+Date"

    work_df = df.dropna(subset=[prod]).copy()
    work_df[prod] = work_df[prod].astype(str).str.strip()
    
    transactions = work_df.groupby(group_col)[prod].apply(lambda x: list(set(x))).tolist()
    avg_len = np.mean([len(t) for t in transactions]) if transactions else 0
    return transactions, avg_len, None

def run_mining_pipeline(transactions, min_support, min_metric, metric_name="lift"):
    """
    Run FP-Growth + association_rules with safety around threshold values.
    If metric_name is 'confidence', clamp min_metric to [0, 1].
    If metric_name is 'lift', clamp min_metric to [1, +inf).
    """
    if not transactions:
        return pd.DataFrame(), pd.DataFrame()
    
    # Clamp metric thresholds to valid ranges
    if metric_name == "confidence":
        min_metric = max(0.0, min(min_metric, 1.0))
    elif metric_name == "lift":
        min_metric = max(min_metric, 1.0)
    
    te = TransactionEncoder()
    try:
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        fi = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
        if fi.empty:
            return pd.DataFrame(), pd.DataFrame()
        rules = association_rules(fi, metric=metric_name, min_threshold=min_metric)
        if not rules.empty:
            rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ', '.join(list(x)))
            rules["consequents_str"] = rules["consequents"].apply(lambda x: ', '.join(list(x)))
        return fi, rules
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

def calculate_rfm(df, schema):
    cust = schema["customer"]
    date_col = schema["date"]
    amt = schema["amount"]
    inv = schema["invoice"]
    
    if not all([cust, date_col, amt]):
        return pd.DataFrame()
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    now = df[date_col].max() + timedelta(days=1)
    
    freq_col = inv if inv else date_col
    freq_agg = 'nunique' if inv else 'count'

    rfm = df.groupby(cust).agg(
        Recency=(date_col, lambda x: (now - x.max()).days),
        Frequency=(freq_col, freq_agg),
        Monetary=(amt, 'sum')
    ).reset_index()
    
    return rfm[rfm['Monetary'] > 0]

# --------------------------------
# 2. Main App Logic
# --------------------------------
st.title("📈 Retail Intelligence Dashboard")

# Initialize Session State for Data persistence
if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame()
if 'enriched' not in st.session_state:
    st.session_state['enriched'] = False

with st.sidebar:
    st.header("1. Data Source")
    source_opt = st.radio("Choose Source", ["Use Demo Data", "Upload CSV"], index=1)
    
    if source_opt == "Use Demo Data":
        if st.button("Load Demo Data"):
            st.session_state['df'] = generate_demo_data()
            st.session_state['enriched'] = False
            st.rerun()
    else:
        upl = st.file_uploader("Upload CSV", type=['csv'])
        if upl is not None:
            st.session_state['df'] = pd.read_csv(upl)
            st.session_state['enriched'] = False

    st.divider()
    st.header("2. Analysis Params")
    basket_mode = st.selectbox("Group By", ["Same Invoice/Date", "Same Customer (History)"])
    grouping_key = "customer" if "Customer" in basket_mode else "invoice"
    
    min_sup = st.slider("Min Support", 0.001, 0.2, 0.01)
    metric = st.selectbox("Metric", ["lift", "confidence"])
    # Threshold slider remains 0.1–5.0, but we clamp inside run_mining_pipeline
    thresh = st.slider("Threshold", 0.1, 5.0, 1.0)

df = st.session_state['df']

if df.empty:
    st.info("👈 Upload data or load demo to start.")
    st.stop()

schema = detect_schema(df)
if not schema["product"]:
    st.error("No Product column found in your data. Please upload a file with a product column.")
    st.stop()

# Tabs
t1, t2, t3 = st.tabs(["🛍️ Market Basket", "👥 Segmentation", "🤖 Recommender"])

# --- TAB 1: Market Basket ---
with t1:
    st.subheader("Association Rules")
    
    # 1. Build Transactions
    transactions, avg_len, err = build_transactions(df, schema, grouping_key)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Baskets", len(transactions))
    col2.metric("Avg Basket Size", f"{avg_len:.2f}")
    if err:
        col3.error(err)
    else:
        col3.success("Basket building OK")
    
    # 2. Run Mining
    fi, rules = run_mining_pipeline(transactions, min_sup, thresh, metric)
    
    # 3. Handle No Rules / Sparse Data
    if rules.empty:
        st.warning(f"No rules found with Support={min_sup} & {metric}={thresh}.")
        
        # --- THE ENRICHMENT FEATURE ---
        st.divider()
        st.markdown("### 🤖 AI Data Fixer")
        st.info("Your data might be too sparse (items rarely bought together).")
        
        if st.button("✨ Inject Synthetic Patterns & Retry"):
            with st.spinner("Injecting relations (Pasta->Cheese, Cereal->Milk, etc.)..."):
                new_df = enrich_data_with_patterns(df)
                st.session_state['df'] = new_df
                st.session_state['enriched'] = True
                st.rerun()
                
    else:
        # Success State
        col3.success(f"{len(rules)} Rules Found!")
        
        st.dataframe(
            rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']]
            .sort_values('lift', ascending=False)
            .head(10),
            use_container_width=True
        )
        
        # Download Button for Enriched Data
        if st.session_state.get('enriched'):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Enriched CSV",
                csv,
                "grocery_enriched.csv",
                "text/csv"
            )

        # Network Viz
        try:
            G = nx.DiGraph()
            for _, r in rules.sort_values("lift", ascending=False).head(30).iterrows():
                G.add_edge(r['antecedents_str'], r['consequents_str'], weight=r['lift'])
            
            pos = nx.spring_layout(G, k=0.6, seed=42)
            edge_x, edge_y = [], []
            for u, v in G.edges():
                x0, y0 = pos[u]; x1, y1 = pos[v]
                edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
            
            node_x = [pos[n][0] for n in G.nodes()]
            node_y = [pos[n][1] for n in G.nodes()]
            
            fig = go.Figure([
                go.Scatter(
                    x=edge_x, y=edge_y,
                    mode='lines',
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none'
                ),
                go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=list(G.nodes()),
                    textposition="top center",
                    marker=dict(size=12, color='royalblue')
                )
            ])
            fig.update_layout(
                showlegend=False,
                margin=dict(t=0,b=0,l=0,r=0),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

# --- TAB 2: RFM / Segmentation ---
with t2:
    st.subheader("Customer Segmentation (RFM + KMeans)")
    rfm = calculate_rfm(df, schema)
    if not rfm.empty:
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
        k = st.slider("K Clusters", 2, 6, 3)
        rfm['Cluster'] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(rfm_scaled)
        
        c1, c2 = st.columns([1, 2])
        c1.dataframe(
            rfm.groupby('Cluster')[['Recency','Frequency','Monetary']].mean().round(1),
            use_container_width=True
        )
        c2.plotly_chart(
            px.scatter_3d(
                rfm,
                x='Recency', y='Frequency', z='Monetary',
                color='Cluster', opacity=0.6,
                title="Customer Clusters in RFM Space"
            ),
            use_container_width=True
        )
    else:
        st.info("Not enough data or missing columns to compute RFM (need customer, date, and amount).")

# --- TAB 3: Recommender ---
with t3:
    st.subheader("Smart Recommender")
    # We reuse 'transactions' and 'rules' from Tab 1
    if 'transactions' in locals() and transactions:
        all_items = sorted(list(set([i for t in transactions for i in t])))
        cart = st.multiselect("Cart Items:", all_items)
        
        if cart and 'rules' in locals() and not rules.empty:
            recs = []
            for _, r in rules.iterrows():
                if set(r['antecedents']).issubset(set(cart)):
                    add_ons = set(r['consequents']) - set(cart)
                    if add_ons:
                        recs.append({"Item": ', '.join(sorted(add_ons)), "Lift": r['lift']})
            if recs:
                rec_df = pd.DataFrame(recs).sort_values("Lift", ascending=False)
                rec_df = rec_df.drop_duplicates("Item")
                st.dataframe(rec_df, use_container_width=True)
            else:
                st.info("No add-ons found for this cart based on current rules.")
        elif not rules.empty:
            st.info("Select items in the cart above to see recommendations.")
        else:
            st.info("No rules available. Please adjust support/metric or enrich the data in the Market Basket tab.")
    else:
        st.info("No transactions built yet. Please check the Market Basket tab first.")
