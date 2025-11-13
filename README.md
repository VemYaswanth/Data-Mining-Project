
# Omnichannel Market Basket App (v2)

Enhanced version with a separate **Product Recommender tab**.

## Tabs
1. **Dataset & Rules** – Run Apriori/FP-Growth and visualize rules.
2. **🛍️ Product Recommender** – Select product(s) to get suggestions using association rules.
3. **📈 Comparison** – Benchmark FP-Growth vs Apriori.

## Run
```bash
pip install -r requirements.txt
streamlit run app_streamlit_v2.py
```

Use sidebar to upload or pick sample data (`chain_cleaned.csv` or `grocery_cleaned.csv`).
Then, mine rules and explore recommendations interactively!
https://dm-protov5.streamlit.app/ link to prototypr-V5
https://dm-dash.streamlit.app/ link to protoype-V4
