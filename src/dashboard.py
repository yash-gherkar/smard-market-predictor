import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="SMARD Market Desk", layout="wide", page_icon="⚡")

st.title("⚡ SMARD Market Inference Desk")
st.markdown("Day-ahead wholesale electricity price and volatility modeling.")

with st.sidebar:
    st.header("Generation Forecast (MWh)")
    st.markdown("Adjust parameters to simulate market conditions.")
    wind_on = st.number_input("Wind Onshore", min_value=0.0, value=15000.0, step=500.0)
    wind_off = st.number_input("Wind Offshore", min_value=0.0, value=5000.0, step=500.0)
    solar = st.number_input("Solar PV", min_value=0.0, value=10000.0, step=500.0)
    gas = st.number_input("Fossil Gas", min_value=0.0, value=8000.0, step=500.0)
    
    st.divider()
    predict_btn = st.button("Execute Inference", type="primary", use_container_width=True)

total_gen = wind_on + wind_off + solar + gas
ren_share = ((wind_on + wind_off + solar) / total_gen) * 100 if total_gen > 0 else 0

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Total Selected Generation", f"{total_gen:,.0f} MWh")
kpi2.metric("Renewable Share", f"{ren_share:.1f} %")
kpi3.metric("Fossil Gas Dependency", f"{gas:,.0f} MWh")

st.divider()

if predict_btn:
    payload = {
        "wind_onshore": wind_on,
        "wind_offshore": wind_off,
        "solar_pv": solar,
        "fossil_gas": gas
    }
    try:
        with st.spinner("Querying regression & classification models..."):
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                prob = result["spike_probability"]
                exact_price = result["predicted_price"]
                
                # Layout for Results
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.metric(label="Predicted Clearing Price", value=f"€ {exact_price:,.2f} / MWh")
                    
                    if result["spike_prediction"] == 1:
                        st.error(f"🚨 **HIGH VOLATILITY ALERT:** {result['status']}")
                    else:
                        st.success(f"🟢 **MARKET STABLE:** {result['status']}")
                
                with res_col2:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob,
                        number={"suffix": "%"},
                        title={'text': "Volatility Confidence (Probability of Spike)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#e74c3c" if prob >= 50 else "#2ecc71"},
                            'steps': [
                                {'range': [0, 50], 'color': "rgba(46, 204, 113, 0.1)"},
                                {'range': [50, 100], 'color': "rgba(231, 76, 60, 0.1)"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 3},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig_gauge.update_layout(margin=dict(t=50, b=20, l=20, r=20), height=250)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
            else:
                st.warning("API connection failed.")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Ensure the FastAPI server (main.py) is running in another terminal.")

st.divider()

col_chart1, col_chart2 = st.columns([1, 2])

with col_chart1:
    st.subheader("Forecast Breakdown")
    df_mix = pd.DataFrame({
        "Source": ["Wind Onshore", "Wind Offshore", "Solar PV", "Fossil Gas"],
        "Category": ["Renewable", "Renewable", "Renewable", "Fossil"],
        "MWh": [wind_on, wind_off, solar, gas]
    })
    fig_tree = px.treemap(
        df_mix, 
        path=["Category", "Source"], 
        values="MWh",
        color="Category",
        color_discrete_map={"Renewable": "#2ecc71", "Fossil": "#e74c3c"}
    )
    fig_tree.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=350)
    st.plotly_chart(fig_tree, use_container_width=True)

with col_chart2:
    st.subheader("Historical Market Context (Last 7 Days)")
    @st.cache_data
    def load_historical_data():
        gen = pd.read_parquet('cleaned_generation.parquet')
        prices = pd.read_parquet('cleaned_prices.parquet')
        df = pd.merge(gen, prices, on=['Start date', 'End date'], how='inner')
        price_col = 'Germany/Luxembourg [€/MWh] Calculated resolutions'
        df[price_col] = pd.to_numeric(df[price_col].astype(str).str.replace(',', '', regex=False), errors='coerce')
        return df.dropna(subset=[price_col]).tail(168).reset_index(drop=True)

    try:
        hist_df = load_historical_data()
        price_col = 'Germany/Luxembourg [€/MWh] Calculated resolutions'
        
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=hist_df.index, 
            y=hist_df[price_col], 
            fill='tozeroy',
            mode='lines',
            line=dict(color='#3498db', width=2),
            name="Market Price"
        ))
        fig_line.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Spike Threshold (>100 €)")
        fig_line.update_layout(
            margin=dict(t=0, b=0, l=0, r=0), 
            height=350,
            xaxis_title="Hours",
            yaxis_title="Price (€/MWh)"
        )
        st.plotly_chart(fig_line, use_container_width=True)
        
    except Exception:
        st.info("Historical data not found. Ensure parquet files are in the root directory to view trends.")