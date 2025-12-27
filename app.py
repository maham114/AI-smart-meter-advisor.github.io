# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from predictor import predict_energy



# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="AI Smart Meter Advisor", page_icon="⚡", layout="wide")

# ----------------------------
# Neon Dark UI CSS
# ----------------------------
st.markdown("""
<style>
.stApp { background: #0B0F1A; color: #FFFFFF; font-family: 'Segoe UI', sans-serif; }
h1,h2,h3,h4,h5,h6 { background: linear-gradient(135deg, #7F5AF0, #4D96FF);
                      -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.stDataFrameContainer div[data-testid="stVerticalBlock"] { background: rgba(255,255,255,0.05); border-radius:12px; padding:8px; }
.stDataFrameContainer th { background:#7F5AF0 !important; color:white !important; }
.stButton>button { background: linear-gradient(135deg, #7F5AF0, #4D96FF); color:white; border-radius:12px;
                   padding:8px 24px; font-weight:bold; }

.neon-card {
    background: #0B0F1A;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 0 20px rgba(127,90,240,0.6);
    transition: 0.3s;
}
.neon-card:hover {
    box-shadow: 0 0 40px rgba(127,90,240,0.8);
}
.neon-icon {
    font-size: 40px;
    color: #7F5AF0;
    text-shadow: 0 0 10px #7F5AF0, 0 0 20px #7F5AF0;
}
.neon-text {
    font-size: 20px;
    color: white;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Hero Section
# ----------------------------
st.markdown("""
<h1 style='text-align:center;'>⚡ AI Smart Meter Advisor</h1>
<p style='text-align:center;'>Predict electricity usage using AI & ML</p>
<hr>
""", unsafe_allow_html=True)

# ----------------------------
# Load Predictions
# ----------------------------
#hist_df, pred_df, avg_usage, ai_advice = predict_energy()
hist_df, pred_df, avg_usage, ai_advice, forecast_explanation = predict_energy()

# ----------------------------
# Historical Data
# ----------------------------
st.subheader("📂 Historical Usage")
with st.expander("Click to view historical daily usage"):
    st.dataframe(hist_df[['Date_only','Daily_kWh','Usage_Type']], width='stretch')

# ----------------------------
# Key Metrics (Neon Glow Cards)
# ----------------------------
st.subheader("📊 Key Insights")

peak_days = pred_df[pred_df["Usage_Type"]=="Peak"]
next_peak = str(peak_days.iloc[0]['Date'].date()) if not peak_days.empty else "No peak"
monthly_estimate = avg_usage * 30

col1, col2, col3 = st.columns(3)
col1.markdown(f"""
<div class="neon-card">
    <div class="neon-icon">⚡</div>
    <div class="neon-text">Average Usage<br>{avg_usage:.2f} kWh/day</div>
</div>
""", unsafe_allow_html=True)

col2.markdown(f"""
<div class="neon-card">
    <div class="neon-icon">📈</div>
    <div class="neon-text">Next Peak Day<br>{next_peak}</div>
</div>
""", unsafe_allow_html=True)

col3.markdown(f"""
<div class="neon-card">
    <div class="neon-icon">📅</div>
    <div class="neon-text">Days Predicted<br>{len(pred_df)}</div>
</div>
""", unsafe_allow_html=True)

# Extra card for monthly estimate
st.markdown(f"""
<div class="neon-card" style="margin-top:20px;">
    <div class="neon-icon">💡</div>
    <div class="neon-text">Estimated Monthly Consumption<br>{monthly_estimate:.1f} kWh</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Predicted Consumption Chart (Plotly)
# ----------------------------
st.subheader("📈 Predicted Electricity Consumption")

fig = px.line(
    pred_df,
    x="Date",
    y="Predicted_kWh",
    markers=True,
    title="7-Day Electricity Usage Forecast"
)
fig.update_traces(
    fill="tozeroy",
    fillcolor="rgba(127,90,240,0.25)",
    line=dict(width=3)
)
fig.update_layout(
    plot_bgcolor="#0B0F1A",
    paper_bgcolor="#0B0F1A",
    font_color="white",
    xaxis_title="Date",
    yaxis_title="Predicted kWh",
    title_x=0.5
)
st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# Forecast Explanation (Friendly)
# ----------------------------
st.subheader("📝 This Week's Forecast Summary")
st.markdown(f"""
<div style="color:white; font-size:16px; background: rgba(255,255,255,0.05); padding:12px; border-radius:12px;">
{forecast_explanation}
</div>
""", unsafe_allow_html=True)


# Prediction Table
st.subheader("📋 Upcoming 7-Day Usage")
st.dataframe(pred_df[['Date','Predicted_kWh','Usage_Type']], width='stretch')

# ----------------------------
# AI Advice Section
# ----------------------------
#st.subheader("🤖 AI Energy Saving Plan")
#with st.expander("Click to view AI-generated 7-day action plan"):
#   st.text(ai_advice)
# ----------------------------
# AI Advice Cards
# ----------------------------
st.markdown("""
<style>
.advice-card {
    background: rgba(255,255,255,0.06);
    padding: 16px;
    border-radius: 14px;
    margin-bottom: 14px;
    transition: 0.3s ease;
    border-left: 4px solid #7F5AF0;
    box-shadow: 0 0 12px rgba(127,90,240,0.35);
}
.advice-card:hover {
    background: rgba(127,90,240,0.25);
    transform: translateY(-4px);
    box-shadow: 0 0 22px rgba(127,90,240,0.6);
}
.day-title {
    font-weight: bold;
    margin-bottom: 6px;
    color: #7F5AF0;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# Split AI advice into lines
advice_dict = ai_advice
st.subheader("🤖 AI Energy Saving Plan")

cols = st.columns(3)

for i, row in pred_df.iterrows():
    date_str = row["Date"].strftime("%Y-%m-%d")
    day_name = row["Date"].strftime("%A")
    tip = advice_dict.get(date_str, "Use electricity efficiently today.")

    with cols[i % 3]:
        st.markdown(f"""
        <div class="advice-card">
            <div class="day-title">📅 {day_name}<br><small>{date_str}</small></div>
            <div>{tip}</div>
        </div>
        """, unsafe_allow_html=True)
# ----------------------------
