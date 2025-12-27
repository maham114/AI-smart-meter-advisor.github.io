# -----------------------------
# Config
# -----------------------------
import pandas as pd
import numpy as np
import requests
import os
from sklearn.ensemble import RandomForestRegressor

os.environ["HF_TOKEN"] = "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"

# CSV is in repo root (Git LFS)
DATA_PATH = "energy_data.csv"



# -----------------------------
# Data Loading
# -----------------------------
def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH, sep=';', low_memory=False)

    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df.drop(columns=['Date', 'Time'], inplace=True)

    numeric_columns = [
        'Global_active_power', 'Global_reactive_power',
        'Voltage', 'Global_intensity',
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].mean(), inplace=True)

    df = df[df['Global_active_power'] > 0]

    df['Date_only'] = df['DateTime'].dt.date
    df['Energy_kWh'] = df['Global_active_power'] / 60

    daily_usage = (
        df.groupby('Date_only')['Energy_kWh']
        .sum()
        .reset_index()
        .rename(columns={'Energy_kWh': 'Daily_kWh'})
    )

    return daily_usage

# -----------------------------
# Main Prediction Function
# -----------------------------
def predict_energy():

    # Historical usage
    daily_usage = load_and_prepare_data()
    avg_usage = daily_usage['Daily_kWh'].mean()

    daily_usage['Usage_Type'] = daily_usage['Daily_kWh'].apply(
        lambda x: 'Peak' if x > avg_usage else 'Normal'
    )

    # Train model
    daily_usage = daily_usage.sort_values('Date_only')
    daily_usage['Day_Num'] = np.arange(len(daily_usage))

    X = daily_usage[['Day_Num']]
    y = daily_usage['Daily_kWh']

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    # Predict next 7 days
    last_day = daily_usage['Day_Num'].iloc[-1]
    future_days = np.arange(last_day + 1, last_day + 8).reshape(-1, 1)

    future_predictions = model.predict(future_days)
    future_predictions += np.random.normal(
        0, daily_usage['Daily_kWh'].std() * 0.05, size=future_predictions.shape
    )

    future_dates = pd.date_range(
        start=pd.to_datetime(daily_usage['Date_only'].iloc[-1]) + pd.Timedelta(days=1),
        periods=7
    )

    prediction_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_kWh': future_predictions
    })

    prediction_df['Usage_Type'] = prediction_df['Predicted_kWh'].apply(
        lambda x: 'Peak' if x > avg_usage else 'Normal'
    )

    # -----------------------------
    # Build Forecast Table for LLM
    # -----------------------------
    forecast_table = ""
    for _, row in prediction_df.iterrows():
        forecast_table += (
            f"{row['Date'].strftime('%Y-%m-%d')} | "
            f"{row['Predicted_kWh']:.1f} kWh | "
            f"{row['Usage_Type']}\n"
        )

    # -----------------------------
    # LLM Prompt: 3 tips per day
    # -----------------------------
    prompt = f"""
You are a friendly energy efficiency advisor.

Here is a 7-day electricity consumption forecast:

DATE | USAGE | TYPE
{forecast_table}

Task:
- For each day, provide exactly 3 practical energy-saving tips.
- Tips must relate to the usage type (Peak or Normal)
- Output STRICTLY in this format:

YYYY-MM-DD | Tip 1; Tip 2; Tip 3

Rules:
- One line per date
- Date format must match exactly (YYYY-MM-DD)
- Tips should be concise and practical
- Do NOT include extra text or headings
"""

    # -----------------------------
    # Call HuggingFace API
    # -----------------------------
    API_URL = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-ai/DeepSeek-V3.2:novita",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 700
    }

    advice_dict = {}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            for line in content.split("\n"):
                if "|" in line:
                    date, tips = line.split("|", 1)
                    advice_dict[date.strip()] = tips.strip()
        else:
            raise Exception("LLM error")
    except Exception:
        # Fallback: 3 generic tips per day
        for _, row in prediction_df.iterrows():
            date_str = row['Date'].strftime("%Y-%m-%d")
            if row['Usage_Type'] == "Peak":
                advice_dict[date_str] = "Turn off unused appliances; Avoid peak hour usage; Use energy-efficient lights"
            else:
                advice_dict[date_str] = "Maintain efficient energy habits; Check for unnecessary loads; Use natural light"

    # -----------------------------
    # Forecast explanation
    # -----------------------------
    forecast_explanation = generate_forecast_explanation(prediction_df)

    return daily_usage, prediction_df, avg_usage, advice_dict, forecast_explanation

# -----------------------------
# Forecast Explanation Function
# -----------------------------
def generate_forecast_explanation(pred_df):
    forecast_summary = ""
    for _, row in pred_df.iterrows():
        day_name = row['Date'].strftime("%A")
        date_str = row['Date'].strftime("%Y-%m-%d")
        kwh = round(row['Predicted_kWh'], 1)
        usage_type = row['Usage_Type']
        forecast_summary += f"{date_str} ({day_name}): {kwh} kWh ({usage_type})\n"

    prompt = f"""
You are a friendly energy advisor.

Here is a 7-day electricity forecast:

{forecast_summary}

Write a short, engaging paragraph summarizing this week's energy usage.
- Mention which days are likely peak and which are normal
- Give it a friendly tone as if explaining to a regular household user
- Keep it concise (3-5 sentences)
- Use future tense
"""

    API_URL = "https://router.huggingface.co/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}", "Content-Type": "application/json"}
    payload = {"model": "deepseek-ai/DeepSeek-V3.2:novita", "messages": [{"role": "user", "content": prompt}], "max_tokens": 150}

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            ai_response = response.json()
            explanation = ai_response["choices"][0]["message"]["content"] if "choices" in ai_response else "This week's forecast is ready."
        else:
            explanation = "This week's forecast is ready. Monitor your usage daily and try to save energy on peak days."
    except Exception:
        explanation = "This week's forecast is ready. Monitor your usage daily and try to save energy on peak days."

    return explanation


