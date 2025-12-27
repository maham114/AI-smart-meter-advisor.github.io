

import warnings
from predictor import predict_energy

warnings.filterwarnings("ignore")  # Hide sklearn warnings

# -------------------------------
# 1. Get all data from predictor
# -------------------------------
historical, predictions, avg_usage, ai_advice = predict_energy()

# -------------------------------
# 2. Historical Daily Usage
# -------------------------------
print("\n" + "="*60)
print("🏠 Historical Daily Usage (Peak/Normal) 🏠")
print("="*60)
print(historical.head(10).to_string(index=False))
print(f"\nAverage daily usage: {avg_usage:.1f} kWh")
print("="*60 + "\n")

# -------------------------------
# 3. Next 7 Days Predictions
# -------------------------------
print("📅 Next 7 Days Predicted Usage 📅")
print("="*60)
for _, row in predictions.iterrows():
    date_str = row['Date'].strftime('%A, %d %b %Y')
    print(f"{date_str}: {row['Predicted_kWh']:.1f} kWh ({row['Usage_Type']})")
print("="*60 + "\n")

# -------------------------------
# 4. AI-Generated Energy Advice
# -------------------------------
print("🤖 AI-GENERATED 7-DAY ACTION PLAN 🤖")
print("="*60)
print(ai_advice)
print("="*60 + "\n")
