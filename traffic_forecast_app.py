# Imports
# --------------------------------------------------
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# Streamlit Setup
# --------------------------------------------------
st.set_page_config(page_title="Smart Traffic Forecast Dashboard", layout="wide")
st.title("🚦 Smart Traffic Flow Prediction Dashboard")

# Load Dataset
# --------------------------------------------------
st.sidebar.header("📁 Dataset Options")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (Metro_Interstate_Traffic_Volume.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.stop()

# Preprocess
# --------------------------------------------------
df["date_time"] = pd.to_datetime(df["date_time"])
df = df.sort_values("date_time")
df["hour"] = df["date_time"].dt.hour
df["dayofweek"] = df["date_time"].dt.dayofweek
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

features = ["temp", "rain_1h", "snow_1h", "clouds_all", "hour", "is_weekend"]
label_col = "traffic_volume"
df = df[features + [label_col, "date_time"]].dropna()

# Model Training (Predict current hour traffic)
# --------------------------------------------------
X = df[features].values
y = df[label_col].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

col1, col2 = st.columns(2)
col1.metric("📊 RMSE", f"{rmse:.2f}")
col2.metric("📈 R² Score", f"{r2:.3f}")

st.markdown("---")

# Forecasting Next 6 Hours
# --------------------------------------------------
st.subheader("🔮 Forecasting Next 6 Hours of Traffic Volume")

# Use last row’s features as starting point
last_data = df.iloc[-1].copy()
future_data = []

for i in range(1, 7):
    next_hour = last_data["hour"] + i
    if next_hour >= 24:
        next_hour -= 24
    new_row = {
        "temp": last_data["temp"] + np.random.uniform(-2, 2),
        "rain_1h": max(0, last_data["rain_1h"] + np.random.uniform(-0.2, 0.5)),
        "snow_1h": max(0, last_data["snow_1h"] + np.random.uniform(-0.1, 0.3)),
        "clouds_all": np.clip(last_data["clouds_all"] + np.random.randint(-10, 10), 0, 100),
        "hour": next_hour,
        "is_weekend": 1 if last_data["is_weekend"] else 0
    }
    future_data.append(new_row)

future_df = pd.DataFrame(future_data)
future_df["predicted_traffic"] = model.predict(future_df[features])
future_df["date_time"] = pd.date_range(start=df["date_time"].iloc[-1] + pd.Timedelta(hours=1), periods=6, freq="H")

# Combine Past + Future
# --------------------------------------------------
plot_df = df[["date_time", label_col]].copy().tail(100)
plot_df["Type"] = "Actual"
forecast_df = future_df[["date_time", "predicted_traffic"]].rename(columns={"predicted_traffic": "traffic_volume"})
forecast_df["Type"] = "Forecast"
combined_df = pd.concat([plot_df, forecast_df], ignore_index=True)

# Visualization
# --------------------------------------------------
fig = px.line(combined_df, x="date_time", y="traffic_volume", color="Type",
              title="Traffic Volume: Actual (Past) vs Predicted (Next 6 Hours)",
              markers=True)
st.plotly_chart(fig, use_container_width=True)

# Show forecast table
# --------------------------------------------------
st.write("### 📅 Predicted Traffic for Next 6 Hours")
st.dataframe(future_df[["date_time", "predicted_traffic"]])

# Download Forecast CSV
# --------------------------------------------------
csv = future_df.to_csv(index=False).encode('utf-8')
st.download_button("💾 Download Forecast Data", data=csv, file_name="Next6Hour_Traffic_Predictions.csv", mime="text/csv")

st.success("✅ Forecast completed successfully!")
