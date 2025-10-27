import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
start_date = datetime(2025, 8, 1, 0, 0)   # Start date
end_date = datetime(2025, 10, 27, 23, 0)  # End date
hours = int((end_date - start_date).total_seconds() / 3600) + 1

# -------------------------------------------------
# WEATHER SIMULATION
# -------------------------------------------------
weather_conditions = [
    ("Clear", "sky is clear", 0, 0),
    ("Clouds", "overcast clouds", 0, 0),
    ("Rain", "light rain", 0.5, 0),
    ("Rain", "heavy rain", 2.0, 0),
    ("Snow", "light snow", 0, 0.5),
]

holidays = ["2025-08-15", "2025-09-02", "2025-10-02", "2025-10-25"]  # Example holidays

data = []

for i in range(hours):
    dt = start_date + timedelta(hours=i)
    dayofweek = dt.weekday()
    hour = dt.hour
    holiday = "Yes" if dt.strftime("%Y-%m-%d") in holidays else "No"

    # Random weather
    weather_main, weather_desc, rain_1h, snow_1h = random.choice(weather_conditions)
    clouds_all = np.clip(int(np.random.normal(50, 25)), 0, 100)
    temp = round(np.random.normal(295, 8), 1)  # around 22°C average

    # Base traffic pattern
    base = 1000 + 500 * np.sin((hour - 7) * np.pi / 12) + random.randint(-200, 200)

    # Rush hours
    if 7 <= hour <= 9 or 16 <= hour <= 18:
        base += random.randint(2000, 4000)

    # Weekends lower traffic
    if dayofweek >= 5:
        base -= random.randint(800, 1500)

    # Weather effect
    if weather_main == "Rain":
        base -= random.randint(200, 800)
    elif weather_main == "Snow":
        base -= random.randint(400, 1000)

    traffic_volume = max(int(base), 200)

    data.append({
        "holiday": holiday,
        "temp": temp,
        "rain_1h": rain_1h,
        "snow_1h": snow_1h,
        "clouds_all": clouds_all,
        "weather_main": weather_main,
        "weather_description": weather_desc,
        "date_time": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "traffic_volume": traffic_volume
    })

# -------------------------------------------------
# CREATE DATAFRAME
# -------------------------------------------------
df = pd.DataFrame(data)
df.to_csv("Traffic_Volume_2025.csv", index=False)

print("✅ Latest 2025 Traffic Dataset Generated Successfully!")
print(df.head(10))
print("\nTotal records:", len(df))
