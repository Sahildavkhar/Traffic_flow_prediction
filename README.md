# Traffic Flow Prediction 🚦

**Preview:** Predicting upcoming traffic volume using environmental & time-based factors, with interactive dashboard visualisation.

---

## 📖 Project Overview  
This project aims to analyse historic hour-by-hour traffic volume data, learn relationships between weather/time features and traffic flow, and then **forecast traffic volume for the next few hours**.  
It uses the same parameters as the dataset:  
`holiday`, `temp`, `rain_1h`, `snow_1h`, `clouds_all`, `weather_main`, `weather_description`, `date_time`, `traffic_volume`.

---

## ✅ Key Features  
- Uses a dataset with the above parameters (either real historic or synthetic for recent year 2025)  
- Pre-processing of date/time features (hour, day of week, weekend, etc.)  
- Model built with Linear Regression (scikit-learn) to predict traffic volume  
- Forecasting module: predicts traffic for the next few hours based on latest data  
- Interactive dashboard built using Streamlit — allows uploading dataset, tweaking model parameters, visualising results, and downloading predictions  
- Visualisations include: Actual vs Predicted scatter, Residual error distribution, Time-series comparison of past vs forecast, Hourly traffic distribution  
- Exportable CSV of forecasted results

---

## 🧰 Tech Stack & Dependencies  
- Python 3.8+  
- Libraries: `pandas`, `numpy`, `scikit-learn`, `streamlit`, `plotly`  
- To install dependencies:  
  ```bash
  pip install pandas numpy scikit-learn streamlit plotly

## To run the app locally:
```bash
streamlit run traffic_forecast_app.py

