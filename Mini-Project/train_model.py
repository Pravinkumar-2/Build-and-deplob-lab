import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

CSV_PATH = "electricity_data.csv"      # columns: Date, Temperature, Consumption

# ---------- 1. Data Collection ----------
def load_data():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    # Simulated daily data (same schema as a real dataset) used when no CSV is present
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    doy = dates.dayofyear.values
    temp = 28 + 6 * np.sin(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 1.5, len(dates))
    noise = np.zeros(len(dates))
    for t in range(1, len(dates)):
        noise[t] = 0.6 * noise[t - 1] + rng.normal(0, 6)
    weekend = (dates.dayofweek >= 5).astype(int)
    trend = np.linspace(0, 15, len(dates))
    kwh = 220 + 7 * (temp - 28) + 18 * weekend + trend + noise
    df = pd.DataFrame({"Date": dates, "Temperature": temp.round(1), "Consumption": kwh.round(2)})
    df.to_csv(CSV_PATH, index=False)
    return df

df = load_data()

# ---------- 2. Data Preprocessing ----------
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
df = df.set_index("Date").interpolate().reset_index()      # handle missing values

# ---------- 3. Feature Engineering ----------
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month
df["Lag_1"] = df["Consumption"].shift(1)                    # yesterday
df["Lag_7"] = df["Consumption"].shift(7)                    # same day last week
df = df.dropna().reset_index(drop=True)

FEATURES = ["Temperature", "DayOfWeek", "Month", "Lag_1", "Lag_7"]
X, y = df[FEATURES], df["Consumption"]

# ---------- 4. Chronological Train-Test Split (80-20) ----------
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# ---------- 5. Model Training ----------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------- 6. Prediction and Evaluation ----------
pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("Total records      :", len(df))
print("Training records   :", len(X_train))
print("Testing records    :", len(X_test))
print("Intercept          :", round(model.intercept_, 4))
for name, coef in zip(FEATURES, model.coef_):
    print(f"Coefficient {name:<12}: {coef:.4f}")
print(f"MAE                : {mae:.4f} kWh")
print(f"MSE                : {mse:.4f}")
print(f"RMSE               : {rmse:.4f} kWh")
print(f"R2 Score           : {r2:.4f}")

# ---------- 7. Visualization ----------
plt.figure(figsize=(10, 4.5))
plt.plot(df["Date"].iloc[split:], y_test.values, label="Actual", color="#1f77b4")
plt.plot(df["Date"].iloc[split:], pred, label="Predicted", color="#ff7f0e", linestyle="--")
plt.title("Electricity Consumption: Actual vs Predicted (Test Set)")
plt.xlabel("Date")
plt.ylabel("Consumption (kWh)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=200)

# ---------- 8. Save model for the web app ----------
joblib.dump(model, "model.pkl")
json.dump({"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "train": len(X_train), "test": len(X_test)},
          open("metrics.json", "w"))
