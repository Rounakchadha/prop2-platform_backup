import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the dataset
csv_path = "data/mumbai_prices_sample.csv"
df = pd.read_csv(csv_path)

# Define localities to compare
localities = ['Andheri', 'Bandra', 'Thane', 'Powai']

# Forecast horizon (months)
months = 6

# Prepare a plot
plt.figure(figsize=(12, 6))
colors = ['blue', 'green', 'red', 'purple']

for i, locality in enumerate(localities):
    df_local = df[df['locality'] == locality].copy()
    df_local.rename(columns={'date': 'ds', 'price_per_sqft': 'y'}, inplace=True)
    df_local['ds'] = pd.to_datetime(df_local['ds'])

    model = Prophet()
    model.fit(df_local)

    future = model.make_future_dataframe(periods=months, freq='M')
    forecast = model.predict(future)

    # Plot only the forecast segment
    forecast_segment = forecast[['ds', 'yhat']].tail(months)
    plt.plot(forecast_segment['ds'], forecast_segment['yhat'], label=locality, color=colors[i])

plt.title("6-Month Price Forecast Comparison by Locality")
plt.xlabel("Date")
plt.ylabel("Predicted Price per sqft")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()