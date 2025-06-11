from model.price_forecaster import forecast_prices

forecast = forecast_prices(
    csv_path="data/mumbai_prices_sample.csv",
    locality="Andheri",
    months=6,
    plot=True,
    save_plot=True,
    plot_path="andheri_forecast.png"
)

print(forecast)