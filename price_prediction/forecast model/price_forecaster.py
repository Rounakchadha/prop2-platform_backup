from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from prophet import Prophet
import random

app = Flask(__name__)
CORS(app)

# Dummy price data for localities
LOCALITY_DATA = {
    "Andheri": [16000, 16500, 16200, 17000, 16800, 16400],
    "Bandra": [21000, 21500, 22000, 22500, 23000, 23500],
    "Thane": [9000, 9200, 9500, 9700, 9600, 9800],
    "Dadar": [18000, 18500, 19000, 18700, 18800, 18900],
    "Powai": [14000, 14500, 14200, 14600, 14900, 14700],
    "Goregaon": [11000, 11500, 11300, 11600, 11800, 12000]
}

@app.route("/locality-insights", methods=["GET"])
def locality_insights():
    locality = request.args.get("locality", "Andheri")
    prices = LOCALITY_DATA.get(locality, LOCALITY_DATA["Andheri"])
    
    df = pd.DataFrame({
        'ds': pd.date_range(start='2023-01-01', periods=len(prices), freq='M'),
        'y': prices
    })
    
    model = Prophet()
    model.fit(df)
    
    future = model.make_future_dataframe(periods=6, freq='M')
    forecast = model.predict(future)
    
    forecast_output = forecast[['ds', 'yhat']].tail(6).to_dict(orient='records')
    
    # Simulated yield
    avg_price = sum(prices) / len(prices)
    rental_yield = round(random.uniform(4.5, 7.5), 2)
    
    top_projects = [
        {"project_name": "Project A", "rental_yield": round(random.uniform(6.5, 8.0), 2)},
        {"project_name": "Project B", "rental_yield": round(random.uniform(5.5, 7.0), 2)},
        {"project_name": "Project C", "rental_yield": round(random.uniform(5.0, 6.5), 2)}
    ]
    
    response = {
        "locality": locality,
        "average_price_per_sqft": round(avg_price, 2),
        "average_rental_yield": rental_yield,
        "price_forecast": forecast_output,
        "top_projects_by_yield": top_projects
    }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)