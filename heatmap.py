# heatmap.py

import pandas as pd
import folium
from folium.plugins import HeatMap

def create_heatmap():
    # Load the datasets
    final_df = pd.read_csv("data/Final_Project.csv")
    map_df = pd.read_csv("data/Map_Location.csv")

    # Clean and standardize 'Region'
    final_df['Region'] = final_df['Region'].str.strip().str.lower()
    map_df['Region'] = map_df['Region'].str.strip().str.lower()

    # Compute average metrics
    avg_metrics = final_df.groupby('Region').agg({
        'Price_Lakh': 'mean',
        'Rate_SqFt': 'mean'
    }).rename(columns={'Price_Lakh': 'Avg_Price_Lakh', 'Rate_SqFt': 'Avg_Rate_SqFt'}).reset_index()

    # Merge
    merged = pd.merge(map_df, avg_metrics, on='Region', how='inner')
    merged['Annual_Rent'] = merged['Avg_Price_Lakh'] * 100000 * 0.025 * 12
    merged['ROI (%)'] = (merged['Annual_Rent'] / (merged['Avg_Price_Lakh'] * 100000)) * 100

    # Create Map
    mumbai_map = folium.Map(location=[19.0760, 72.8777], zoom_start=11)
    heat_data = [
        [row['Latitude'], row['Longitude'], row['ROI (%)']]
        for _, row in merged.iterrows()
        if not pd.isnull(row['Latitude']) and not pd.isnull(row['Longitude'])
    ]
    HeatMap(heat_data, radius=12).add_to(mumbai_map)
    mumbai_map.save("static/mumbai_roi_heatmap.html")