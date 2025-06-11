import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

class MarketComparisonTool:
    def __init__(self):
        self.known_localities = [
            'andheri', 'bandra', 'bhandup', 'byculla', 'chembur', 'colaba', 'dadar', 'dharavi',
            'fort', 'ghatkopar', 'girgaon', 'goregaon', 'govandi', 'grant road', 'jogeshwari',
            'juhu', 'khar', 'kurla', 'lalbaug', 'lokhandwala', 'mahalakshmi', 'mahim',
            'malabar hill', 'malad', 'marine drive', 'masjid', 'matunga', 'mulund',
            'nariman point', 'parel', 'powai', 'prabhadevi', 'santacruz', 'sion', 'tardeo',
            'vidyavihar', 'vikhroli', 'vile parle', 'wadala', 'worli'
        ]
    
    def preprocess_data(self, price_df, rent_df):
        """Preprocess and clean the data"""
        # Clean column names
        price_df.columns = [col.strip().lower().replace(" ", "_") for col in price_df.columns]
        rent_df.columns = [col.strip().lower().replace(" ", "_") for col in rent_df.columns]
        
        # Rename columns for clarity
        price_df.rename(columns={"location": "locality"}, inplace=True)
        if "rent/month" in rent_df.columns:
            rent_df.rename(columns={"rent/month": "rent"}, inplace=True)
        
        # Clean and normalize locality strings
        price_df['locality'] = price_df['locality'].astype(str).str.lower().str.strip()
        rent_df['locality'] = rent_df['locality'].astype(str).str.lower().str.strip()
        
        # Extract clean locality names from verbose strings in price_df
        pattern = r'\b(' + '|'.join(self.known_localities) + r')\b'
        price_df['locality'] = price_df['locality'].str.extract(pattern, expand=False).fillna('unknown')
        
        # Merge data for ROI calculation
        merged = pd.merge(price_df, rent_df, on="locality", how="inner")
        
        # Calculate ROI (annual rent / property price)
        merged["roi"] = (merged["rent"] * 12) / (merged["price_lakh"] * 100000) * 100
        
        return merged
    
    def get_locality_summary(self, merged_data, locality_input):
        """Get summary statistics for a locality"""
        # Create summary table
        summary = merged_data.groupby("locality").agg({
            "price_lakh": ["mean", "min", "max", "std"],
            "rate_sqft": ["mean", "min", "max"],
            "rent": ["mean", "min", "max"],
            "roi": ["mean", "min", "max"]
        }).reset_index()
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns]
        
        # Match in summary
        loc_summary = summary[summary["locality"].str.contains(locality_input, na=False)]
        
        # Fallback: match in merged data if summary fails
        if loc_summary.empty:
            loc_summary = self.get_fallback_summary(merged_data, locality_input)
        
        return loc_summary
    
    def get_fallback_summary(self, merged_data, loc_input):
        """Fallback method to get summary when direct match fails"""
        match_df = merged_data[merged_data["locality"].str.contains(loc_input, na=False)]
        if not match_df.empty:
            top_loc = match_df["locality"].value_counts().idxmax()
            fallback_data = match_df[match_df["locality"] == top_loc].groupby("locality").agg({
                "price_lakh": ["mean", "min", "max", "std"],
                "rate_sqft": ["mean", "min", "max"],
                "rent": ["mean", "min", "max"],
                "roi": ["mean", "min", "max"]
            }).reset_index()
            
            # Flatten column names
            fallback_data.columns = ['_'.join(col).strip('_') for col in fallback_data.columns]
            return fallback_data
        
        return pd.DataFrame()
    
    def compare_localities(self, loc1_input, loc2_input, market_data):
        """
        Compare two localities and return comprehensive comparison data
        
        Args:
            loc1_input (str): First locality name
            loc2_input (str): Second locality name
            market_data (DataFrame): Combined market data
            
        Returns:
            dict: Comparison results formatted for frontend
        """
        try:
            # Preprocess if needed (assuming market_data is already processed)
            if 'roi' not in market_data.columns:
                # If ROI not calculated, calculate it
                market_data["roi"] = (market_data["rent"] * 12) / (market_data["price_lakh"] * 100000) * 100
            
            # Get summaries for both localities
            loc1_summary = self.get_locality_summary(market_data, loc1_input.lower().strip())
            loc2_summary = self.get_locality_summary(market_data, loc2_input.lower().strip())
            
            if loc1_summary.empty or loc2_summary.empty:
                return None
            
            # Extract data for comparison
            loc1_data = {
                'name': loc1_summary.iloc[0]['locality'].title(),
                'stats': {
                    'avg_price': round(loc1_summary.iloc[0].get('price_lakh_mean', 0), 2),
                    'price_range': {
                        'min': round(loc1_summary.iloc[0].get('price_lakh_min', 0), 2),
                        'max': round(loc1_summary.iloc[0].get('price_lakh_max', 0), 2)
                    },
                    'avg_rent': round(loc1_summary.iloc[0].get('rent_mean', 0), 2),
                    'avg_roi': round(loc1_summary.iloc[0].get('roi_mean', 0), 2),
                    'roi_range': {
                        'min': round(loc1_summary.iloc[0].get('roi_min', 0), 2),
                        'max': round(loc1_summary.iloc[0].get('roi_max', 0), 2)
                    },
                    'avg_rate_sqft': round(loc1_summary.iloc[0].get('rate_sqft_mean', 0), 2)
                }
            }
            
            loc2_data = {
                'name': loc2_summary.iloc[0]['locality'].title(),
                'stats': {
                    'avg_price': round(loc2_summary.iloc[0].get('price_lakh_mean', 0), 2),
                    'price_range': {
                        'min': round(loc2_summary.iloc[0].get('price_lakh_min', 0), 2),
                        'max': round(loc2_summary.iloc[0].get('price_lakh_max', 0), 2)
                    },
                    'avg_rent': round(loc2_summary.iloc[0].get('rent_mean', 0), 2),
                    'avg_roi': round(loc2_summary.iloc[0].get('roi_mean', 0), 2),
                    'roi_range': {
                        'min': round(loc2_summary.iloc[0].get('roi_min', 0), 2),
                        'max': round(loc2_summary.iloc[0].get('roi_max', 0), 2)
                    },
                    'avg_rate_sqft': round(loc2_summary.iloc[0].get('rate_sqft_mean', 0), 2)
                }
            }
            
            # Calculate comparison metrics
            price_difference = loc2_data['stats']['avg_price'] - loc1_data['stats']['avg_price']
            roi_difference = loc2_data['stats']['avg_roi'] - loc1_data['stats']['avg_roi']
            rent_difference = loc2_data['stats']['avg_rent'] - loc1_data['stats']['avg_rent']
            
            # Determine better options
            better_roi = loc1_data['name'] if loc1_data['stats']['avg_roi'] > loc2_data['stats']['avg_roi'] else loc2_data['name']
            better_price = loc1_data['name'] if loc1_data['stats']['avg_price'] < loc2_data['stats']['avg_price'] else loc2_data['name']
            better_rent = loc1_data['name'] if loc1_data['stats']['avg_rent'] > loc2_data['stats']['avg_rent'] else loc2_data['name']
            
            # Generate chart data for frontend
            chart_data = self.generate_comparison_chart_data(loc1_data, loc2_data)
            
            return {
                'loc1': loc1_data,
                'loc2': loc2_data,
                'comparison': {
                    'price_difference': round(price_difference, 2),
                    'roi_difference': round(roi_difference, 2),
                    'rent_difference': round(rent_difference, 2),
                    'better_roi': better_roi,
                    'better_price': better_price,
                    'better_rent': better_rent
                },
                'chart_data': chart_data,
                'summary': self.generate_comparison_summary(loc1_data, loc2_data)
            }
            
        except Exception as e:
            print(f"Error in comparison: {e}")
            return None
    
    def generate_comparison_chart_data(self, loc1_data, loc2_data):
        """Generate data for frontend charts"""
        return {
            'labels': ['Price (₹L)', 'Rent (₹)', 'ROI (%)', 'Rate/sqft (₹)'],
            'loc1_values': [
                loc1_data['stats']['avg_price'],
                loc1_data['stats']['avg_rent'],
                loc1_data['stats']['avg_roi'],
                loc1_data['stats']['avg_rate_sqft']
            ],
            'loc2_values': [
                loc2_data['stats']['avg_price'],
                loc2_data['stats']['avg_rent'],
                loc2_data['stats']['avg_roi'],
                loc2_data['stats']['avg_rate_sqft']
            ]
        }
    
    def generate_comparison_summary(self, loc1_data, loc2_data):
        """Generate a summary of the comparison"""
        summary = []
        
        # Price comparison
        if loc1_data['stats']['avg_price'] < loc2_data['stats']['avg_price']:
            summary.append(f"{loc1_data['name']} is more affordable with lower average prices")
        else:
            summary.append(f"{loc2_data['name']} is more affordable with lower average prices")
        
        # ROI comparison
        if loc1_data['stats']['avg_roi'] > loc2_data['stats']['avg_roi']:
            summary.append(f"{loc1_data['name']} offers better ROI for investors")
        else:
            summary.append(f"{loc2_data['name']} offers better ROI for investors")
        
        # Rent comparison
        if loc1_data['stats']['avg_rent'] > loc2_data['stats']['avg_rent']:
            summary.append(f"{loc1_data['name']} has higher rental income potential")
        else:
            summary.append(f"{loc2_data['name']} has higher rental income potential")
        
        return summary
    
    def get_available_localities(self, market_data):
        """Get list of available localities for dropdown"""
        return sorted(market_data['locality'].unique())
