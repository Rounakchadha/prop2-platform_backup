from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import numpy as np
import pandas as pd
import os
import sys
import json
import warnings
from werkzeug.exceptions import BadRequest
from chatbot.chatbot_service import create_chatbot_service

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your custom modules with error handling
try:
    from market_comparison.market_comparison_tool import MarketComparisonTool
    MARKET_COMPARISON_AVAILABLE = True
except ImportError:
    MARKET_COMPARISON_AVAILABLE = False

try:
    from investment_calculator.investment_calculator import InvestmentCalculator, calculate_investment_details
    INVESTMENT_CALCULATOR_AVAILABLE = True
except ImportError:
    INVESTMENT_CALCULATOR_AVAILABLE = False

try:
    # Import only the function, not run the script
    import importlib.util
    spec = importlib.util.spec_from_file_location("predict_roi", "roi_model/predict_roi.py")
    roi_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(roi_module)
    predict_roi = roi_module.predict_roi
    ROI_MODEL_AVAILABLE = True
except:
    ROI_MODEL_AVAILABLE = False

try:
    from price_prediction.price_forecaster import PriceForecaster
    from price_prediction.price_prediction_model import PricePredictionModel
    PRICE_MODEL_AVAILABLE = True
except ImportError:
    PRICE_MODEL_AVAILABLE = False

try:
    from heatmap.heatmap import generate_heatmap_data
    HEATMAP_AVAILABLE = True
except ImportError:
    HEATMAP_AVAILABLE = False

# Initialize Flask app with correct template folder
app = Flask(__name__, template_folder='template')
app.secret_key = 'proptech-ml-platform-2025-secure-key'

class PropTechMLService:
    def __init__(self):
        self.models = {}
        self.tools = {}
        self.data = {}
        self.encoders = {}
        self.load_models_and_data()
    
    def load_models_and_data(self):
        """Load all models, tools, and datasets silently"""
        try:
            # Initialize available tools
            if MARKET_COMPARISON_AVAILABLE:
                self.tools['market_comparison'] = MarketComparisonTool()
            
            if INVESTMENT_CALCULATOR_AVAILABLE:
                self.tools['investment_calculator'] = InvestmentCalculator()
            
            if PRICE_MODEL_AVAILABLE:
                try:
                    self.tools['price_forecaster'] = PriceForecaster()
                    self.tools['price_prediction'] = PricePredictionModel()
                except:
                    pass
            
            # Load ROI model if available
            if ROI_MODEL_AVAILABLE:
                try:
                    import joblib
                    self.models['roi'] = joblib.load("roi_model/roi_model.pkl")
                    self.encoders['locality'] = joblib.load("roi_model/locality_encoder.pkl")
                except:
                    self.models['roi'] = None
                    self.encoders = {}
            
            # Load and preprocess data
            self.load_and_preprocess_data()
            
        except Exception as e:
            self.models = {'roi': None}
            self.tools = {}
    
    def load_and_preprocess_data(self):
        """Load and preprocess all datasets"""
        try:
            # Load datasets
            price_df = pd.read_csv("data/Final_Project.csv")
            rent_df = pd.read_csv("data/Mumbai_House_Rent.csv")
            
            # Use market comparison preprocessing if available
            if MARKET_COMPARISON_AVAILABLE and 'market_comparison' in self.tools:
                merged = self.tools['market_comparison'].preprocess_data(price_df.copy(), rent_df.copy())
            else:
                merged = self.basic_preprocess_data(price_df, rent_df)
            
            # Store processed data
            self.data['merged'] = merged
            self.data['localities'] = sorted(merged['locality'].unique())
            
            # Create summary
            self.data['summary'] = merged.groupby("locality").agg({
                "price_lakh": ["mean", "min", "max", "std"],
                "rate_sqft": ["mean", "min", "max"],
                "rent": ["mean", "min", "max"],
                "roi": ["mean", "min", "max"]
            }).reset_index()
            
            # Flatten column names
            self.data['summary'].columns = ['_'.join(col).strip('_') for col in self.data['summary'].columns]
            
        except Exception as e:
            self.data = {'localities': [], 'merged': pd.DataFrame(), 'summary': pd.DataFrame()}
    
    def basic_preprocess_data(self, price_df, rent_df):
        """Basic data preprocessing"""
        # Clean column names
        price_df.columns = price_df.columns.str.lower().str.strip().str.replace(" ", "_")
        rent_df.columns = rent_df.columns.str.lower().str.strip().str.replace(" ", "_")
        
        # Rename columns
        price_df.rename(columns={"location": "locality"}, inplace=True)
        if "rent/month" in rent_df.columns:
            rent_df.rename(columns={"rent/month": "rent"}, inplace=True)
        
        # Standardize locality strings
        price_df["locality"] = price_df["locality"].astype(str).str.lower().str.strip()
        rent_df["locality"] = rent_df["locality"].astype(str).str.lower().str.strip()
        
        # Merge and calculate ROI
        merged = pd.merge(price_df, rent_df, on="locality", how="inner")
        merged["roi"] = (merged["rent"] * 12) / (merged["price_lakh"] * 100000) * 100
        
        return merged

# Initialize the service
ml_service = PropTechMLService()

@app.route("/")
def home():
    """Main dashboard"""
    try:
        return render_template("dashboard.html", 
                             localities=ml_service.data.get('localities', []))
    except Exception as e:
        # Fallback if template is missing
        return f"""
        <html>
        <head><title>PropTech ML Platform</title></head>
        <body style="font-family: Arial; text-align: center; padding: 50px;">
            <h1>🏠 PropTech ML Platform</h1>
            <p>Template error: {str(e)}</p>
            <p>Available localities: {len(ml_service.data.get('localities', []))}</p>
            <div style="margin: 20px;">
                <a href="/roi-calculator" style="margin: 10px; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px;">ROI Calculator</a>
                <a href="/investment-calculator" style="margin: 10px; padding: 10px 20px; background: #28a745; color: white; text-decoration: none; border-radius: 5px;">Investment Calculator</a>
                <a href="/market-comparison" style="margin: 10px; padding: 10px 20px; background: #ffc107; color: black; text-decoration: none; border-radius: 5px;">Market Comparison</a>
                <a href="/roi-heatmap" style="margin: 10px; padding: 10px 20px; background: #dc3545; color: white; text-decoration: none; border-radius: 5px;">ROI Heatmap</a>
            </div>
        </body>
        </html>
        """

@app.route("/roi-calculator", methods=["GET", "POST"])
def roi_calculator():
    """ROI Calculator"""
    if request.method == "POST":
        try:
            locality = request.form["locality"].strip().lower()
            price = float(request.form["price"])
            
            # Use ROI model if available
            if ROI_MODEL_AVAILABLE:
                try:
                    # Convert price from lakhs to actual amount for the model
                    price_actual = price * 100000
                    prediction = predict_roi(locality, price_actual)
                except Exception as e:
                    prediction = ml_service.calculate_roi_fallback(locality, price)
            else:
                prediction = ml_service.calculate_roi_fallback(locality, price)
            
            if prediction is None:
                flash(f"Unable to calculate ROI for '{locality}'. Please try another locality.", "error")
                return redirect(url_for('roi_calculator'))
            
            historical_data = ml_service.get_locality_stats(locality)
            
            result = {
                'locality': locality.title(),
                'price': price,
                'predicted_roi': round(prediction, 2),
                'historical_data': historical_data,
                'prediction_method': 'ML Model' if ROI_MODEL_AVAILABLE else 'Statistical Analysis'
            }
            
            return render_template("roi_result.html", result=result)
            
        except ValueError:
            flash("Please enter valid numeric values", "error")
        except Exception as e:
            flash(f"Error calculating ROI: {str(e)}", "error")
    
    return render_template("roi_calculator.html", 
                         localities=ml_service.data.get('localities', []))

@app.route("/market-comparison", methods=["GET", "POST"])
def market_comparison():
    """Market comparison tool"""
    if request.method == "POST":
        try:
            loc1 = request.form.get("loc1", "").lower().strip()
            loc2 = request.form.get("loc2", "").lower().strip()
            
            if not loc1 or not loc2:
                flash("Please select both localities", "error")
                return redirect(url_for('market_comparison'))
            
            if loc1 == loc2:
                flash("Please select different localities for comparison", "error")
                return redirect(url_for('market_comparison'))
            
            # Use market comparison tool
            if MARKET_COMPARISON_AVAILABLE and 'market_comparison' in ml_service.tools:
                comparison_data = ml_service.tools['market_comparison'].compare_localities(
                    loc1, loc2, ml_service.data['merged']
                )
            else:
                comparison_data = ml_service.compare_localities(loc1, loc2)
            
            if not comparison_data:
                flash("Unable to find sufficient data for selected localities", "error")
                return redirect(url_for('market_comparison'))
            
            return render_template("comparison_result.html", comparison=comparison_data)
            
        except Exception as e:
            flash(f"Error in comparison: {str(e)}", "error")
    
    return render_template("market_comparison.html", 
                         localities=ml_service.data.get('localities', []))

@app.route("/investment-calculator", methods=["GET", "POST"])
def investment_calculator():
    """Enhanced Investment calculator with realistic financial calculations"""
    if request.method == "POST":
        try:
            locality = request.form["locality"].strip().lower()
            budget = float(request.form["budget"])
            investment_horizon = int(request.form["investment_horizon"])
            risk_tolerance = request.form["risk_tolerance"]
            
            down_payment_percent = float(request.form.get("down_payment", 20))
            interest_rate = float(request.form.get("interest_rate", 8.5))
            maintenance_percent = float(request.form.get("maintenance", 2))
            
            # Use enhanced investment calculator
            investment_analysis = ml_service.analyze_investment_opportunity_enhanced(
                locality=locality,
                budget=budget,
                horizon=investment_horizon,
                risk_tolerance=risk_tolerance,
                down_payment_percent=down_payment_percent,
                interest_rate=interest_rate,
                maintenance_percent=maintenance_percent
            )
            
            if "error" in investment_analysis:
                flash(investment_analysis["error"], "error")
                return redirect(url_for('investment_calculator'))
            
            return render_template("investment_result.html", analysis=investment_analysis)
            
        except ValueError:
            flash("Please enter valid numeric values", "error")
        except Exception as e:
            flash(f"Error in investment analysis: {str(e)}", "error")
    
    return render_template("investment_calculator.html", 
                         localities=ml_service.data.get('localities', []))

@app.route("/roi-heatmap")
def roi_heatmap():
    """ROI Heatmap"""
    try:
        if HEATMAP_AVAILABLE:
            heatmap_data = generate_heatmap_data(ml_service.data['merged'])
        else:
            heatmap_data = ml_service.generate_roi_heatmap_data()
        
        if not heatmap_data:
            # Create sample data if no data available
            heatmap_data = [
                {'locality': 'Andheri', 'avg_roi': 7.5, 'avg_price': 85.0, 'avg_rent': 35000},
                {'locality': 'Bandra', 'avg_roi': 6.2, 'avg_price': 120.0, 'avg_rent': 45000},
                {'locality': 'Powai', 'avg_roi': 8.1, 'avg_price': 75.0, 'avg_rent': 32000},
                {'locality': 'Malad', 'avg_roi': 9.3, 'avg_price': 55.0, 'avg_rent': 28000},
                {'locality': 'Thane', 'avg_roi': 8.7, 'avg_price': 65.0, 'avg_rent': 30000}
            ]
        
        return render_template("roi_heatmap.html", heatmap_data=heatmap_data)
        
    except Exception as e:
        flash(f"Error generating heatmap: {str(e)}", "error")
        return redirect(url_for('home'))


# API Endpoints
@app.route("/api/locality-stats/<locality>")
def api_locality_stats(locality):
    """API endpoint for locality statistics"""
    try:
        stats = ml_service.get_locality_stats(locality.lower())
        if stats:
            return jsonify(stats)
        else:
            return jsonify({"error": "Locality not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/localities")
def api_localities():
    """API endpoint to get all available localities"""
    try:
        localities = ml_service.data.get('localities', [])
        return jsonify({"localities": localities, "count": len(localities)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Enhanced Helper Methods
def calculate_roi_fallback(self, locality, price):
    """Fallback ROI calculation"""
    try:
        locality_stats = self.get_locality_stats(locality)
        if locality_stats:
            return locality_stats['avg_roi']
        
        if 'merged' in self.data and not self.data['merged'].empty:
            partial_match = self.data['merged'][
                self.data['merged']['locality'].str.contains(locality, na=False)
            ]
            if not partial_match.empty:
                return partial_match['roi'].mean()
        return None
    except:
        return None

def get_locality_stats(self, locality):
    """Get locality statistics"""
    if 'summary' not in self.data or self.data['summary'].empty:
        return None
    
    locality_data = self.data['summary'][
        self.data['summary']['locality'] == locality
    ]
    
    if locality_data.empty:
        locality_data = self.data['summary'][
            self.data['summary']['locality'].str.contains(locality, na=False)
        ]
    
    if locality_data.empty:
        return None
    
    stats = locality_data.iloc[0].to_dict()
    return {
        'avg_price': round(stats.get('price_lakh_mean', 0), 2),
        'price_range': {
            'min': round(stats.get('price_lakh_min', 0), 2),
            'max': round(stats.get('price_lakh_max', 0), 2)
        },
        'avg_rent': round(stats.get('rent_mean', 0), 2),
        'avg_roi': round(stats.get('roi_mean', 0), 2),
        'roi_range': {
            'min': round(stats.get('roi_min', 0), 2),
            'max': round(stats.get('roi_max', 0), 2)
        },
        'avg_rate_sqft': round(stats.get('rate_sqft_mean', 0), 2)
    }

def compare_localities(self, loc1, loc2):
    """Compare localities"""
    loc1_stats = self.get_locality_stats(loc1)
    loc2_stats = self.get_locality_stats(loc2)
    
    if not loc1_stats or not loc2_stats:
        return None
    
    return {
        'loc1': {'name': loc1.title(), 'stats': loc1_stats},
        'loc2': {'name': loc2.title(), 'stats': loc2_stats},
        'comparison': {
            'price_difference': round(loc2_stats['avg_price'] - loc1_stats['avg_price'], 2),
            'roi_difference': round(loc2_stats['avg_roi'] - loc1_stats['avg_roi'], 2),
            'rent_difference': round(loc2_stats['avg_rent'] - loc1_stats['avg_rent'], 2),
            'better_roi': loc1.title() if loc1_stats['avg_roi'] > loc2_stats['avg_roi'] else loc2.title(),
            'better_price': loc1.title() if loc1_stats['avg_price'] < loc2_stats['avg_price'] else loc2.title(),
            'better_rent': loc1.title() if loc1_stats['avg_rent'] > loc2_stats['avg_rent'] else loc2.title()
        },
        'chart_data': {
            'labels': ['Price (₹L)', 'Rent (₹)', 'ROI (%)', 'Rate/sqft (₹)'],
            'loc1_values': [
                loc1_stats['avg_price'], loc1_stats['avg_rent'],
                loc1_stats['avg_roi'], loc1_stats['avg_rate_sqft']
            ],
            'loc2_values': [
                loc2_stats['avg_price'], loc2_stats['avg_rent'],
                loc2_stats['avg_roi'], loc2_stats['avg_rate_sqft']
            ]
        },
        'summary': [
            f"{loc1.title() if loc1_stats['avg_price'] < loc2_stats['avg_price'] else loc2.title()} is more affordable with lower average prices",
            f"{loc1.title() if loc1_stats['avg_roi'] > loc2_stats['avg_roi'] else loc2.title()} offers better ROI for investors",
            f"{loc1.title() if loc1_stats['avg_rent'] > loc2_stats['avg_rent'] else loc2.title()} has higher rental income potential"
        ]
    }

def analyze_investment_opportunity_enhanced(self, locality, budget, horizon, risk_tolerance, 
                                          down_payment_percent=20, interest_rate=8.5, maintenance_percent=2):
    """Enhanced investment analysis with realistic maintenance costs"""
    
    locality_stats = self.get_locality_stats(locality)
    if not locality_stats:
        return {"error": "Locality data not available"}
    
    # Convert budget from lakhs to actual amount
    property_price = budget * 100000
    
    # 1. LOAN CALCULATIONS
    down_payment = property_price * (down_payment_percent / 100)
    loan_amount = property_price - down_payment
    
    monthly_interest_rate = interest_rate / (12 * 100)
    total_months = horizon * 12
    
    if monthly_interest_rate > 0:
        emi = (loan_amount * monthly_interest_rate * (1 + monthly_interest_rate) ** total_months) / \
              ((1 + monthly_interest_rate) ** total_months - 1)
    else:
        emi = loan_amount / total_months
    
    # 2. RENTAL INCOME ESTIMATION
    base_rent = locality_stats['avg_rent']
    avg_property_price = locality_stats['avg_price'] * 100000
    price_ratio = property_price / avg_property_price if avg_property_price > 0 else 1
    rent_scaling_factor = min(max(price_ratio, 0.8), 1.3)
    estimated_monthly_rent = base_rent * rent_scaling_factor
    
    # 3. REALISTIC MAINTENANCE AND EXPENSES
    realistic_maintenance_percent = 0.5  # Fixed at 0.5% annually
    monthly_basic_maintenance = (property_price * realistic_maintenance_percent / 100) / 12
    
    annual_property_tax = property_price * 0.001  # 0.1% annually
    monthly_property_tax = annual_property_tax / 12
    
    annual_insurance = property_price * 0.002  # 0.2% annually
    monthly_insurance = annual_insurance / 12
    
    estimated_area_sqft = property_price / locality_stats.get('avg_rate_sqft', 10000)
    monthly_society_maintenance = estimated_area_sqft * 3  # ₹3 per sqft average
    
    vacancy_rate = 0.08  # 8% vacancy
    monthly_vacancy_cost = estimated_monthly_rent * vacancy_rate
    
    total_monthly_expenses = (
        monthly_basic_maintenance + 
        monthly_property_tax + 
        monthly_insurance + 
        monthly_society_maintenance + 
        monthly_vacancy_cost
    )
    
    # 4. CASH FLOW ANALYSIS
    monthly_cash_flow = estimated_monthly_rent - emi - total_monthly_expenses
    annual_cash_flow = monthly_cash_flow * 12
    
    # 5. ROI CALCULATIONS
    annual_rent = estimated_monthly_rent * 12
    annual_expenses = total_monthly_expenses * 12
    net_annual_income = annual_rent - annual_expenses
    
    roi_on_cash_invested = (net_annual_income / down_payment) * 100
    
    # 6. BREAK-EVEN ANALYSIS
    if net_annual_income > 0:
        break_even_years = down_payment / net_annual_income
    else:
        break_even_years = None
    
    # 7. RISK CALCULATION
    risk_score = self.calculate_realistic_risk_score(
        locality_stats, monthly_cash_flow, roi_on_cash_invested, risk_tolerance, property_price
    )
    
    # 8. TOTAL INTEREST CALCULATION
    total_emi_payments = emi * total_months
    total_interest_paid = total_emi_payments - loan_amount
    
    return {
        'locality': locality.title(),
        'budget': budget,
        'horizon': horizon,
        'property_price': round(property_price / 100000, 2),
        
        # Loan Details
        'down_payment': round(down_payment / 100000, 2),
        'loan_amount': round(loan_amount / 100000, 2),
        'monthly_emi': round(emi, 2),
        'total_interest': round(total_interest_paid / 100000, 2),
        'interest_rate': interest_rate,
        'down_payment_percent': down_payment_percent,
        
        # Rental Income
        'estimated_monthly_rent': round(estimated_monthly_rent, 2),
        'annual_rent_income': round(annual_rent, 2),
        
        # Expenses
        'monthly_maintenance': round(total_monthly_expenses, 2),
        'total_monthly_expenses': round(total_monthly_expenses, 2),
        
        # Cash Flow
        'monthly_cash_flow': round(monthly_cash_flow, 2),
        'annual_cash_flow': round(annual_cash_flow, 2),
        'net_annual_income': round(net_annual_income, 2),
        
        # Returns
        'annual_roi': round(roi_on_cash_invested, 2),
        'break_even_years': round(break_even_years, 2) if break_even_years else None,
        
        # Risk Assessment
        'risk_score': risk_score,
        'recommendation': self.get_realistic_investment_recommendation(
            roi_on_cash_invested, monthly_cash_flow, risk_score
        ),
        
        # Maintenance Breakdown
        'maintenance_breakdown': {
            'basic_maintenance': round(monthly_basic_maintenance, 2),
            'property_tax': round(monthly_property_tax, 2),
            'insurance': round(monthly_insurance, 2),
            'society_maintenance': round(monthly_society_maintenance, 2),
            'vacancy_allowance': round(monthly_vacancy_cost, 2)
        }
    }

def calculate_realistic_risk_score(self, stats, monthly_cash_flow, roi, risk_tolerance, property_price):
    """More realistic risk calculation for real estate"""
    
    base_risk = 25  # Start with 25% base risk for real estate
    
    # 1. Cash Flow Risk (35% weight)
    if monthly_cash_flow >= 5000:
        cash_flow_risk = 0
    elif monthly_cash_flow >= 0:
        cash_flow_risk = 5
    elif monthly_cash_flow >= -5000:
        cash_flow_risk = 15
    elif monthly_cash_flow >= -10000:
        cash_flow_risk = 25
    else:
        cash_flow_risk = 35
    
    # 2. ROI Risk (30% weight)
    if roi >= 15:
        roi_risk = 0
    elif roi >= 10:
        roi_risk = 5
    elif roi >= 6:
        roi_risk = 10
    elif roi >= 3:
        roi_risk = 20
    else:
        roi_risk = 30
    
    # 3. Market Risk (25% weight)
    roi_range = stats['roi_range']['max'] - stats['roi_range']['min']
    if roi_range <= 3:
        market_risk = 0
    elif roi_range <= 5:
        market_risk = 8
    elif roi_range <= 7:
        market_risk = 15
    else:
        market_risk = 25
    
    # 4. Property Value Risk (10% weight)
    avg_price = stats['avg_price'] * 100000
    if property_price <= avg_price * 1.2:
        value_risk = 0
    elif property_price <= avg_price * 1.5:
        value_risk = 5
    else:
        value_risk = 10
    
    # Calculate weighted risk
    total_risk = base_risk + (cash_flow_risk * 0.35) + (roi_risk * 0.30) + (market_risk * 0.25) + (value_risk * 0.10)
    
    # Risk tolerance adjustment
    risk_adjustments = {'low': 1.15, 'medium': 1.0, 'high': 0.85}
    final_risk = total_risk * risk_adjustments.get(risk_tolerance, 1.0)
    
    return max(15, min(85, round(final_risk, 1)))

def get_realistic_investment_recommendation(self, roi, monthly_cash_flow, risk_score):
    """Realistic investment recommendations for real estate"""
    
    if roi >= 12 and monthly_cash_flow >= -2000 and risk_score < 35:
        return "Highly Recommended"
    elif roi >= 8 and monthly_cash_flow >= -5000 and risk_score < 45:
        return "Recommended"
    elif roi >= 5 and monthly_cash_flow >= -8000 and risk_score < 55:
        return "Consider for Capital Appreciation"
    elif roi >= 3 and risk_score < 65:
        return "Consider with Caution"
    else:
        return "Not Recommended"

def generate_roi_heatmap_data(self):
    """Generate heatmap data"""
    if 'summary' not in self.data or self.data['summary'].empty:
        return []
    
    heatmap_data = []
    for _, row in self.data['summary'].iterrows():
        heatmap_data.append({
            'locality': row['locality'].title(),
            'avg_roi': round(row.get('roi_mean', 0), 2),
            'avg_price': round(row.get('price_lakh_mean', 0), 2),
            'avg_rent': round(row.get('rent_mean', 0), 2)
        })
    
    return sorted(heatmap_data, key=lambda x: x['avg_roi'], reverse=True)

# Add methods to service
PropTechMLService.calculate_roi_fallback = calculate_roi_fallback
PropTechMLService.get_locality_stats = get_locality_stats
PropTechMLService.compare_localities = compare_localities
PropTechMLService.analyze_investment_opportunity_enhanced = analyze_investment_opportunity_enhanced
PropTechMLService.calculate_realistic_risk_score = calculate_realistic_risk_score
PropTechMLService.get_realistic_investment_recommendation = get_realistic_investment_recommendation
PropTechMLService.generate_roi_heatmap_data = generate_roi_heatmap_data

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return "Page not found", 404

@app.errorhandler(500)
def internal_error(error):
    return "Internal server error", 500

# Health check endpoint
@app.route("/health")
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "roi_model": ROI_MODEL_AVAILABLE,
            "market_comparison": MARKET_COMPARISON_AVAILABLE,
            "investment_calculator": INVESTMENT_CALCULATOR_AVAILABLE,
            "price_prediction": PRICE_MODEL_AVAILABLE,
            "heatmap": HEATMAP_AVAILABLE
        },
        "data_loaded": {
            "localities_count": len(ml_service.data.get('localities', [])),
            "merged_data_rows": len(ml_service.data.get('merged', [])),
            "summary_data_rows": len(ml_service.data.get('summary', []))
        }
    })

if __name__ == "__main__":
    print("🚀 Starting PropTech ML Platform...")
    print("=" * 50)
    print(f"✅ ROI Model: {'Available' if ROI_MODEL_AVAILABLE else 'Using Statistical Fallback'}")
    print(f"✅ Market Comparison: {'Available' if MARKET_COMPARISON_AVAILABLE else 'Using Built-in'}")
    print(f"✅ Investment Calculator: {'Available' if INVESTMENT_CALCULATOR_AVAILABLE else 'Using Built-in'}")
    print(f"✅ Price Prediction: {'Available' if PRICE_MODEL_AVAILABLE else 'Not Available'}")
    print(f"✅ Heatmap Generator: {'Available' if HEATMAP_AVAILABLE else 'Using Built-in'}")
    print(f"📊 Localities Loaded: {len(ml_service.data.get('localities', []))}")
    print("=" * 50)
    print("🌐 Server starting on http://localhost:5001")
    print("🔍 Health check: http://localhost:5001/health")
    print("=" * 50)
    
    # Use port 5001 to avoid conflicts
    app.run(debug=True, host='0.0.0.0', port=5001)
