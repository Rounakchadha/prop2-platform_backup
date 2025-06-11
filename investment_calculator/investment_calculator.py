def calculate_investment_details(price, down_payment_percent, loan_years, interest_rate, monthly_rent, maintenance):
    """
    Calculate comprehensive investment details including EMI, ROI, and loan information
    """
    # 1. Down Payment and Loan Amount
    down_payment = (down_payment_percent / 100) * price
    loan_amount = price - down_payment

    # 2. Monthly EMI (using standard formula)
    monthly_interest_rate = interest_rate / (12 * 100)
    total_months = loan_years * 12
    
    if monthly_interest_rate > 0:
        emi = (loan_amount * monthly_interest_rate * (1 + monthly_interest_rate) ** total_months) / \
              ((1 + monthly_interest_rate) ** total_months - 1)
    else:
        emi = loan_amount / total_months  # If no interest

    # 3. Total interest and repayment
    total_payment = emi * total_months
    total_interest = total_payment - loan_amount

    # 4. ROI = (Annual Rent - Annual Maintenance) / Price * 100
    annual_rent = monthly_rent * 12
    annual_maintenance = maintenance * 12
    net_annual_income = annual_rent - annual_maintenance
    roi = (net_annual_income / price) * 100

    # 5. Additional calculations
    monthly_cash_flow = monthly_rent - maintenance - emi
    annual_cash_flow = monthly_cash_flow * 12
    
    # 6. Break-even analysis
    break_even_years = price / net_annual_income if net_annual_income > 0 else float('inf')

    return {
        "down_payment": round(down_payment, 2),
        "loan_amount": round(loan_amount, 2),
        "monthly_emi": round(emi, 2),
        "total_interest": round(total_interest, 2),
        "total_repayment": round(total_payment, 2),
        "annual_rent_income": round(annual_rent, 2),
        "annual_maintenance": round(annual_maintenance, 2),
        "net_annual_income": round(net_annual_income, 2),
        "roi_percent": round(roi, 2),
        "monthly_cash_flow": round(monthly_cash_flow, 2),
        "annual_cash_flow": round(annual_cash_flow, 2),
        "break_even_years": round(break_even_years, 2) if break_even_years != float('inf') else None,
        "loan_years": loan_years,
        "interest_rate": interest_rate,
        "down_payment_percent": down_payment_percent
    }

class InvestmentCalculator:
    def __init__(self):
        pass
    
    def analyze_investment(self, locality, budget, horizon, risk_tolerance, market_data, 
                          down_payment_percent=20, interest_rate=8.5, maintenance_percent=2):
        """
        Enhanced investment analysis using the detailed calculator
        """
        try:
            # Get locality stats for rent estimation
            locality_data = market_data[market_data['locality'].str.contains(locality, na=False)]
            
            if locality_data.empty:
                return {"error": "Locality data not available"}
            
            # Calculate average rent and price for the locality
            avg_rent = locality_data['rent'].mean()
            avg_price_per_lakh = locality_data['price_lakh'].mean()
            
            # Convert budget from lakhs to actual price
            property_price = budget * 100000
            
            # Estimate monthly rent based on locality average (scaled by property price)
            estimated_monthly_rent = avg_rent * (budget / avg_price_per_lakh)
            
            # Calculate monthly maintenance (percentage of property price)
            monthly_maintenance = (property_price * maintenance_percent / 100) / 12
            
            # Use the detailed investment calculator
            investment_details = calculate_investment_details(
                price=property_price,
                down_payment_percent=down_payment_percent,
                loan_years=horizon,
                interest_rate=interest_rate,
                monthly_rent=estimated_monthly_rent,
                maintenance=monthly_maintenance
            )
            
            # Risk assessment
            risk_score = self.calculate_risk_score(investment_details, risk_tolerance)
            recommendation = self.get_investment_recommendation(investment_details, risk_score)
            
            # Format for frontend
            return {
                'locality': locality.title(),
                'budget': budget,
                'horizon': horizon,
                'property_price': round(property_price / 100000, 2),  # Convert back to lakhs
                'down_payment': round(investment_details['down_payment'] / 100000, 2),
                'loan_amount': round(investment_details['loan_amount'] / 100000, 2),
                'monthly_emi': round(investment_details['monthly_emi'], 2),
                'estimated_monthly_rent': round(estimated_monthly_rent, 2),
                'monthly_maintenance': round(monthly_maintenance, 2),
                'monthly_cash_flow': round(investment_details['monthly_cash_flow'], 2),
                'annual_roi': round(investment_details['roi_percent'], 2),
                'annual_rent_income': round(investment_details['annual_rent_income'], 2),
                'net_annual_income': round(investment_details['net_annual_income'], 2),
                'total_interest': round(investment_details['total_interest'] / 100000, 2),
                'break_even_years': investment_details['break_even_years'],
                'risk_score': risk_score,
                'recommendation': recommendation,
                'interest_rate': interest_rate,
                'down_payment_percent': down_payment_percent
            }
            
        except Exception as e:
            return {"error": f"Calculation error: {str(e)}"}
    
    def calculate_risk_score(self, investment_details, risk_tolerance):
        """Calculate risk score based on investment metrics"""
        base_risk = 0
        
        # Cash flow risk
        if investment_details['monthly_cash_flow'] < 0:
            base_risk += 40
        elif investment_details['monthly_cash_flow'] < 5000:
            base_risk += 20
        
        # ROI risk
        if investment_details['roi_percent'] < 4:
            base_risk += 30
        elif investment_details['roi_percent'] < 6:
            base_risk += 15
        
        # Break-even risk
        if investment_details['break_even_years'] and investment_details['break_even_years'] > 20:
            base_risk += 20
        elif investment_details['break_even_years'] and investment_details['break_even_years'] > 15:
            base_risk += 10
        
        # Adjust for risk tolerance
        risk_multipliers = {'low': 0.8, 'medium': 1.0, 'high': 1.2}
        final_risk = min(base_risk * risk_multipliers.get(risk_tolerance, 1.0), 100)
        
        return round(final_risk, 1)
    
    def get_investment_recommendation(self, investment_details, risk_score):
        """Generate investment recommendation"""
        roi = investment_details['roi_percent']
        cash_flow = investment_details['monthly_cash_flow']
        
        if roi > 8 and cash_flow > 0 and risk_score < 30:
            return "Highly Recommended"
        elif roi > 6 and cash_flow >= 0 and risk_score < 50:
            return "Recommended"
        elif roi > 4 and risk_score < 70:
            return "Consider with Caution"
        else:
            return "Not Recommended"
