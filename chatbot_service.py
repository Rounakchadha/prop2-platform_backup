import re
import json
import random
from datetime import datetime
import pandas as pd
import numpy as np

class PropTechChatbot:
    def __init__(self, ml_service):
        self.ml_service = ml_service
        self.conversation_history = []
        self.user_context = {}
        self.session_data = {}
        self.load_responses()
        self.load_knowledge_base()
    
    def load_responses(self):
        """Load predefined responses and patterns"""
        self.patterns = {
            'greeting': [
                r'hi|hello|hey|good morning|good afternoon|good evening|namaste',
                [
                    "Hello! I'm your PropTech investment assistant. How can I help you with Mumbai real estate today?",
                    "Hi there! Ready to explore Mumbai's real estate opportunities? What would you like to know?",
                    "Welcome to PropTech ML! I can help you with ROI calculations, market analysis, and investment advice. What interests you?",
                    "Namaste! I'm here to help you make smart real estate investments in Mumbai. What can I assist you with?"
                ]
            ],
            'roi_query': [
                r'roi|return on investment|calculate roi|what.*roi|roi.*(\d+).*lakh|(\d+).*lakh.*roi|returns|profit',
                [
                    "I can help you calculate ROI! Please provide the locality and property price. For example: 'Calculate ROI for ₹50L property in Andheri'",
                    "Let me calculate the ROI for you. Which locality and what property price are you considering?",
                    "ROI calculation is my specialty! Just tell me the area and budget, and I'll give you detailed analysis."
                ]
            ],
            'locality_info': [
                r'tell me about|information about|details about|what about|how is|locality|area|neighborhood',
                [
                    "I can provide detailed information about any Mumbai locality. Which area interests you?",
                    "I have comprehensive data on Mumbai localities. Which area would you like to explore?",
                    "I know Mumbai real estate inside out! Which locality would you like to learn about?"
                ]
            ],
            'investment_advice': [
                r'investment advice|recommend|suggest|best area|where to invest|crore budget|lakh budget|portfolio|strategy',
                [
                    "I'd be happy to provide investment advice! What's your budget and investment goals?",
                    "Let me help you find the best investment opportunities. What's your budget range?",
                    "Investment strategy is crucial! Tell me your budget and I'll create a personalized plan."
                ]
            ],
            'price_comparison': [
                r'compare|comparison|vs|versus|difference between|which is better|better option',
                [
                    "I can compare different localities for you. Which areas would you like me to compare?",
                    "Great! I'll help you compare localities. Please specify which areas you're interested in.",
                    "Comparison analysis coming up! Which two or more areas should I analyze?"
                ]
            ],
            'market_trends': [
                r'trends|market|growth|future|prediction|forecast|outlook|analysis',
                [
                    "Based on our data analysis, I can share market trends and predictions. What specific trend interests you?",
                    "I have insights on Mumbai real estate trends. Are you interested in price trends, ROI trends, or rental trends?",
                    "Market analysis is my forte! What aspect of Mumbai real estate trends would you like to explore?"
                ]
            ],
            'emi_calculator': [
                r'emi|loan|calculate emi|monthly payment|installment|mortgage|home loan',
                [
                    "I can calculate EMI for you! Please provide loan amount, interest rate, and tenure. Example: 'EMI for ₹50L at 8.5% for 20 years'",
                    "Let me help with EMI calculation. What's the loan amount, interest rate, and tenure?",
                    "EMI calculations made easy! Just give me the loan details and I'll do the math."
                ]
            ],
            'risk_assessment': [
                r'risk|risky|safe|safety|secure|investment risk|market risk',
                [
                    "I can assess investment risks for you. Which locality or investment are you concerned about?",
                    "Risk analysis is important! Tell me about your investment and I'll evaluate the risks.",
                    "Let me help you understand the risks. What specific investment are you considering?"
                ]
            ],
            'rental_yield': [
                r'rental yield|rent|rental income|tenant|renting|lease',
                [
                    "I can help you understand rental yields! Which area are you looking at for rental income?",
                    "Rental yield analysis coming up! Tell me the locality and I'll show you the potential.",
                    "Rental income is a great strategy! Which area interests you for rental properties?"
                ]
            ],
            'help': [
                r'help|what can you do|features|capabilities|commands|options|menu',
                ["""🤖 **PropTech AI Assistant - Complete Feature Guide**

**🏠 CORE REAL ESTATE FUNCTIONS:**
• **ROI Calculator** - "Calculate ROI for ₹50L in Andheri"
• **Investment Advisor** - "Investment advice for ₹1 crore budget"
• **Market Comparison** - "Compare Bandra vs Powai"
• **Locality Analysis** - "Tell me about Thane real estate"
• **Price Predictions** - "Future prices in Goregaon"

**💰 FINANCIAL CALCULATIONS:**
• **EMI Calculator** - "EMI for ₹80L at 8.5% for 20 years"
• **Affordability Check** - "What can I afford with ₹20L down payment"
• **Rental Yield** - "Rental income potential in Malad"
• **Risk Assessment** - "Investment risks in Lower Parel"

**📊 MARKET INTELLIGENCE:**
• **Heat Map Insights** - "Explain ROI hotspots in Mumbai"
• **Trend Analysis** - "Mumbai real estate trends 2024"
• **Growth Areas** - "Which areas are growing fastest"
• **Infrastructure Impact** - "Metro effect on property prices"

**🎯 PERSONALIZED ADVICE:**
• **Portfolio Planning** - "Diversify ₹2 crore across Mumbai"
• **First-time Buyer** - "Best areas for first property"
• **Investor Strategy** - "Rental vs appreciation focus"
• **Exit Strategy** - "When to sell property in Bandra"

**💡 SMART FEATURES:**
• **Quick Calculations** - Instant ROI, EMI, affordability
• **Area Recommendations** - Based on your budget & goals
• **Market Alerts** - Hot deals and opportunities
• **Investment Timeline** - When to buy/sell/hold

Just ask me anything about Mumbai real estate! I'm here 24/7 to help! 🚀"""]
            ],
            'thanks': [
                r'thank you|thanks|appreciate|helpful|great|awesome|excellent',
                [
                    "You're welcome! I'm here whenever you need real estate advice. Feel free to ask more questions!",
                    "Happy to help! Is there anything else about Mumbai real estate you'd like to know?",
                    "Glad I could assist! Don't hesitate to reach out for more investment insights.",
                    "My pleasure! I'm always here to help with your real estate journey."
                ]
            ],
            'goodbye': [
                r'bye|goodbye|see you|exit|quit|end',
                [
                    "Goodbye! Best of luck with your real estate investments. Come back anytime for more advice!",
                    "See you later! Remember, I'm here 24/7 for all your Mumbai real estate questions.",
                    "Take care! May your investments bring great returns. Feel free to chat again soon!",
                    "Bye! Happy investing, and remember - location, location, location! 🏠"
                ]
            ]
        }
    
    def load_knowledge_base(self):
        """Load real estate knowledge base"""
        self.knowledge_base = {
            'mumbai_areas': {
                'premium': ['bandra', 'juhu', 'worli', 'lower parel', 'marine drive', 'colaba'],
                'emerging': ['thane', 'kalyan', 'vasai', 'virar', 'dombivli', 'badlapur'],
                'established': ['andheri', 'goregaon', 'malad', 'kandivali', 'borivali', 'powai'],
                'commercial': ['bkc', 'nariman point', 'fort', 'churchgate', 'andheri east']
            },
            'investment_tips': {
                'first_time': "Start with established suburbs, focus on connectivity, check legal documents",
                'experienced': "Diversify across micro-markets, consider commercial properties, time the market",
                'rental_focus': "Choose areas with IT hubs, good transport, young demographics",
                'appreciation_focus': "Look for infrastructure development, upcoming projects, land scarcity"
            },
            'market_insights': {
                'growth_drivers': ['metro expansion', 'it sector growth', 'infrastructure development', 'government policies'],
                'risk_factors': ['regulatory changes', 'interest rate fluctuations', 'oversupply', 'economic slowdown']
            }
        }
    
    def process_message(self, user_message):
        """Process user message and generate response"""
        try:
            original_message = user_message
            user_message = user_message.lower().strip()
            
            # Store conversation
            self.conversation_history.append({
                "user": original_message, 
                "timestamp": datetime.now(),
                "processed": user_message
            })
            
            # Extract entities
            entities = self.extract_entities(user_message)
            
            # Update user context
            self.update_user_context(user_message, entities)
            
            # Generate response
            response = self.generate_response(user_message, entities)
            
            # Store bot response
            self.conversation_history.append({
                "bot": response, 
                "timestamp": datetime.now(),
                "entities": entities
            })
            
            return response
            
        except Exception as e:
            return "Sorry, I encountered an error processing your message. Please try rephrasing your question or ask for help to see what I can do!"
    
    def extract_entities(self, message):
        """Extract entities like localities, prices, percentages from message"""
        entities = {
            'localities': [],
            'prices': [],
            'percentages': [],
            'numbers': [],
            'intent': None,
            'budget_range': None,
            'timeframe': None
        }
        
        try:
            # Extract localities
            localities = self.ml_service.data.get('localities', [])
            for locality in localities:
                if locality.lower() in message:
                    entities['localities'].append(locality)
            
            # Extract prices with various formats
            price_patterns = [
                r'₹(\d+(?:\.\d+)?)\s*cr|₹(\d+(?:\.\d+)?)\s*crore',  # Crores
                r'₹(\d+(?:\.\d+)?)\s*l|₹(\d+(?:\.\d+)?)\s*lakh',    # Lakhs
                r'(\d+(?:\.\d+)?)\s*crore',                          # Crores without ₹
                r'(\d+(?:\.\d+)?)\s*lakh',                           # Lakhs without ₹
                r'₹(\d+(?:\.\d+)?)\s*k|₹(\d+(?:\.\d+)?)\s*thousand' # Thousands
            ]
            
            for pattern in price_patterns:
                matches = re.findall(pattern, message, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        for m in match:
                            if m:
                                value = float(m)
                                if 'crore' in pattern:
                                    entities['prices'].append(value * 100)  # Convert to lakhs
                                elif 'thousand' in pattern or 'k' in pattern:
                                    entities['prices'].append(value / 100)  # Convert to lakhs
                                else:
                                    entities['prices'].append(value)
                    else:
                        entities['prices'].append(float(match))
            
            # Extract percentages
            percentage_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', message)
            entities['percentages'] = [float(p) for p in percentage_matches]
            
            # Extract general numbers
            number_matches = re.findall(r'\b(\d+(?:\.\d+)?)\b', message)
            entities['numbers'] = [float(n) for n in number_matches]
            
            # Determine budget range
            if entities['prices']:
                max_price = max(entities['prices'])
                if max_price >= 100:
                    entities['budget_range'] = 'high'  # 1 crore+
                elif max_price >= 50:
                    entities['budget_range'] = 'medium'  # 50L - 1Cr
                else:
                    entities['budget_range'] = 'low'  # Under 50L
            
            # Extract timeframe
            timeframe_patterns = [
                r'(\d+)\s*year', r'(\d+)\s*month', r'short term', r'long term',
                r'immediate', r'urgent', r'soon', r'future'
            ]
            for pattern in timeframe_patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    entities['timeframe'] = re.search(pattern, message, re.IGNORECASE).group()
                    break
            
        except Exception as e:
            pass
        
        return entities
    
    def update_user_context(self, message, entities):
        """Update user context for personalized responses"""
        if entities['budget_range']:
            self.user_context['budget_range'] = entities['budget_range']
        
        if entities['localities']:
            self.user_context['interested_areas'] = entities['localities']
        
        if 'first time' in message or 'beginner' in message:
            self.user_context['experience_level'] = 'beginner'
        elif 'experienced' in message or 'investor' in message:
            self.user_context['experience_level'] = 'experienced'
        
        if 'rental' in message or 'rent' in message:
            self.user_context['investment_goal'] = 'rental_income'
        elif 'appreciation' in message or 'growth' in message:
            self.user_context['investment_goal'] = 'capital_appreciation'
    
    def generate_response(self, message, entities):
        """Generate appropriate response based on message and entities"""
        
        # Check for specific queries first (priority order)
        if self.is_investment_advice(message, entities):
            return self.handle_investment_advice(message, entities)
        
        elif self.is_roi_calculation(message, entities):
            return self.handle_roi_calculation(message, entities)
        
        elif self.is_locality_query(message, entities):
            return self.handle_locality_query(message, entities)
        
        elif self.is_comparison_query(message, entities):
            return self.handle_comparison_query(message, entities)
        
        elif self.is_emi_calculation(message, entities):
            return self.handle_emi_calculation(message, entities)
        
        elif self.is_risk_assessment(message, entities):
            return self.handle_risk_assessment(message, entities)
        
        elif self.is_rental_yield_query(message, entities):
            return self.handle_rental_yield(message, entities)
        
        elif self.is_market_trend_query(message):
            return self.handle_market_trends(message)
        
        elif self.is_affordability_query(message, entities):
            return self.handle_affordability(message, entities)
        
        # Pattern matching for general responses
        for intent, (pattern, responses) in self.patterns.items():
            if re.search(pattern, message, re.IGNORECASE):
                return random.choice(responses)
        
        # Context-aware default response
        return self.get_contextual_response(message, entities)
    
    def is_investment_advice(self, message, entities):
        """Check if user wants investment advice"""
        advice_keywords = ['investment advice', 'recommend', 'suggest', 'budget', 'crore', 'where to invest', 'portfolio', 'strategy']
        return any(keyword in message for keyword in advice_keywords)
    
    def handle_investment_advice(self, message, entities):
        """Handle investment advice requests with comprehensive analysis"""
        try:
            # Determine budget
            budget_lakhs = None
            if entities['prices']:
                budget_lakhs = max(entities['prices'])
            elif 'crore' in message:
                crore_matches = re.findall(r'(\d+(?:\.\d+)?)\s*crore', message)
                if crore_matches:
                    budget_lakhs = float(crore_matches[0]) * 100
                else:
                    budget_lakhs = 100  # Default 1 crore
            
            if budget_lakhs:
                if budget_lakhs >= 200:  # 2+ crores
                    return self.get_premium_investment_advice(budget_lakhs)
                elif budget_lakhs >= 100:  # 1-2 crores
                    return self.get_high_budget_advice(budget_lakhs)
                elif budget_lakhs >= 50:  # 50L - 1 crore
                    return self.get_medium_budget_advice(budget_lakhs)
                else:  # Under 50L
                    return self.get_budget_friendly_advice(budget_lakhs)
            else:
                return self.get_general_investment_advice()
                
        except Exception as e:
            return "I'd love to help with investment advice! Please specify your budget (e.g., ₹50 lakhs, ₹1 crore) and I'll provide detailed recommendations."
    
    def get_premium_investment_advice(self, budget_lakhs):
        """Investment advice for 2+ crore budget"""
        return f"""💎 **Premium Investment Strategy for ₹{budget_lakhs/100:.1f} Crore Portfolio**

**🏢 DIVERSIFIED PREMIUM PORTFOLIO:**

**Tier 1: Premium Properties (40% - ₹{budget_lakhs*0.4/100:.1f} Cr)**
• **Bandra West/Juhu**: ₹80L-1.2Cr (Prime location, celebrity area)
• **Lower Parel/Worli**: ₹70L-1Cr (Business district, high appreciation)
• **Powai**: ₹60L-80L (IT hub, consistent demand)

**Tier 2: Growth Areas (35% - ₹{budget_lakhs*0.35/100:.1f} Cr)**
• **Thane/Ghodbunder Road**: ₹40L-60L (Infrastructure boom)
• **Goregaon/Malad**: ₹35L-50L (Metro connectivity)
• **Kandivali/Borivali**: ₹30L-45L (Established suburbs)

**Tier 3: Emerging Markets (20% - ₹{budget_lakhs*0.2/100:.1f} Cr)**
• **Kalyan/Dombivli**: ₹25L-35L (Affordable, high growth)
• **Vasai/Virar**: ₹20L-30L (Future potential)

**Cash Reserve (5% - ₹{budget_lakhs*0.05/100:.1f} Cr)**
• Market opportunities & emergencies

**📊 EXPECTED PORTFOLIO RETURNS:**
• **Rental Yield**: 7-12% annually
• **Capital Appreciation**: 10-15% annually
• **Total ROI**: 15-25% potential
• **Risk Level**: Balanced (Medium)

**🎯 STRATEGIC ADVANTAGES:**
✅ Geographic diversification across Mumbai
✅ Price point diversification (₹20L to ₹1.2Cr)
✅ Mix of established and emerging areas
✅ Balanced rental income and appreciation
✅ Liquidity options across segments

**📈 5-YEAR PROJECTION:**
• Portfolio Value: ₹{budget_lakhs*1.8/100:.1f} - ₹{budget_lakhs*2.2/100:.1f} Crores
• Annual Income: ₹{budget_lakhs*0.08:.0f}L - ₹{budget_lakhs*0.12:.0f}L
• Total Returns: 80-120% over 5 years

**🔗 Next Steps:**
• Detailed area analysis for each tier
• Property inspection and due diligence
• Financing strategy optimization
• Legal documentation review

Want specific analysis for any tier or area? Just ask!"""
    
    def get_high_budget_advice(self, budget_lakhs):
        """Investment advice for 1-2 crore budget"""
        return f"""💰 **Strategic Investment Plan for ₹{budget_lakhs/100:.1f} Crore Budget**

**🏠 BALANCED PORTFOLIO APPROACH:**

**Primary Investment (60% - ₹{budget_lakhs*0.6:.0f}L)**
• **Powai/Hiranandani**: ₹50-70L (IT professionals, stable demand)
• **Thane West**: ₹45-65L (Infrastructure development, metro coming)
• **Goregaon East**: ₹40-60L (Commercial hub, good connectivity)

**Secondary Investment (30% - ₹{budget_lakhs*0.3:.0f}L)**
• **Malad/Kandivali**: ₹25-40L (Established areas, metro connectivity)
• **Kalyan/Dombivli**: ₹20-35L (High growth potential, affordability)

**Opportunity Fund (10% - ₹{budget_lakhs*0.1:.0f}L)**
• Market corrections, distress sales, new launches

**📊 EXPECTED RETURNS:**
• **Rental Yield**: 8-12% annually
• **Capital Appreciation**: 8-12% annually  
• **Total ROI**: 14-20% potential
• **Payback Period**: 6-8 years

**🎯 RECOMMENDED STRATEGY:**
1. **Start with Tier 1** (Powai/Thane) - Lower risk, steady returns
2. **Add Tier 2** within 6 months - Higher growth potential
3. **Keep opportunity fund** liquid for 12-18 months
4. **Reinvest returns** for compounding effect

**🚇 INFRASTRUCTURE ADVANTAGE:**
• Metro Line 2B (Thane-Powai corridor)
• Coastal Road (Worli-Versova connectivity)
• Mumbai Trans Harbour Link (Thane connectivity)

**💡 PRO TIPS:**
• Buy during monsoon season (better deals)
• Focus on 2-3 BHK for rental demand
• Ensure parking and amenities
• Check builder reputation and delivery record

**🔗 Want detailed analysis for specific areas? Ask me about any locality!**"""
    
    def get_medium_budget_advice(self, budget_lakhs):
        """Investment advice for 50L-1 crore budget"""
        return f"""🏠 **Smart Investment Strategy for ₹{budget_lakhs:.0f} Lakh Budget**

**🎯 FOCUSED APPROACH:**

**Option 1: Single Premium Property**
• **Powai**: ₹{budget_lakhs*0.8:.0f}L (IT hub, 10-12% ROI)
• **Thane West**: ₹{budget_lakhs*0.85:.0f}L (Growth area, 12-14% ROI)
• **Goregaon**: ₹{budget_lakhs*0.9:.0f}L (Established, 8-10% ROI)

**Option 2: Dual Investment**
• **Primary**: ₹{budget_lakhs*0.6:.0f}L in Thane/Kalyan
• **Secondary**: ₹{budget_lakhs*0.3:.0f}L in emerging area
• **Reserve**: ₹{budget_lakhs*0.1:.0f}L for opportunities

**📊 EXPECTED PERFORMANCE:**
• **Monthly Rental**: ₹{budget_lakhs*0.008:.0f} - ₹{budget_lakhs*0.012:.0f}
• **Annual ROI**: 10-15%
• **5-Year Growth**: 60-80%

**🎯 TOP RECOMMENDATIONS:**

**1. THANE (Best Overall)**
• Price Range: ₹{budget_lakhs*0.7:.0f}L - ₹{budget_lakhs:.0f}L
• ROI: 12-15%
• Growth Drivers: Metro, IT parks, infrastructure

**2. KALYAN (High Growth)**
• Price Range: ₹{budget_lakhs*0.5:.0f}L - ₹{budget_lakhs*0.8:.0f}L  
• ROI: 14-18%
• Growth Drivers: Affordability, connectivity

**3. POWAI (Stable)**
• Price Range: ₹{budget_lakhs*0.8:.0f}L - ₹{budget_lakhs:.0f}L
• ROI: 8-12%
• Growth Drivers: IT sector, established market

**💰 FINANCING STRATEGY:**
• Down Payment: ₹{budget_lakhs*0.2:.0f}L (20%)
• Home Loan: ₹{budget_lakhs*0.8:.0f}L (80%)
• EMI: ₹{budget_lakhs*0.8*0.008:.0f} (approx)

**🔗 Ready for detailed area analysis? Ask about any specific locality!**"""
    
    def get_budget_friendly_advice(self, budget_lakhs):
        """Investment advice for under 50L budget"""
        return f"""🌟 **Smart Entry Strategy for ₹{budget_lakhs:.0f} Lakh Budget**

**🎯 HIGH-GROWTH OPPORTUNITIES:**

**Tier 1: Best Value (₹{budget_lakhs*0.8:.0f}L - ₹{budget_lakhs:.0f}L)**
• **Kalyan East**: 15-18% ROI, excellent connectivity
• **Dombivli**: 14-16% ROI, family-friendly area
• **Vasai East**: 16-20% ROI, emerging market

**Tier 2: Ultra Affordable (₹{budget_lakhs*0.6:.0f}L - ₹{budget_lakhs*0.8:.0f}L)**
• **Virar**: 18-22% ROI, highest growth potential
• **Badlapur**: 16-19% ROI, industrial growth
• **Ambernath**: 17-20% ROI, upcoming area

**📊 EXPECTED RETURNS:**
• **Monthly Rental**: ₹{budget_lakhs*0.012:.0f} - ₹{budget_lakhs*0.018:.0f}
• **Annual ROI**: 14-22%
• **3-Year Growth**: 50-70%

**🚀 GROWTH CATALYSTS:**
• Mumbai Metro expansion
• Industrial corridor development  
• IT park establishments
• Infrastructure projects

**💡 SMART STRATEGY:**
1. **Start with Kalyan/Dombivli** (proven track record)
2. **Focus on 1-2 BHK** (high rental demand)
3. **Near railway stations** (connectivity premium)
4. **Ready possession** (immediate rental income)

**🎯 INVESTMENT TIMELINE:**
• **Month 1-2**: Property search and finalization
• **Month 3**: Documentation and registration
• **Month 4**: Rental tenant acquisition
• **Year 1**: 14-18% returns expected
• **Year 3**: 50-70% capital appreciation

**🔗 Want specific property recommendations in any area? Just ask!**"""
    
    def get_general_investment_advice(self):
        """General investment advice when budget is not specified"""
        return """💡 **Mumbai Real Estate Investment Guide**

**🎯 BUDGET-WISE RECOMMENDATIONS:**

**₹20-40 Lakhs (Entry Level)**
• **Areas**: Kalyan, Dombivli, Vasai, Virar
• **ROI**: 15-22% annually
• **Strategy**: High growth, emerging markets

**₹40-70 Lakhs (Mid Range)**  
• **Areas**: Thane, Malad, Kandivali, Goregaon
• **ROI**: 10-15% annually
• **Strategy**: Balanced growth and stability

**₹70L-1 Crore (Premium)**
• **Areas**: Powai, Andheri, Bandra (outskirts)
• **ROI**: 8-12% annually
• **Strategy**: Stable returns, prime locations

**₹1+ Crore (Luxury)**
• **Areas**: Bandra, Juhu, Lower Parel, Worli
• **ROI**: 6-10% annually
• **Strategy**: Capital appreciation, prestige

**📊 INVESTMENT PRINCIPLES:**

**🎯 Location Factors:**
• Metro/railway connectivity
• IT hubs and commercial areas
• Schools, hospitals, malls nearby
• Future infrastructure projects

**💰 Financial Planning:**
• 20-25% down payment
• EMI should be <40% of income
• Keep 6-month EMI as emergency fund
• Consider rental income vs EMI

**📈 Market Timing:**
• **Best to Buy**: Monsoon season, year-end
• **Avoid**: Festival seasons, new launch hype
• **Hold Period**: Minimum 3-5 years
• **Exit Strategy**: Plan before buying

**🔗 Ready to get specific advice? Tell me your budget and I'll create a personalized strategy!**"""
    
    def is_roi_calculation(self, message, entities):
        """Check if user wants ROI calculation"""
        roi_keywords = ['roi', 'return', 'calculate', 'profit', 'yield']
        return any(keyword in message for keyword in roi_keywords) and (entities['localities'] or entities['prices'])
    
    def handle_roi_calculation(self, message, entities):
        """Handle ROI calculation requests"""
        try:
            if entities['localities'] and entities['prices']:
                locality = entities['localities'][0]
                price = entities['prices'][0]
                
                # Use your existing ROI calculation
                roi = self.ml_service.calculate_roi_fallback(locality, price)
                
                if roi:
                    locality_stats = self.ml_service.get_locality_stats(locality)
                    
                    # Calculate additional metrics
                    monthly_rent = locality_stats['avg_rent']
                    annual_rent = monthly_rent * 12
                    property_value = price * 100000
                    
                    response = f"""🎯 **Comprehensive ROI Analysis for {locality.title()}**

**💰 INVESTMENT OVERVIEW:**
• **Property Price**: ₹{price} Lakhs
• **Expected Monthly Rent**: ₹{monthly_rent:,.0f}
• **Annual Rental Income**: ₹{annual_rent:,.0f}

**📈 ROI BREAKDOWN:**
• **Predicted ROI**: {roi:.2f}% annually
• **Market Average ROI**: {locality_stats['avg_roi']:.2f}%
• **Rental Yield**: {(annual_rent/property_value)*100:.2f}%

**📊 MARKET COMPARISON:**
• **Area Price Range**: ₹{locality_stats['price_range']['min']:.1f}L - ₹{locality_stats['price_range']['max']:.1f}L
• **ROI Range**: {locality_stats['roi_range']['min']:.1f}% - {locality_stats['roi_range']['max']:.1f}%
• **Average Rate/sqft**: ₹{locality_stats['avg_rate_sqft']:,.0f}

**💡 INVESTMENT INSIGHT:**
"""
                    if roi > locality_stats['avg_roi']:
                        response += f"✅ **Excellent Choice!** Your ROI of {roi:.2f}% is {roi - locality_stats['avg_roi']:.1f}% above the market average.\n"
                        response += "🚀 This property offers above-average returns for the area."
                    elif roi >= locality_stats['avg_roi'] * 0.9:
                        response += f"✅ **Good Investment!** Your ROI of {roi:.2f}% is close to market average.\n"
                        response += "📈 Solid investment with market-aligned returns."
                    else:
                        response += f"⚠️ **Consider Carefully** - Your ROI of {roi:.2f}% is {locality_stats['avg_roi'] - roi:.1f}% below market average.\n"
                        response += "💭 You might want to negotiate the price or explore other options."
                    
                    # Add investment recommendations
                    response += f"""

**🎯 INVESTMENT RECOMMENDATIONS:**
• **Break-even Period**: {property_value / annual_rent:.1f} years
• **Monthly Cash Flow**: ₹{monthly_rent - (price * 1000):.0f} (after ₹{price * 1000:.0f} EMI estimate)
• **5-Year Projection**: ₹{property_value * 1.5 / 100000:.1f}L - ₹{property_value * 1.8 / 100000:.1f}L

**📋 NEXT STEPS:**
• Verify actual rental rates in the area
• Check property condition and legal documents
• Compare with 2-3 similar properties
• Consider financing options and EMI impact

🔗 Want detailed investment analysis? Use our [Investment Calculator](/investment-calculator)
🗺️ See area performance: [ROI Heatmap](/roi-heatmap)"""
                    
                    return response
                else:
                    return f"❌ Sorry, I don't have enough data for {locality.title()}. Try popular areas like Andheri, Bandra, Thane, Powai, or Goregaon."
            
            elif entities['localities']:
                locality = entities['localities'][0]
                stats = self.ml_service.get_locality_stats(locality)
                if stats:
                    return f"""📍 **ROI Information for {locality.title()}**
                    
**📊 Market Overview:**
• **Average ROI**: {stats['avg_roi']:.2f}% annually
• **ROI Range**: {stats['roi_range']['min']:.1f}% - {stats['roi_range']['max']:.1f}%
• **Average Price**: ₹{stats['avg_price']:.1f} Lakhs
• **Average Rent**: ₹{stats['avg_rent']:,.0f}/month

To calculate ROI for a specific property, tell me your budget!
Example: "Calculate ROI for ₹{stats['avg_price']:.0f}L property in {locality.title()}" """
                else:
                    return f"I can calculate ROI for {locality.title()}! What's your property budget? (e.g., ₹50 lakhs)"
            
            elif entities['prices']:
                price = entities['prices'][0]
                return f"I can calculate ROI for ₹{price} lakhs! Which locality are you considering? Popular options: Andheri, Bandra, Thane, Powai, Goregaon."
            
            else:
                return """🧮 **ROI Calculator Ready!**
                
I can calculate detailed ROI analysis including:
• Annual return percentage
• Monthly rental income potential  
• Break-even period
• Market comparison
• Investment recommendations

**Just tell me:**
• Locality name (e.g., Andheri, Thane, Powai)
• Property price (e.g., ₹50 lakhs, ₹1 crore)

**Example**: "Calculate ROI for ₹60L property in Thane" """
        
        except Exception as e:
            return "Sorry, I encountered an error calculating ROI. Please try again with format: 'Calculate ROI for ₹[amount] in [locality]'"
    
    def is_locality_query(self, message, entities):
        """Check if user wants locality information"""
        return entities['localities'] or any(word in message for word in ['tell me about', 'information about', 'locality', 'area', 'neighborhood'])
    
    def handle_locality_query(self, message, entities):
        """Handle locality information requests"""
        if entities['localities']:
            locality = entities['localities'][0]
            stats = self.ml_service.get_locality_stats(locality)
            
            if stats:
                # Determine locality category
                category = self.get_locality_category(locality.lower())
                
                return f"""📍 **Complete Guide to {locality.title()}**

**🏠 MARKET OVERVIEW:**
• **Category**: {category}
• **Average Property Price**: ₹{stats['avg_price']:.1f} Lakhs
• **Price Range**: ₹{stats['price_range']['min']:.1f}L - ₹{stats['price_range']['max']:.1f}L
• **Average Rent**: ₹{stats['avg_rent']:,.0f}/month
• **Rate per sqft**: ₹{stats['avg_rate_sqft']:,.0f}

**📈 INVESTMENT METRICS:**
• **Average ROI**: {stats['avg_roi']:.2f}% annually
• **ROI Range**: {stats['roi_range']['min']:.1f}% - {stats['roi_range']['max']:.1f}%
• **Investment Grade**: {self.get_investment_grade(stats['avg_roi'])}

**🎯 AREA HIGHLIGHTS:**
{self.get_locality_highlights(locality.lower())}

**💡 INVESTMENT INSIGHT:**
{self.get_detailed_locality_insight(stats, locality.lower())}

**🔗 QUICK ACTIONS:**
• Calculate ROI: "ROI for ₹{stats['avg_price']:.0f}L in {locality.title()}"
• Compare areas: "Compare {locality.title()} vs [other area]"
• Investment advice: "Investment advice for {locality.title()}"

Want more specific information about {locality.title()}? Just ask!"""
            else:
                return f"""📍 **{locality.title()} Information**
                
I don't have detailed market data for {locality.title()} yet, but I can help you with:

**🏠 Popular Mumbai Areas I Cover:**
• **Premium**: Bandra, Juhu, Worli, Lower Parel
• **Established**: Andheri, Powai, Goregaon, Malad  
• **Emerging**: Thane, Kalyan, Vasai, Virar
• **Commercial**: BKC, Nariman Point, Fort

**💡 Alternative Suggestions:**
Try asking about nearby areas like Andheri, Thane, or Powai for detailed analysis.

**🔗 Or explore:**
• [ROI Heatmap](/roi-heatmap) for visual area comparison
• [Market Comparison](/market-comparison) for side-by-side analysis"""
        else:
            return """🏠 **Mumbai Locality Guide**
            
Which area would you like to know about? I have comprehensive data on:

**🌟 PREMIUM AREAS:**
• Bandra, Juhu, Worli, Lower Parel, Marine Drive

**🏘️ ESTABLISHED SUBURBS:**  
• Andheri, Powai, Goregaon, Malad, Kandivali, Borivali

**🚀 EMERGING HOTSPOTS:**
• Thane, Kalyan, Dombivli, Vasai, Virar

**🏢 COMMERCIAL HUBS:**
• BKC, Nariman Point, Fort, Andheri East

**Just ask**: "Tell me about [area name]" or "Information about Thane"

I'll provide detailed market analysis, investment insights, and recommendations!"""
    
    def get_locality_category(self, locality):
        """Determine locality category"""
        if locality in self.knowledge_base['mumbai_areas']['premium']:
            return "Premium/Luxury"
        elif locality in self.knowledge_base['mumbai_areas']['emerging']:
            return "Emerging/High Growth"
        elif locality in self.knowledge_base['mumbai_areas']['established']:
            return "Established Suburb"
        elif locality in self.knowledge_base['mumbai_areas']['commercial']:
            return "Commercial Hub"
        else:
            return "Residential Area"
    
    def get_investment_grade(self, roi):
        """Get investment grade based on ROI"""
        if roi >= 12:
            return "A+ (Excellent)"
        elif roi >= 10:
            return "A (Very Good)"
        elif roi >= 8:
            return "B+ (Good)"
        elif roi >= 6:
            return "B (Average)"
        else:
            return "C (Below Average)"
    
    def get_locality_highlights(self, locality):
        """Get specific highlights for localities"""
        highlights = {
            'andheri': "• Bollywood hub with film studios\n• Excellent metro and airport connectivity\n• Mix of commercial and residential\n• Good rental demand from professionals",
            'bandra': "• Celebrity and upscale residential area\n• Bandra-Kurla Complex nearby\n• Premium shopping and dining\n• High capital appreciation potential",
            'thane': "• Rapidly developing infrastructure\n• Upcoming metro connectivity\n• IT parks and commercial growth\n• More affordable than central Mumbai",
            'powai': "• Major IT hub with Hiranandani Gardens\n• Premium residential complexes\n• Good schools and healthcare\n• Consistent rental demand",
            'goregaon': "• Film City and entertainment industry\n• Metro connectivity (Line 1)\n• Balanced residential and commercial\n• Good connectivity to western suburbs",
            'malad': "• Growing commercial and IT sector\n• Metro Line 7 connectivity\n• Mix of affordable and premium housing\n• Good infrastructure development"
        }
        return highlights.get(locality, "• Developing residential area\n• Good connectivity options\n• Growing infrastructure\n• Investment potential")
    
    def get_detailed_locality_insight(self, stats, locality):
        """Get detailed investment insight for locality"""
        roi = stats['avg_roi']
        price = stats['avg_price']
        
        if roi >= 10 and price <= 60:
            return "🔥 **High ROI + Affordable**: Excellent for rental income investors and first-time buyers."
        elif roi >= 8 and price >= 80:
            return "💎 **Balanced Premium**: Good for long-term capital appreciation with decent rental yields."
        elif roi >= 12:
            return "🚀 **ROI Superstar**: Outstanding returns, perfect for income-focused investors."
        elif price >= 100:
            return "🏆 **Premium Market**: Focus on capital appreciation, prestige location."
        else:
            return "📈 **Steady Growth**: Reliable investment with balanced risk-reward profile."
    
    def is_comparison_query(self, message, entities):
        """Check if user wants to compare localities"""
        return len(entities['localities']) >= 2 or any(word in message for word in ['compare', 'vs', 'versus', 'difference', 'better'])
    
    def handle_comparison_query(self, message, entities):
        """Handle locality comparison requests"""
        if len(entities['localities']) >= 2:
            loc1, loc2 = entities['localities'][0], entities['localities'][1]
            comparison = self.ml_service.compare_localities(loc1, loc2)
            
            if comparison:
                return f"""⚖️ **Detailed Comparison: {loc1.title()} vs {loc2.title()}**

**📊 HEAD-TO-HEAD ANALYSIS:**

**{loc1.title()}:**
🏠 **Price**: ₹{comparison['loc1']['stats']['avg_price']:.1f}L | 📈 **ROI**: {comparison['loc1']['stats']['avg_roi']:.1f}% | 🏡 **Rent**: ₹{comparison['loc1']['stats']['avg_rent']:,.0f}

**{loc2.title()}:**  
🏠 **Price**: ₹{comparison['loc2']['stats']['avg_price']:.1f}L | 📈 **ROI**: {comparison['loc2']['stats']['avg_roi']:.1f}% | 🏡 **Rent**: ₹{comparison['loc2']['stats']['avg_rent']:,.0f}

**🏆 WINNER ANALYSIS:**
• **💰 More Affordable**: {comparison['comparison']['better_price']} (₹{abs(comparison['comparison']['price_difference']):.1f}L difference)
• **📈 Better ROI**: {comparison['comparison']['better_roi']} ({abs(comparison['comparison']['roi_difference']):.1f}% higher)
• **🏡 Higher Rent**: {comparison['comparison']['better_rent']} (₹{abs(comparison['comparison']['rent_difference']):,.0f} more)

**🎯 INVESTMENT RECOMMENDATION:**

**For Rental Income Focus:**
Choose **{comparison['comparison']['better_roi']}** - Higher ROI means better rental returns

**For Affordability:**
Choose **{comparison['comparison']['better_price']}** - Lower entry cost, easier financing

**For Premium Investment:**
Choose the area with higher absolute rent potential

**📋 DETAILED COMPARISON:**
• **Price Difference**: ₹{abs(comparison['comparison']['price_difference']):.1f} Lakhs
• **ROI Difference**: {abs(comparison['comparison']['roi_difference']):.1f}% annually  
• **Rent Difference**: ₹{abs(comparison['comparison']['rent_difference']):,.0f} monthly

🔗 [Visual Comparison](/market-comparison) | [Detailed Analysis](/investment-calculator)"""
            else:
                return f"Sorry, I couldn't compare {loc1.title()} and {loc2.title()}. Please ensure both are valid Mumbai localities I have data for."
        else:
            return """⚖️ **Area Comparison Tool**
            
I can compare any two Mumbai localities for you! Just specify the areas.

**Examples:**
• "Compare Andheri and Bandra"
• "Thane vs Powai comparison"  
• "Which is better - Malad or Goregaon?"

**I'll analyze:**
• Property prices and affordability
• ROI and rental yields
• Market trends and growth potential
• Investment recommendations
• Risk factors

**Popular Comparisons:**
• Andheri vs Bandra (Established vs Premium)
• Thane vs Kalyan (Emerging markets)
• Powai vs Goregaon (IT hubs)
• Malad vs Kandivali (Suburbs)

Which areas would you like me to compare?"""
    
    def is_emi_calculation(self, message, entities):
        """Check if user wants EMI calculation"""
        return any(word in message for word in ['emi', 'loan', 'mortgage', 'monthly payment', 'installment'])
    
    def handle_emi_calculation(self, message, entities):
        """Handle EMI calculation requests"""
        try:
            numbers = entities['numbers']
            percentages = entities['percentages']
            
            # Extract loan details
            loan_amount = None
            tenure = 20  # Default
            interest_rate = 8.5  # Default
            
            if entities['prices']:
                loan_amount = entities['prices'][0]
            elif numbers:
                loan_amount = numbers[0]
            
            if len(numbers) >= 2:
                tenure = numbers[1]
            
            if percentages:
                interest_rate = percentages[0]
            
            if loan_amount:
                # Calculate EMI
                principal = loan_amount * 100000  # Convert to actual amount
                monthly_rate = interest_rate / (12 * 100)
                total_months = tenure * 12
                
                if monthly_rate > 0:
                    emi = (principal * monthly_rate * (1 + monthly_rate) ** total_months) / \
                          ((1 + monthly_rate) ** total_months - 1)
                else:
                    emi = principal / total_months
                
                total_payment = emi * total_months
                total_interest = total_payment - principal
                
                return f"""💳 **Comprehensive EMI Analysis**

**🏠 LOAN DETAILS:**
• **Loan Amount**: ₹{loan_amount} Lakhs
• **Interest Rate**: {interest_rate}% per annum
• **Tenure**: {tenure} years ({total_months} months)

**💰 EMI BREAKDOWN:**
• **Monthly EMI**: ₹{emi:,.0f}
• **Total Amount Payable**: ₹{total_payment/100000:.1f} Lakhs
• **Total Interest**: ₹{total_interest/100000:.1f} Lakhs
• **Interest %**: {(total_interest/principal)*100:.1f}% of loan amount

**📊 AFFORDABILITY ANALYSIS:**
• **Required Monthly Income**: ₹{emi*2.5:,.0f} (40% EMI rule)
• **Annual Income Needed**: ₹{emi*30:,.0f}
• **Down Payment (20%)**: ₹{loan_amount*0.25:.1f} Lakhs

**💡 SMART TIPS:**
• EMI should be <40% of monthly income
• Consider floating vs fixed rates
• Factor in processing fees (0.5-1% of loan)
• Keep 6-month EMI as emergency fund

**🎯 OPTIMIZATION OPTIONS:**
• **Longer Tenure**: ₹{(principal * (interest_rate/1200) * (1 + interest_rate/1200)**(25*12)) / ((1 + interest_rate/1200)**(25*12) - 1):,.0f} EMI for 25 years
• **Higher Down Payment**: Reduce loan amount to lower EMI
• **Prepayment**: Save ₹{total_interest*0.3/100000:.1f}L+ with annual prepayments

🔗 Need investment analysis? Try our [Investment Calculator](/investment-calculator)"""
            else:
                return """💳 **EMI Calculator Ready!**
                
I can calculate detailed EMI analysis for your home loan.

**Just provide:**
• **Loan Amount** (e.g., ₹50 lakhs, ₹80 lakhs)
• **Interest Rate** (e.g., 8.5%, 9.2%) - Optional, I'll use current rates
• **Tenure** (e.g., 20 years, 25 years) - Optional, default is 20 years

**Examples:**
• "EMI for ₹60L at 8.5% for 20 years"
• "Calculate EMI for ₹80 lakh loan"
• "Monthly payment for ₹1 crore home loan"

**I'll show you:**
• Monthly EMI amount
• Total interest payable
• Affordability requirements
• Optimization suggestions

What loan amount are you considering?"""
        
        except Exception as e:
            return "I can calculate EMI for you! Please provide loan amount and optionally interest rate and tenure. Example: 'EMI for ₹50L at 8.5% for 20 years'"
    
    def is_risk_assessment(self, message, entities):
        """Check if user wants risk assessment"""
        return any(word in message for word in ['risk', 'risky', 'safe', 'safety', 'secure'])
    
    def handle_risk_assessment(self, message, entities):
        """Handle risk assessment requests"""
        if entities['localities']:
            locality = entities['localities'][0]
            stats = self.ml_service.get_locality_stats(locality)
            
            if stats:
                risk_score = self.calculate_risk_score(stats, locality.lower())
                risk_level = self.get_risk_level(risk_score)
                
                return f"""🛡️ **Risk Assessment for {locality.title()}**

**📊 OVERALL RISK SCORE: {risk_score}/100**
**🎯 Risk Level: {risk_level}**

**🔍 RISK BREAKDOWN:**

**Market Risk (30%):**
• ROI Volatility: {self.get_roi_volatility(stats)}
• Price Stability: {self.get_price_stability(stats)}
• Market Maturity: {self.get_market_maturity(locality.lower())}

**Location Risk (25%):**
• Infrastructure: {self.get_infrastructure_risk(locality.lower())}
• Connectivity: {self.get_connectivity_risk(locality.lower())}
• Development Stage: {self.get_development_risk(locality.lower())}

**Financial Risk (25%):**
• Liquidity: {self.get_liquidity_risk(stats)}
• Entry Cost: {self.get_entry_cost_risk(stats)}
• Rental Demand: {self.get_rental_demand_risk(locality.lower())}

**External Risk (20%):**
• Regulatory: {self.get_regulatory_risk(locality.lower())}
• Economic Factors: {self.get_economic_risk()}
• Competition: {self.get_competition_risk(locality.lower())}

**💡 RISK MITIGATION STRATEGIES:**
{self.get_risk_mitigation_strategies(risk_level, locality.lower())}

**🎯 INVESTMENT RECOMMENDATION:**
{self.get_risk_based_recommendation(risk_level, stats)}

Want detailed analysis of any specific risk factor? Just ask!"""
            else:
                return f"I can assess investment risks for {locality.title()}! However, I need more market data for detailed analysis. Try popular areas like Andheri, Thane, or Powai for comprehensive risk assessment."
        else:
            return """🛡️ **Investment Risk Assessment**
            
I can analyze investment risks for any Mumbai locality!

**Risk Factors I Evaluate:**
• **Market Risk**: Price volatility, ROI consistency
• **Location Risk**: Infrastructure, connectivity, development
• **Financial Risk**: Liquidity, entry cost, rental demand  
• **External Risk**: Regulations, economic factors

**Just ask:**
• "Risk assessment for Thane"
• "Is Powai a safe investment?"
• "Investment risks in Bandra"

**I'll provide:**
• Overall risk score (0-100)
• Detailed risk breakdown
• Mitigation strategies
• Investment recommendations

Which area would you like me to assess?"""
    
    def calculate_risk_score(self, stats, locality):
        """Calculate risk score for a locality"""
        base_score = 50
        
        # ROI consistency (lower volatility = lower risk)
        roi_range = stats['roi_range']['max'] - stats['roi_range']['min']
        if roi_range <= 3:
            base_score -= 10
        elif roi_range >= 8:
            base_score += 15
        
        # Price level (very high prices = higher risk)
        if stats['avg_price'] >= 100:
            base_score += 20
        elif stats['avg_price'] <= 40:
            base_score -= 10
        
        # Area maturity
        if locality in self.knowledge_base['mumbai_areas']['premium']:
            base_score -= 5  # Lower risk
        elif locality in self.knowledge_base['mumbai_areas']['emerging']:
            base_score += 15  # Higher risk, higher reward
        
        return max(10, min(90, base_score))
    
    def get_risk_level(self, score):
        """Get risk level based on score"""
        if score <= 30:
            return "🟢 LOW RISK"
        elif score <= 50:
            return "🟡 MODERATE RISK"
        elif score <= 70:
            return "🟠 HIGH RISK"
        else:
            return "🔴 VERY HIGH RISK"
    
    def get_roi_volatility(self, stats):
        """Assess ROI volatility"""
        roi_range = stats['roi_range']['max'] - stats['roi_range']['min']
        if roi_range <= 3:
            return "Low (Stable returns)"
        elif roi_range <= 6:
            return "Moderate (Some fluctuation)"
        else:
            return "High (Volatile returns)"
    
    def get_price_stability(self, stats):
        """Assess price stability"""
        price_range = stats['price_range']['max'] - stats['price_range']['min']
        avg_price = stats['avg_price']
        volatility = (price_range / avg_price) * 100
        
        if volatility <= 30:
            return "Stable (Low price variation)"
        elif volatility <= 60:
            return "Moderate (Some price variation)"
        else:
            return "Volatile (High price variation)"
    
    def get_market_maturity(self, locality):
        """Assess market maturity"""
        if locality in self.knowledge_base['mumbai_areas']['premium']:
            return "Mature (Established market)"
        elif locality in self.knowledge_base['mumbai_areas']['established']:
            return "Developing (Growing market)"
        else:
            return "Emerging (New market)"
    
    def get_infrastructure_risk(self, locality):
        """Assess infrastructure risk"""
        if locality in ['andheri', 'bandra', 'powai']:
            return "Low (Excellent infrastructure)"
        elif locality in ['thane', 'goregaon', 'malad']:
            return "Moderate (Good infrastructure)"
        else:
            return "High (Developing infrastructure)"
    
    def get_connectivity_risk(self, locality):
        """Assess connectivity risk"""
        if locality in ['andheri', 'bandra', 'thane']:
            return "Low (Multiple transport options)"
        elif locality in ['powai', 'goregaon', 'malad']:
            return "Moderate (Good connectivity)"
        else:
            return "High (Limited connectivity)"
    
    def get_development_risk(self, locality):
        """Assess development stage risk"""
        if locality in self.knowledge_base['mumbai_areas']['emerging']:
            return "High (Early development stage)"
        elif locality in self.knowledge_base['mumbai_areas']['established']:
            return "Moderate (Steady development)"
        else:
            return "Low (Fully developed)"
    
    def get_liquidity_risk(self, stats):
        """Assess liquidity risk"""
        if stats['avg_price'] >= 100:
            return "High (Premium properties, limited buyers)"
        elif stats['avg_price'] <= 40:
            return "Low (Affordable, high demand)"
        else:
            return "Moderate (Balanced market)"
    
    def get_entry_cost_risk(self, stats):
        """Assess entry cost risk"""
        if stats['avg_price'] >= 80:
            return "High (High capital requirement)"
        elif stats['avg_price'] <= 50:
            return "Low (Affordable entry)"
        else:
            return "Moderate (Reasonable entry cost)"
    
    def get_rental_demand_risk(self, locality):
        """Assess rental demand risk"""
        if locality in ['powai', 'andheri', 'thane']:
            return "Low (High rental demand)"
        elif locality in ['goregaon', 'malad', 'kandivali']:
            return "Moderate (Steady rental demand)"
        else:
            return "High (Limited rental market)"
    
    def get_regulatory_risk(self, locality):
        """Assess regulatory risk"""
        return "Moderate (Standard Mumbai regulations)"
    
    def get_economic_risk(self):
        """Assess economic risk"""
        return "Moderate (General market conditions)"
    
    def get_competition_risk(self, locality):
        """Assess competition risk"""
        if locality in self.knowledge_base['mumbai_areas']['emerging']:
            return "High (Many new projects)"
        else:
            return "Moderate (Established competition)"
    
    def get_risk_mitigation_strategies(self, risk_level, locality):
        """Get risk mitigation strategies"""
        if "LOW" in risk_level:
            return """• Maintain property well for consistent returns
• Consider leveraging for portfolio expansion
• Focus on long-term holding strategy"""
        elif "MODERATE" in risk_level:
            return """• Diversify across 2-3 properties if possible
• Keep 6-month expense reserve
• Monitor market trends closely
• Consider professional property management"""
        else:
            return """• Limit exposure to 20-30% of portfolio
• Ensure strong rental agreements
• Keep higher cash reserves (12+ months)
• Consider exit strategy before buying
• Get comprehensive insurance coverage"""
    
    def get_risk_based_recommendation(self, risk_level, stats):
        """Get investment recommendation based on risk"""
        if "LOW" in risk_level:
            return f"✅ **RECOMMENDED** - Suitable for conservative investors seeking stable returns of {stats['avg_roi']:.1f}%"
        elif "MODERATE" in risk_level:
            return f"⚠️ **PROCEED WITH CAUTION** - Good for experienced investors comfortable with {stats['avg_roi']:.1f}% returns and moderate risk"
        else:
            return f"🚨 **HIGH RISK** - Only for aggressive investors seeking high returns. Consider smaller allocation despite {stats['avg_roi']:.1f}% potential"
    
    def is_rental_yield_query(self, message, entities):
        """Check if user wants rental yield information"""
        return any(word in message for word in ['rental yield', 'rent', 'rental income', 'tenant', 'renting'])
    
    def handle_rental_yield(self, message, entities):
        """Handle rental yield queries"""
        if entities['localities']:
            locality = entities['localities'][0]
            stats = self.ml_service.get_locality_stats(locality)
            
            if stats:
                annual_rent = stats['avg_rent'] * 12
                property_value = stats['avg_price'] * 100000
                rental_yield = (annual_rent / property_value) * 100
                
                return f"""🏠 **Rental Yield Analysis for {locality.title()}**

**💰 RENTAL METRICS:**
• **Average Monthly Rent**: ₹{stats['avg_rent']:,.0f}
• **Annual Rental Income**: ₹{annual_rent:,.0f}
• **Rental Yield**: {rental_yield:.2f}% annually
• **Gross ROI**: {stats['avg_roi']:.2f}% (including appreciation)

**📊 YIELD BREAKDOWN:**
• **Rental Component**: {rental_yield:.2f}%
• **Appreciation Component**: {stats['avg_roi'] - rental_yield:.2f}%
• **Total Return**: {stats['avg_roi']:.2f}%

**🎯 RENTAL MARKET INSIGHTS:**
• **Tenant Profile**: {self.get_tenant_profile(locality.lower())}
• **Rental Demand**: {self.get_rental_demand_level(locality.lower())}
• **Vacancy Risk**: {self.get_vacancy_risk(locality.lower())}
• **Rent Growth**: {self.get_rent_growth_potential(locality.lower())}

**💡 RENTAL STRATEGY:**
{self.get_rental_strategy(rental_yield, locality.lower())}

**📋 RENTAL INCOME PROJECTION:**
• **Year 1**: ₹{annual_rent:,.0f}
• **Year 3**: ₹{annual_rent * 1.15:,.0f} (5% annual growth)
• **Year 5**: ₹{annual_rent * 1.28:,.0f} (5% annual growth)

**🔗 Want detailed rental analysis? Ask about tenant management or rental agreements!"""
            else:
                return f"I can provide rental yield analysis for {locality.title()}! However, I need more market data. Try areas like Powai, Andheri, or Thane for detailed rental insights."
        else:
            return """🏠 **Rental Yield Analysis**
            
I can analyze rental income potential for any Mumbai locality!

**What I'll Show You:**
• Monthly and annual rental income
• Rental yield percentage
• Tenant demographics and demand
• Vacancy risks and market trends
• Rental growth projections

**Examples:**
• "Rental yield in Powai"
• "Rent potential in Thane"
• "Rental income for Andheri property"

**Popular High-Yield Areas:**
• **Powai**: 8-12% yield (IT professionals)
• **Thane**: 10-14% yield (growing demand)
• **Kalyan**: 12-16% yield (affordable segment)
• **Goregaon**: 8-10% yield (established market)

Which area interests you for rental investment?"""
    
    def get_tenant_profile(self, locality):
        """Get typical tenant profile for locality"""
        profiles = {
            'powai': "IT professionals, young couples, corporate executives",
            'andheri': "Film industry professionals, young professionals, families",
            'thane': "Middle-class families, IT professionals, government employees",
            'bandra': "High-income professionals, celebrities, expatriates",
            'goregaon': "Working professionals, small families, students",
            'malad': "Middle-income families, working professionals, students"
        }
        return profiles.get(locality, "Working professionals and families")
    
    def get_rental_demand_level(self, locality):
        """Get rental demand level for locality"""
        if locality in ['powai', 'andheri', 'thane']:
            return "High (Strong consistent demand)"
        elif locality in ['goregaon', 'malad', 'kandivali']:
            return "Moderate (Steady demand)"
        else:
            return "Growing (Increasing demand)"
    
    def get_vacancy_risk(self, locality):
        """Get vacancy risk assessment"""
        if locality in ['powai', 'andheri']:
            return "Low (2-5% typical vacancy)"
        elif locality in ['thane', 'goregaon']:
            return "Moderate (5-8% typical vacancy)"
        else:
            return "Variable (8-12% typical vacancy)"
    
    def get_rent_growth_potential(self, locality):
        """Get rent growth potential"""
        if locality in self.knowledge_base['mumbai_areas']['emerging']:
            return "High (8-12% annual growth potential)"
        elif locality in self.knowledge_base['mumbai_areas']['established']:
            return "Moderate (5-8% annual growth)"
        else:
            return "Stable (3-6% annual growth)"
    
    def get_rental_strategy(self, rental_yield, locality):
        """Get rental strategy recommendations"""
        if rental_yield >= 10:
            return f"""**INCOME-FOCUSED STRATEGY:**
• Excellent rental yield of {rental_yield:.1f}%
• Focus on tenant retention and property maintenance
• Consider leveraging for portfolio expansion
• Target long-term tenants for stability"""
        elif rental_yield >= 7:
            return f"""**BALANCED STRATEGY:**
• Good rental yield of {rental_yield:.1f}%
• Balance between rental income and appreciation
• Invest in property improvements for higher rents
• Monitor market for optimization opportunities"""
        else:
            return f"""**APPRECIATION-FOCUSED STRATEGY:**
• Moderate rental yield of {rental_yield:.1f}%
• Focus on capital appreciation over rental income
• Consider premium upgrades to justify higher rents
• Hold for long-term value growth"""
    
    def is_market_trend_query(self, message):
        """Check if user wants market trends"""
        trend_keywords = ['trends', 'market', 'growth', 'future', 'prediction', 'forecast', 'outlook']
        return any(keyword in message for keyword in trend_keywords)
    
    def handle_market_trends(self, message):
        """Handle market trend queries with comprehensive analysis"""
        try:
            # Get top performing areas from data
            top_areas = []
            if 'summary' in self.ml_service.data and not self.ml_service.data['summary'].empty:
                df = self.ml_service.data['summary']
                top_roi = df.nlargest(5, 'roi_mean')
                for _, row in top_roi.iterrows():
                    top_areas.append({
                        'name': row['locality'].title(),
                        'roi': row['roi_mean']
                    })
            
            return f"""📈 **Mumbai Real Estate Market Trends & Analysis 2024-25**

**🔥 CURRENT MARKET HOTSPOTS:**
{self.format_top_areas(top_areas)}

**📊 KEY MARKET INDICATORS:**

**🏠 Price Trends:**
• Suburban areas: 15-20% annual growth
• Premium locations: 8-12% steady appreciation  
• Emerging markets: 20-25% rapid growth
• Overall Mumbai: 12-15% average growth

**💰 ROI Performance:**
• High-yield areas: 12-18% (Kalyan, Dombivli, Vasai)
• Balanced areas: 8-12% (Thane, Powai, Goregaon)
• Premium areas: 6-10% (Bandra, Juhu, Worli)

**🚇 INFRASTRUCTURE IMPACT:**

**Metro Expansion Effects:**
• Line 2B (Thane-Powai): 25-30% price boost expected
• Line 7 (Malad-Kandivali): 20-25% appreciation
• Coastal Road: 15-20% impact on western suburbs

**🎯 2024-25 PREDICTIONS:**

**Growth Drivers:**
• IT sector expansion in Thane/Navi Mumbai
• Infrastructure completion (Metro, Coastal Road)
• Affordable housing demand surge
• Work-from-home trend favoring suburbs

**Market Opportunities:**
• **Q1-Q2 2024**: Best buying season (monsoon discounts)
• **Emerging hotspots**: Kalyan, Vasai, Virar showing 20%+ growth
• **Rental demand**: Sustained high demand, 8-12% rent growth

**💡 INVESTMENT STRATEGY FOR 2024:**

**Short-term (1-2 years):**
• Focus on ready-to-move properties
• Target high-rental-yield areas (Thane, Kalyan)
• Avoid under-construction in uncertain times

**Medium-term (3-5 years):**
• Invest in metro-connected areas
• Diversify across 2-3 micro-markets
• Consider commercial properties in IT hubs

**Long-term (5+ years):**
• Emerging suburbs with infrastructure plans
• Land parcels in developing corridors
• Premium properties for appreciation

**⚠️ RISK FACTORS TO WATCH:**
• Interest rate fluctuations
• Regulatory changes in real estate
• Economic slowdown impact
• Oversupply in certain micro-markets

**🔗 Want specific trend analysis for any area? Just ask!**

**Next Steps:**
• Explore specific localities: "Trends in Thane"
• Investment timing: "When to buy in Powai"
• Sector analysis: "IT hub property trends" """
        
        except Exception as e:
            return """📈 **Mumbai Real Estate Market Trends**

**🔥 Current Market Highlights:**
• Suburban growth outpacing central Mumbai
• Infrastructure development driving 20%+ appreciation
• Rental yields strongest in emerging areas
• Premium locations focusing on capital appreciation

**📊 Key Trends:**
• **Thane Corridor**: Fastest growing region
• **Metro Impact**: 25-30% price boost near stations  
• **IT Hubs**: Sustained rental demand
• **Affordable Housing**: High demand segment

**💡 Investment Opportunities:**
• Emerging suburbs for high ROI
• Metro-connected areas for appreciation
• IT corridors for rental income

🗺️ Check our [Heat Map](/roi-heatmap) for visual market trends!"""
    
    def format_top_areas(self, top_areas):
        """Format top performing areas"""
        if not top_areas:
            return "• Data being updated - check back soon!"
        
        formatted = ""
        for i, area in enumerate(top_areas, 1):
            emoji = "🔥" if area['roi'] >= 10 else "📈" if area['roi'] >= 8 else "📊"
            formatted += f"• **{area['name']}**: {area['roi']:.1f}% ROI {emoji}\n"
        
        return formatted
    
    def is_affordability_query(self, message, entities):
        """Check if user wants affordability analysis"""
        return any(word in message for word in ['afford', 'affordability', 'budget', 'down payment', 'income'])
    
    def handle_affordability(self, message, entities):
        """Handle affordability queries"""
        try:
            # Extract income or down payment info
            income = None
            down_payment = None
            
            if 'income' in message and entities['numbers']:
                income = entities['numbers'][0] * 1000  # Assume in thousands
            elif 'down payment' in message and entities['prices']:
                down_payment = entities['prices'][0] * 100000  # Convert to actual amount
            
            if income:
                max_emi = income * 0.4  # 40% rule
                max_loan = self.calculate_max_loan(max_emi)
                max_property_value = max_loan / 0.8  # Assuming 20% down payment
                
                return f"""💰 **Affordability Analysis**

**📊 BASED ON YOUR INCOME:**
• **Monthly Income**: ₹{income:,.0f}
• **Max EMI (40% rule)**: ₹{max_emi:,.0f}
• **Max Loan Amount**: ₹{max_loan/100000:.1f} Lakhs
• **Max Property Value**: ₹{max_property_value/100000:.1f} Lakhs
• **Required Down Payment**: ₹{max_property_value*0.2/100000:.1f} Lakhs

**🏠 RECOMMENDED AREAS:**
{self.get_affordable_areas(max_property_value/100000)}

**💡 AFFORDABILITY TIPS:**
• Keep EMI under 40% of income
• Maintain 6-month EMI emergency fund
• Factor in maintenance costs (₹2-3K/month)
• Consider location vs affordability trade-off

**📈 FINANCING OPTIONS:**
• **Home Loan**: 80% of property value
• **Interest Rate**: 8.5-9.5% current range
• **Tenure**: 15-30 years available
• **Processing Fee**: 0.5-1% of loan amount

Want specific property recommendations in your budget? Just ask!"""
            
            elif down_payment:
                max_property_value = down_payment / 0.2  # Assuming 20% down payment
                loan_amount = max_property_value * 0.8
                emi = self.calculate_emi(loan_amount, 8.5, 20)
                required_income = emi / 0.4
                
                return f"""💰 **Affordability Analysis**

**📊 BASED ON YOUR DOWN PAYMENT:**
• **Down Payment Available**: ₹{down_payment/100000:.1f} Lakhs
• **Max Property Value**: ₹{max_property_value/100000:.1f} Lakhs
• **Loan Required**: ₹{loan_amount/100000:.1f} Lakhs
• **Monthly EMI**: ₹{emi:,.0f}
• **Required Income**: ₹{required_income:,.0f}

**🏠 PROPERTY OPTIONS:**
{self.get_affordable_areas(max_property_value/100000)}

**💡 FINANCIAL PLANNING:**
• Ensure income supports ₹{emi:,.0f} EMI
• Keep additional funds for registration/legal costs
• Plan for monthly maintenance expenses
• Consider property insurance costs

Ready to explore specific properties in your budget?"""
            
            else:
                return """💰 **Affordability Calculator**
                
I can help you determine what you can afford based on:

**Option 1: Income-Based**
Tell me your monthly income and I'll calculate:
• Maximum EMI you can afford
• Maximum loan amount
• Property price range
• Down payment required

**Option 2: Down Payment-Based**  
Tell me your available down payment and I'll show:
• Maximum property value
• Required monthly income
• EMI obligations
• Suitable areas

**Examples:**
• "I earn ₹80,000 monthly, what can I afford?"
• "I have ₹20 lakhs down payment, what property can I buy?"
• "Affordability with ₹1 lakh monthly income"

**Quick Rules:**
• EMI should be <40% of income
• Down payment typically 20-25%
• Keep 6-month EMI as emergency fund

What's your income or available down payment?"""
        
        except Exception as e:
            return "I can help with affordability analysis! Please share your monthly income or available down payment, and I'll show you what properties you can afford."
    
    def calculate_max_loan(self, max_emi, interest_rate=8.5, tenure=20):
        """Calculate maximum loan amount based on EMI capacity"""
        monthly_rate = interest_rate / (12 * 100)
        total_months = tenure * 12
        
        if monthly_rate > 0:
            max_loan = max_emi * ((1 + monthly_rate) ** total_months - 1) / \
                      (monthly_rate * (1 + monthly_rate) ** total_months)
        else:
            max_loan = max_emi * total_months
        
        return max_loan
    
    def calculate_emi(self, loan_amount, interest_rate, tenure):
        """Calculate EMI for given loan parameters"""
        monthly_rate = interest_rate / (12 * 100)
        total_months = tenure * 12
        
        if monthly_rate > 0:
            emi = (loan_amount * monthly_rate * (1 + monthly_rate) ** total_months) / \
                  ((1 + monthly_rate) ** total_months - 1)
        else:
            emi = loan_amount / total_months
        
        return emi
    
    def get_affordable_areas(self, max_budget):
        """Get areas within budget range"""
        if max_budget >= 100:
            return """**Premium Options (₹80L-120L+):**
• Powai, Andheri, Bandra (outskirts)
• Thane West (premium projects)
• Goregaon East (commercial proximity)"""
        elif max_budget >= 60:
            return """**Mid-Range Options (₹40L-80L):**
• Thane, Goregaon, Malad
• Kandivali, Borivali
• Powai (older projects)"""
        elif max_budget >= 40:
            return """**Value Options (₹25L-50L):**
• Kalyan, Dombivli
• Vasai, Virar
• Thane (outskirts)"""
        else:
            return """**Budget Options (₹15L-35L):**
• Kalyan East, Dombivli
• Vasai East, Virar
• Badlapur, Ambernath"""
    
    def get_contextual_response(self, message, entities):
        """Generate contextual response based on conversation history and user context"""
        
        # Check user context for personalized response
        if self.user_context.get('budget_range'):
            budget_context = self.user_context['budget_range']
            if budget_context == 'high':
                context_intro = "Given your premium budget range, "
            elif budget_context == 'medium':
                context_intro = "For your mid-range budget, "
            else:
                context_intro = "With your budget-conscious approach, "
        else:
            context_intro = ""
        
        # Generate contextual suggestions
        suggestions = []
        if entities['localities']:
            suggestions.append(f"Learn more about {entities['localities'][0].title()}")
        if entities['prices']:
            suggestions.append(f"Explore options for ₹{entities['prices'][0]} lakh budget")
        
        # Default contextual response
        return f"""{context_intro}I'm here to help with Mumbai real estate insights!

🤔 **I didn't quite understand that, but I can help with:**

**🏠 Property Analysis:**
• ROI calculations for any locality
• Investment advice based on your budget
• Market comparisons between areas
• Risk assessment for investments

**💰 Financial Planning:**
• EMI calculations and affordability
• Down payment planning
• Rental yield analysis
• Portfolio diversification

**📊 Market Intelligence:**
• Current trends and predictions
• Area-specific insights
• Infrastructure impact analysis
• Growth potential assessment

**💡 Try asking:**
• "ROI for ₹50L property in Thane"
• "Compare Andheri vs Powai"
• "Investment advice for ₹1 crore"
• "Market trends in Mumbai"

{f"**Based on our conversation:** {', '.join(suggestions)}" if suggestions else ""}

What specific aspect of Mumbai real estate interests you most?"""

# Initialize chatbot service
def create_chatbot_service(ml_service):
    return PropTechChatbot(ml_service)
