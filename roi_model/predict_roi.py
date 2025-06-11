import joblib
import numpy as np
import pandas as pd

# Load your trained model and encoder (adjust paths as needed)
try:
    model = joblib.load("roi_model/roi_model.pkl")
    locality_encoder = joblib.load("roi_model/locality_encoder.pkl")
    MODEL_LOADED = True
except:
    MODEL_LOADED = False
    model = None
    locality_encoder = None

def predict_roi(locality, price):
    """
    Predict ROI for a given locality and price
    
    Args:
        locality (str): Locality name
        price (float): Property price
        
    Returns:
        float: Predicted ROI percentage
    """
    if not MODEL_LOADED:
        raise Exception("ROI model not loaded")
    
    try:
        # Clean locality name
        locality_clean = locality.lower().strip()
        
        # Check if locality is in encoder classes
        if locality_clean not in locality_encoder.classes_:
            # Try to find a partial match
            available_localities = locality_encoder.classes_
            matches = [loc for loc in available_localities if locality_clean in loc or loc in locality_clean]
            if matches:
                locality_clean = matches[0]
            else:
                raise Exception(f"Locality '{locality}' not found in training data")
        
        # Encode locality
        locality_encoded = locality_encoder.transform([locality_clean])[0]
        
        # Make prediction
        prediction = model.predict(np.array([[locality_encoded, price]]))[0]
        
        return float(prediction)
        
    except Exception as e:
        raise Exception(f"Error predicting ROI: {str(e)}")

# Command-line interface (only runs when script is executed directly)
if __name__ == "__main__":
    print("🏠 ROI Prediction Model")
    
    if not MODEL_LOADED:
        print("❌ Model files not found!")
        exit(1)
    
    try:
        locality = input("Enter locality: ").strip()
        price = float(input("Enter property price (in INR): "))
        
        roi = predict_roi(locality, price)
        print(f"📈 Predicted ROI for {locality.title()} at ₹{price:,.0f} is {roi:.2f}%")
        
    except ValueError:
        print("❌ Please enter valid numeric input for price.")
    except Exception as e:
        print(f"❌ Error: {e}")
