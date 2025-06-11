# roi_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load cleaned dataset
df = pd.read_csv('data/roi_dataset_cleaned.csv')

# Drop rows with missing values
df = df.dropna(subset=['Locality', 'Price', 'ROI (%)'])

# Clean Locality text
df['Locality'] = df['Locality'].astype(str).str.strip().str.lower()

# Features and target
X = df[['Locality', 'Price']].copy()
y = df['ROI (%)']

# Encode Locality
le = LabelEncoder()
X['Locality'] = le.fit_transform(X['Locality'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"✅ Model trained. RMSE: {rmse:.2f}")

# Save model and encoder
joblib.dump(model, 'roi_model/roi_model.pkl')
joblib.dump(le, 'roi_model/locality_encoder.pkl')
print("💾 Model and encoder saved successfully in roi_model/")