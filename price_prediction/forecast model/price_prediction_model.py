import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt

# ✅ Load the CSV (adjusted for your actual path)
df = pd.read_csv("data/Final_Project.csv")

# ✅ Select only the required features based on your dataset
features = ['Location', 'Bedroom', 'Area_SqFt', 'Bathroom', 'Availability', 'Area_Tpye', 'Property_Age']
target = 'Price_Lakh'

# ✅ Drop missing values
df = df[features + [target]].dropna()

# ✅ One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=['Location', 'Availability', 'Area_Tpye', 'Property_Age'])

# ✅ Define X and y
X = df_encoded.drop(target, axis=1)
y = df_encoded[target]

# ✅ Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n✅ Model trained successfully!")
print(f"📉 MAE: ₹{mae:,.2f} Lakh")
print(f"📈 R² Score: {r2:.4f}")

# ✅ Save the model
joblib.dump(model, "model/price_prediction_model.pkl")
print("💾 Model saved as: model/price_prediction_model.pkl")

# ✅ Optional: Compare actual vs predicted prices
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Price (Lakh)")
plt.ylabel("Predicted Price (Lakh)")
plt.title("Actual vs Predicted Price")
plt.grid(True)
plt.tight_layout()
plt.show()