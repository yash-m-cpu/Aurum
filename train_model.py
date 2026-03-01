import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib

print("Loading data for training...")
# Load and clean data just like before
df = pd.read_csv('concrete_data.csv')
df.columns = [
    'Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water', 
    'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate', 
    'Age_Days', 'Strength_MPa'
]

# 1. Split the data into "Features" (X) and "Target" (y)
# X = Everything EXCEPT Strength (The ingredients and the age)
X = df.drop('Strength_MPa', axis=1)
# y = ONLY the Strength (What we want to predict)
y = df['Strength_MPa']

# 2. Split into Training Data (80%) and Testing Data (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

# 3. Initialize and Train the AI Model (XGBoost)
print("\nTraining the XGBoost AI model... (This might take a few seconds)")
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 4. Test the Model
print("Testing the model's accuracy...")
predictions = model.predict(X_test)

# Calculate the error margin
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n--- Model Performance ---")
print(f"🎯 Mean Absolute Error (MAE): {mae:.2f} MPa")
print(f"📈 Accuracy Score (R2): {r2:.2f} (closer to 1.0 is better)")
print("-------------------------")

if mae < 5.0:
    print("✅ Excellent! Your AI's predictions are highly accurate.")
else:
    print("⚠️ The error margin is a bit high, but okay for a prototype.")

# 5. Save the trained model so the web app can use it later!
joblib.dump(model, 'xgboost_concrete_model.pkl')
print("\n💾 Model saved successfully as 'xgboost_concrete_model.pkl'")