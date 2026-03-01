import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the dataset
# Make sure 'concrete_data.csv' is in the same directory as this script
print("Loading data...")
try:
    df = pd.read_csv('concrete_data.csv')
    print("✅ Data loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Could not find 'concrete_data.csv'. Please check the file name and location.")
    exit()

# 2. Clean up the column names
# The original dataset has very long, complex column names. Let's simplify them.
df.columns = [
    'Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water', 
    'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate', 
    'Age_Days', 'Strength_MPa'
]

# 3. Check for missing data
print("\n--- Checking for Missing Values ---")
print(df.isnull().sum())
# Note: This dataset is usually very clean, so everything should be 0.

# 4. Show a quick preview of our clean data
print("\n--- Data Preview ---")
print(df.head())

# 5. Visualize the relationships (Optional but great for your pitch deck!)
# This creates a heatmap showing how different ingredients affect Strength
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation: Ingredients vs. Concrete Strength")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
print("\n✅ Saved 'correlation_heatmap.png' to your folder. (Use this in your pitch!)")