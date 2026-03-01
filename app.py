import streamlit as st
import pandas as pd
import joblib
import requests
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Page Setup ---
st.set_page_config(page_title="CreaTech Precast Optimizer", layout="wide")
st.title("🏗️ AI Precast Cycle Time Optimizer")
st.markdown("Predict optimal element cycle time using AI, physics (Concrete Maturity), and live climatic conditions.")

# --- 2. Load the AI Model ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('xgboost_concrete_model.pkl')
    except FileNotFoundError:
        st.error("Model file not found! Please run train_model.py first.")
        st.stop()

model = load_model()

# --- 3. Sidebar: User Inputs ---
st.sidebar.header("1. Concrete Mix Design (kg/m³)")
cement = st.sidebar.slider("Cement", 100.0, 500.0, 350.0)
slag = st.sidebar.slider("Blast Furnace Slag", 0.0, 300.0, 50.0)
ash = st.sidebar.slider("Fly Ash", 0.0, 200.0, 0.0)
water = st.sidebar.slider("Water", 100.0, 250.0, 180.0)
superplast = st.sidebar.slider("Superplasticizer", 0.0, 20.0, 5.0)
coarse_agg = st.sidebar.slider("Coarse Aggregate", 800.0, 1200.0, 1000.0)
fine_agg = st.sidebar.slider("Fine Aggregate", 500.0, 900.0, 750.0)

st.sidebar.header("2. Project & Operations")
project_type = st.sidebar.selectbox("Element Type (Infrastructure vs Building)", 
                                    ["Building: Wall Panel (Target: 15 MPa)", 
                                     "Infrastructure: Bridge Girder (Target: 30 MPa)"])
city = st.sidebar.selectbox("Yard Location (Live Weather)", ["Chennai", "Delhi", "Mumbai", "Bangalore"])
curing_method = st.sidebar.radio("Curing Method", ["Ambient (Natural Weather)", "Steam Curing (Accelerated 60°C)"])
automation_level = st.sidebar.radio("Yard Automation Level", ["High (Automated Gantry/Robotics)", "Manual (Workers)"])

# Extract Target Strength dynamically based on project type
target_strength = 15 if "15" in project_type else 30

# --- 4. Fetch Live Weather Data ---
def get_weather(city_name):
    coords = {
        "Chennai": (13.0827, 80.2707),
        "Delhi": (28.6139, 77.2090),
        "Mumbai": (19.0760, 72.8777),
        "Bangalore": (12.9716, 77.5946)
    }
    lat, lon = coords[city_name]
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        response = requests.get(url).json()
        return response['current_weather']['temperature']
    except:
        return 25.0 # Fallback temperature in case API fails

current_temp = get_weather(city)
st.info(f"🌤️ **Live Weather in {city}:** {current_temp}°C")

# --- 5. The Engineering AI Engine (Nurse-Saul Maturity Method) ---
st.subheader("⏱️ Total Element Cycle Time Prediction")

# Set actual curing temperature and calculate costs
if "Steam" in curing_method:
    actual_curing_temp = 60.0  # Steam curing is typically 60°C
    curing_cost_penalty = 500  # Extra fuel cost per m³
else:
    actual_curing_temp = current_temp
    curing_cost_penalty = 0

# Set Operational Reset Time (De-moulding to next pour)
reset_time_hours = 2 if "High" in automation_level else 6

# Nurse-Saul Equivalent Age Calculation
datum_temp = -10.0
standard_temp = 20.0
maturity_factor = (actual_curing_temp - datum_temp) / (standard_temp - datum_temp)

# Simulate chronological curing over time
actual_hours_array = np.arange(6, 169, 1) # Check from 6h to 168h
predictions = []

for actual_hour in actual_hours_array:
    actual_days = actual_hour / 24.0
    
    # Calculate EQUIVALENT age to feed to the AI
    equivalent_age_days = actual_days * maturity_factor
    
    input_data = pd.DataFrame([[cement, slag, ash, water, superplast, coarse_agg, fine_agg, equivalent_age_days]], 
                              columns=['Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water', 
                                       'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate', 'Age_Days'])
    
    # Predict strength using the Equivalent Age
    predicted_strength = model.predict(input_data)[0]
    predictions.append(predicted_strength)

# Find the exact chronological hour it hits target strength
demould_time_hours = None
for i, strength in enumerate(predictions):
    if strength >= target_strength:
        demould_time_hours = actual_hours_array[i]
        break

# --- 6. Output & Visualization ---
col1, col2 = st.columns(2)

with col1:
    if demould_time_hours:
        total_cycle_time = demould_time_hours + reset_time_hours
        st.success(f"### 🔄 Total Cycle Time: {int(total_cycle_time)} Hours")
        st.write(f"**Breakdown:**")
        st.write(f"- ⏳ Curing Time to {target_strength} MPa: **{int(demould_time_hours)} hours**")
        st.write(f"- 🏗️ Mould Reset Time ({'Automated' if reset_time_hours == 2 else 'Manual'}): **{reset_time_hours} hours**")
    else:
        st.error(f"### 🔴 Target {target_strength} MPa not reached within 7 days.")
        st.write("Consider upgrading to Steam Curing or altering the mix design.")
        
    st.markdown("---")
    st.subheader("💰 Financial Impact")
    material_cost = (cement * 6) + (slag * 4) + (superplast * 50) + (coarse_agg * 1) + (fine_agg * 1)
    total_cost = material_cost + curing_cost_penalty
    st.write(f"**Total Cost per m³:** ₹{total_cost:,.2f}")
    if curing_cost_penalty > 0:
        st.caption(f"(Includes ₹{curing_cost_penalty} premium for Steam Curing energy costs)")


with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(actual_hours_array, predictions, label='AI Predicted Strength', color='#2E86C1', linewidth=2.5)
    ax.axhline(y=target_strength, color='#E74C3C', linestyle='--', label=f'Target ({target_strength} MPa)')
    
    if demould_time_hours:
        ax.axvline(x=demould_time_hours, color='#27AE60', linestyle=':', label=f'De-mould at {int(demould_time_hours)}h')
    
    ax.set_xlabel('Chronological Time (Hours)')
    ax.set_ylabel('Compressive Strength (MPa)')
    ax.set_title(f'Real-World Curing Curve ({curing_method.split()[0]})')
    ax.set_xlim(6, 72)
    ax.legend()
    st.pyplot(fig)


# --- 7. THE AI AUTO-OPTIMIZER (Prescriptive Analytics) ---
st.markdown("---")
st.header("🧠 AI Auto-Optimizer (Lowest Cost Finder)")
st.write("Set your constraints below. The AI will simulate 1,000 combinations to find the cheapest mix that hits your target.")

col3, col4 = st.columns(2)
with col3:
    target_opt_time = st.number_input("Required De-mould Time (Hours)", min_value=6, max_value=72, value=18)
with col4:
    target_opt_strength = st.number_input("Required Strength (MPa)", min_value=10, max_value=50, value=20)

if st.button("🚀 Run 1,000 Simulations"):
    with st.spinner("Simulating 1,000 mix combinations..."):
        # 1. Generate 1,000 random but realistic mix designs
        n_sims = 1000
        sim_cement = np.random.uniform(200, 450, n_sims)
        sim_slag = np.random.uniform(0, 150, n_sims)
        sim_ash = np.random.uniform(0, 100, n_sims)
        sim_water = np.random.uniform(140, 200, n_sims)
        sim_super = np.random.uniform(0, 15, n_sims)
        sim_coarse = np.random.uniform(850, 1100, n_sims)
        sim_fine = np.random.uniform(600, 850, n_sims)
        
        # Apply Maturity Factor to the target chronological time
        chronological_days = target_opt_time / 24.0
        equivalent_age_days = chronological_days * maturity_factor
        
        sim_age = np.full(n_sims, equivalent_age_days) 
        
        # Create a DataFrame for the AI to predict in bulk
        sim_df = pd.DataFrame({
            'Cement': sim_cement, 'Blast_Furnace_Slag': sim_slag, 'Fly_Ash': sim_ash, 
            'Water': sim_water, 'Superplasticizer': sim_super, 
            'Coarse_Aggregate': sim_coarse, 'Fine_Aggregate': sim_fine, 'Age_Days': sim_age
        })
        
        # 2. Predict strength for all 1,000 mixes instantly
        sim_df['Predicted_Strength'] = model.predict(sim_df)
        
        # 3. Calculate the cost for each mix (Materials + Curing Penalty)
        sim_df['Cost_INR'] = (sim_df['Cement']*6) + (sim_df['Blast_Furnace_Slag']*4) + (sim_df['Superplasticizer']*50) + (sim_df['Coarse_Aggregate']*1) + (sim_df['Fine_Aggregate']*1) + curing_cost_penalty
        
        # 4. Filter and Find the Winner
        valid_mixes = sim_df[sim_df['Predicted_Strength'] >= target_opt_strength]
        
        if valid_mixes.empty:
            st.error("❌ Impossible constraints! The AI couldn't find a mix to reach that strength in that time. Try a lower strength or longer time.")
        else:
            # Sort by cost (lowest first) and get the top 1
            best_mix = valid_mixes.sort_values(by='Cost_INR', ascending=True).iloc[0]
            
            st.success("### 🎉 Optimal Mix Found!")
            st.write(f"Out of 1,000 simulated combinations, this is the most cost-effective mix to reach **{target_opt_strength} MPa** in **{target_opt_time} hours** under current {city} weather with {curing_method.split()[0]} curing.")
            
            # Display results in a clean table
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(label="Estimated Total Cost", value=f"₹{best_mix['Cost_INR']:,.2f}")
                st.metric(label="Predicted Strength", value=f"{best_mix['Predicted_Strength']:.2f} MPa")
            
            with res_col2:
                st.write("**The Optimized Recipe (kg/m³):**")
                recipe_df = pd.DataFrame({
                    "Ingredient": ["Cement", "Slag", "Fly Ash", "Water", "Superplasticizer", "Coarse Agg", "Fine Agg"],
                    "Amount": [f"{best_mix['Cement']:.1f}", f"{best_mix['Blast_Furnace_Slag']:.1f}", 
                               f"{best_mix['Fly_Ash']:.1f}", f"{best_mix['Water']:.1f}", 
                               f"{best_mix['Superplasticizer']:.1f}", f"{best_mix['Coarse_Aggregate']:.1f}", 
                               f"{best_mix['Fine_Aggregate']:.1f}"]
                })
                st.dataframe(recipe_df, hide_index=True)