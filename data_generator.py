import pandas as pd
import numpy as np

def generate_building_data(days=365):
    """
    Generates a synthetic hourly dataset for a commercial building.
    Includes outdoor temperature, baseline loads, pricing, and grid carbon emissions.
    """
    np.random.seed(42)
    hours = days * 24
    timestamps = pd.date_range(start="2025-01-01", periods=hours, freq="h")
    
    # 1. Simulate Outdoor Temperature (sine wave + noise + seasonal drift)
    day_of_year = timestamps.dayofyear
    seasonal_temp = 15 + 12 * np.sin(2 * np.pi * (day_of_year - 120) / 365)
    hour_of_day = timestamps.hour
    daily_temp = 6 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    noise = np.random.normal(0, 1.5, hours)
    temperatures = seasonal_temp + daily_temp + noise
    
    # 2. Simulate Electricity Pricing ($/kWh)
    # High prices during peak hours (2 PM to 8 PM / 14:00 - 20:00), lower at night
    pricing = np.zeros(hours)
    for i, ts in enumerate(timestamps):
        if ts.hour >= 14 and ts.hour <= 20:
            pricing[i] = 0.25 + np.random.normal(0, 0.02) # Peak Price
        else:
            pricing[i] = 0.08 + np.random.normal(0, 0.01) # Off-peak Price
            
    # 3. Simulate Grid Carbon Intensity (gCO2/kWh)
    # Solar peak (noon to 4 PM) lowers carbon. Night coal/gas increases it.
    carbon_intensity = np.zeros(hours)
    for i, ts in enumerate(timestamps):
        base_intensity = 350 # average coal/gas base
        solar_effect = -150 * np.exp(-((ts.hour - 13)**2) / 8) # drop in afternoon due to solar grid input
        carbon_intensity[i] = base_intensity + solar_effect + np.random.normal(0, 10)
        
    # 4. Simulate Building Energy Consumption (kWh)
    # Baseline load + HVAC (highly temperature dependent) + Lighting
    hvac_load = np.maximum(0, (temperatures - 22) * 5.0) # cooling load above 22°C
    hvac_load += np.maximum(0, (18 - temperatures) * 3.5) # heating load below 18°C
    
    # Base occupancy load (higher on weekdays 8 AM - 6 PM)
    weekday_mask = timestamps.dayofweek < 5
    work_hours = (timestamps.hour >= 8) & (timestamps.hour <= 18)
    base_load = np.where(weekday_mask & work_hours, 25.0, 5.0) + np.random.normal(0, 1.2, hours)
    
    # Lighting load (turns on at night / early morning when dark)
    lighting_load = np.where((timestamps.hour >= 17) | (timestamps.hour <= 7), 8.0, 1.0)
    
    total_load = np.maximum(2.0, base_load + hvac_load + lighting_load)

    # Combine into a DataFrame
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Temperature": temperatures,
        "Electricity_Price": pricing,
        "Grid_Carbon_Intensity": carbon_intensity,
        "HVAC_Load": hvac_load,
        "Base_Load": base_load,
        "Lighting_Load": lighting_load,
        "Total_Load": total_load
    })
    
    return df

if __name__ == "__main__":
    print("Generating simulated building data...")
    df = generate_building_data()
    df.to_csv("building_energy_data.csv", index=False)
    print("Dataset saved as 'building_energy_data.csv' successfully! Shape:", df.shape)