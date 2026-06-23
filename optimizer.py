import numpy as np
from scipy.optimize import minimize

def optimize_schedule(baseline_load, prices, carbon_intensities, weight_cost=0.5):
    """
    Optimizes the flexible part of the building energy load to minimize cost and/or carbon.
    
    Parameters:
    - baseline_load: array of 24 hourly load values (kWh)
    - prices: array of 24 hourly electricity rates ($/kWh)
    - carbon_intensities: array of 24 hourly grid carbon values (gCO2/kWh)
    - weight_cost: float between 0.0 (optimize only carbon) and 1.0 (optimize only cost)
    """
    n_hours = len(baseline_load)
    
    # 1. Split load: 80% is non-flexible, 20% is flexible
    non_flexible_load = baseline_load * 0.8
    baseline_flexible = baseline_load * 0.2
    total_flexible_needed = np.sum(baseline_flexible)
    
    # 2. Normalize price and carbon vectors so they have equal influence in optimization
    norm_prices = prices / np.mean(prices)
    norm_carbon = carbon_intensities / np.mean(carbon_intensities)
    
    # 3. Define the objective function to minimize
    # x is the array of 24 values representing the optimized flexible load for each hour
    def objective(x):
        cost_term = np.sum(x * norm_prices)
        carbon_term = np.sum(x * norm_carbon)
        # Combine using the user-specified weight slider
        return weight_cost * cost_term + (1 - weight_cost) * carbon_term
    
    # 4. Constraints
    # Constraint: The total sum of optimized flexible load MUST equal the sum of baseline flexible load.
    # (i.e. we are shifting energy, not deleting it)
    constraints = ({
        'type': 'eq',
        'fun': lambda x: np.sum(x) - total_flexible_needed
    })
    
    # 5. Bounds
    # The flexible load at any hour must be between 0 and 1.8x of the baseline flexible load
    # (limiting maximum grid draw capacity per hour)
    bounds = [(0, val * 1.8) for val in baseline_flexible]
    
    # 6. Initial guess (start with the baseline flexible distribution)
    x0 = baseline_flexible.copy()
    
    # 7. Run SciPy Optimizer (Sequential Least Squares Programming - SLSQP)
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not res.success:
        print("Optimization failed! Using baseline load.")
        return baseline_load, baseline_load
        
    optimized_flexible = res.x
    optimized_total_load = non_flexible_load + optimized_flexible
    
    return optimized_total_load, optimized_flexible

if __name__ == "__main__":
    # Test block to verify optimization math
    print("Testing Optimizer...")
    # Mocking 24 hours of data
    mock_load = np.array([10.0, 8.0, 7.0, 7.0, 8.0, 12.0, 20.0, 25.0, 30.0, 32.0, 30.0, 28.0, 
                          30.0, 32.0, 35.0, 38.0, 37.0, 32.0, 28.0, 25.0, 20.0, 15.0, 12.0, 10.0])
    
    mock_prices = np.array([0.08]*14 + [0.25]*7 + [0.08]*3) # Peak price $0.25 (hours 14-20)
    mock_carbon = np.array([350.0]*10 + [200.0]*6 + [350.0]*8) # Low carbon during solar peak (hours 10-15)
    
    # Optimize favoring cost reduction (weight = 0.8)
    opt_load, _ = optimize_schedule(mock_load, mock_prices, mock_carbon, weight_cost=0.8)
    
    baseline_cost = np.sum(mock_load * mock_prices)
    optimized_cost = np.sum(opt_load * mock_prices)
    
    print(f"Test complete!")
    print(f"Baseline Daily Cost: ${baseline_cost:.2f}")
    print(f"Optimized Daily Cost: ${optimized_cost:.2f} (Savings: {((baseline_cost - optimized_cost)/baseline_cost)*100:.1f}%)")