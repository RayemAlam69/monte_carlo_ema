import numpy as np

# -----------------------
# PARAMETERS
# -----------------------
S0 = 100      # Initial stock price
K = 105       # Option strike price
T = 1         # Time to expiration in years
r = 0.05      # Risk-free interest rate (annual)
sigma = 0.2   # Volatility (annual)
paths = 5000  # Number of simulated stock paths
steps = 252   # Daily steps in one year

# Time increment per step
dt = T / steps

# Fixing random numbers for reproducibility
np.random.seed(42)

# -----------------------
# SIMULATE STOCK PATHS
# -----------------------
# An array to hold all paths: rows=time, columns=simulations
S = np.zeros((steps+1, paths))
S[0] = S0  # First row = initial stock price

# Loop through time steps
for t in range(1, steps+1):
    # Random shock for each path
    Z = np.random.normal(size=paths)
    # GBM formula for next price
    S[t] = S[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

# -----------------------
# CALCULATE OPTION PRICE
# -----------------------
# European Call Payoff
payoff = np.maximum(S[-1] - K, 0)
# Discount back to present value
option_price = np.exp(-r * T) * np.mean(payoff)

print("Monte Carlo Option Price:", option_price)