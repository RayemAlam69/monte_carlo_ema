import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Download daily gold data
data = yf.download("GC=F", period="3mo", interval="1d")
close_prices = data['Close']['GC=F']

# Calculate daily returns
returns = np.log(close_prices / close_prices.shift(1)).dropna()
mu = returns.mean()
sigma = returns.std()
print(f"Average daily return: {mu:.6f}, Daily volatility: {sigma:.6f}")

# Monte Carlo simulation
S0 = close_prices[-1]
T = 5
steps_per_day = 24
steps = T * steps_per_day
paths = 500

dt = 1 / steps_per_day
sigma_hourly = sigma / np.sqrt(steps_per_day)
mu_hourly = mu / steps_per_day

simulations = np.zeros((steps, paths))
simulations[0] = S0

for t in range(1, steps):
    Z = np.random.normal(size=paths)
    simulations[t] = simulations[t-1] * np.exp((mu_hourly - 0.5*sigma_hourly**2)*dt + sigma_hourly*np.sqrt(dt)*Z)

plt.figure(figsize=(12,6))
plt.plot(simulations)
plt.title("Monte Carlo Simulation: GC=F (Gold Futures)")
plt.xlabel("Hourly steps")
plt.ylabel("Price")
plt.show()