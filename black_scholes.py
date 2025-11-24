import math
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Calculates the Black-Scholes option pricing model for European options.
    
    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiry in years
    r (float): Risk-free interest rate as a decimal
    sigma (float): Volatility as a decimal
    option_type (str): Type of option - "call" or "put"
    
    Returns:
    float: Option price based on the Black-Scholes formula
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return price

def generate_sensitivity_heatmap(K, T, r, sigma, option_type):
    """
    Generates a heatmap for option pricing sensitivity based on varying stock prices and volatilities.
    """
    stock_prices = np.linspace(50, 150, 20)  # Range of stock prices
    volatilities = np.linspace(0.1, 0.5, 20)  # Range of volatilities
    
    data = []
    for vol in volatilities:
        row = [black_scholes(S, K, T, r, vol, option_type) for S in stock_prices]
        data.append(row)
    
    df = pd.DataFrame(data, index=volatilities, columns=stock_prices)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, cmap="coolwarm", annot=False, xticklabels=5, yticklabels=5)
    plt.xlabel("Stock Price")
    plt.ylabel("Volatility")
    plt.title(f"{option_type.capitalize()} Option Sensitivity Analysis")
    st.pyplot(plt)

def generate_pnl_heatmap(K, T, r, sigma, option_type, purchase_price):
    """
    Generates a heatmap for P&L based on varying stock prices and volatilities.
    """
    stock_prices = np.linspace(50, 150, 20)  # Range of stock prices
    volatilities = np.linspace(0.1, 0.5, 20)  # Range of volatilities
    
    data = []
    for vol in volatilities:
        row = [black_scholes(S, K, T, r, vol, option_type) - purchase_price for S in stock_prices]
        data.append(row)
    
    df = pd.DataFrame(data, index=volatilities, columns=stock_prices)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, cmap="RdYlGn", annot=False, xticklabels=5, yticklabels=5)
    plt.xlabel("Stock Price")
    plt.ylabel("Volatility")
    plt.title(f"{option_type.capitalize()} Option P&L Sensitivity")
    st.pyplot(plt)

# Streamlit UI
st.title("Black-Scholes Option Pricing Calculator")

S = st.number_input("Enter stock price", min_value=0.01, format="%.2f")
K = st.number_input("Enter strike price", min_value=0.01, format="%.2f")
T = st.number_input("Enter time to expiry (years)", min_value=0.01, format="%.2f")
r = st.number_input("Enter risk-free interest rate (decimal)", min_value=0.0, format="%.4f")
sigma = st.number_input("Enter volatility (decimal)", min_value=0.01, format="%.4f")
option_type = st.selectbox("Select option type", ["call", "put"])
purchase_price = st.number_input("Enter purchase price of option", min_value=0.0, format="%.4f")

if st.button("Calculate Option Price"):
    price = black_scholes(S, K, T, r, sigma, option_type)
    st.write(f"### {option_type.capitalize()} Option Price: {price:.4f}")

if st.button("Generate Sensitivity Heatmap"):
    generate_sensitivity_heatmap(K, T, r, sigma, option_type)

if st.button("Generate P&L Heatmap"):
    generate_pnl_heatmap(K, T, r, sigma, option_type, purchase_price)
