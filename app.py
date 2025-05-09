import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# ----------------------------- 🔧 PAGE CONFIG -----------------------------
st.set_page_config(
    page_title="YTD Analog Year Correlation Explorer",
    layout="wide",
    initial_sidebar_state="auto"
)

# ----------------------------- 📘 HOW TO USE -----------------------------
st.title("📈 YTD Analog Year Correlation Explorer")
st.markdown("""
This app compares the **Year-To-Date (YTD) performance** of a selected ticker (e.g., S&P 500, Nasdaq, or a stock like AAPL)
to the same ticker’s performance in **previous years** based on return correlation.

- Enter any valid Yahoo Finance ticker.
- It will download daily historical data since 1980.
- Then it finds the most similar YTD paths in history.
- Correlation (ρ) is calculated based on trading days so far this year.
- It **plots full-year analogs** to help you visualize how those years ended.

---

""")

# ----------------------------- 🎛️ UI INPUTS -----------------------------
ticker = st.text_input("Enter ticker symbol (e.g. ^GSPC, ^IXIC, AAPL)", value="^GSPC")
top_n = st.slider("Top N analog years to show", 1, 10, 5)

# ----------------------------- 📥 FETCH DATA -----------------------------
@st.cache_data(show_spinner=False)
def load_data(ticker):
    df = yf.download(ticker, start="1980-01-01", auto_adjust=True)
    df = df[["Close"]].dropna()
    df["Year"] = df.index.year
    return df

try:
    df = load_data(ticker)
except Exception as e:
    st.error(f"Failed to load data for {ticker}: {e}")
    st.stop()

# ----------------------------- 📊 CALCULATE YTD RETURNS -----------------------------
ytd_returns_by_year = {}

for year, group in df.groupby("Year"):
    try:
        start_price = group["Close"].iloc[0]
        returns = (group["Close"] / start_price) - 1
        returns.index = group.index.dayofyear
        ytd_returns_by_year[year] = returns
    except Exception:
        continue

if not ytd_returns_by_year:
    st.warning(f"No YTD return data found for {ticker.upper()}. Try a different symbol (e.g. ^GSPC or AAPL).")
    st.stop()

ytd_df = pd.DataFrame(ytd_returns_by_year)

# ----------------------------- 📈 FIND TOP CORRELATIONS -----------------------------
current_year = datetime.datetime.now().year

if current_year not in ytd_df.columns:
    st.warning(f"No YTD return data available for {ticker.upper()} in {current_year}.")
    st.stop()

current_ytd = ytd_df[current_year].dropna()
correlations = {}

for year in ytd_df.columns:
    if year == current_year:
        continue
    historical = ytd_df[year].dropna()
    min_len = min(len(current_ytd), len(historical))
    if min_len < 30:
        continue
    corr = np.corrcoef(current_ytd[:min_len], historical[:min_len])[0, 1]
    correlations[year] = corr

if not correlations:
    st.warning(f"No valid historical correlations found for {ticker.upper()} vs {current_year}.")
    st.stop()

# ----------------------------- 🏆 TOP ANALOGS -----------------------------
top_matches = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:top_n]

st.markdown(f"### Top {top_n} most correlated years to **{current_year}**")
for year, corr in top_matches:
    st.markdown(f"**{year}**: ρ = `{corr:.4f}`")

# ----------------------------- 📉 PLOT -----------------------------
st.markdown("---")
fig, ax = plt.subplots(figsize=(14, 6))
days = current_ytd.index
ax.plot(days, current_ytd, label=f"{current_year}", linewidth=3, color="black")

for year, corr in top_matches:
    full_year = ytd_df[year].dropna()
    ax.plot(full_year.index, full_year, linestyle="--", linewidth=1.6, label=f"{year} (ρ={corr:.2f})")

ax.set_title(f"{ticker.upper()} YTD {current_year} vs Historical Analogs", fontsize=16)
ax.set_xlabel("Trading Day of Year")
ax.set_ylabel("YTD Return")
ax.axhline(0, color="gray", linewidth=1, linestyle="--")
ax.grid(True, linestyle=':', linewidth=0.6)
ax.legend()
st.pyplot(fig)
