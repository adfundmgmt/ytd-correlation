import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

# Must be FIRST Streamlit command
st.set_page_config(page_title="YTD Correlation Dashboard", layout="wide")

with st.sidebar:
    st.title("📘 How to Use This Tool")

    st.markdown("""
This dashboard compares the **current year-to-date (YTD) performance** of a chosen ticker (e.g., S&P 500 or Nasdaq) with prior years, based on **correlation of daily return paths**.

---

### 🔧 Steps:

1. **Enter a ticker**  
   Examples:
   - `^GSPC` (S&P 500)
   - `^IXIC` (Nasdaq Composite)
   - `AAPL`, `TSLA`, etc.

2. **Adjust the Top N slider**  
   This selects how many past years to overlay based on similarity to the current year.

3. **Interpret the chart**  
   - **Black line** = current year's YTD path  
   - **Dashed lines** = top correlated historical years  
   - **Legend** shows correlation coefficients (ρ)

---

### 💡 Tip:
This tool is useful for:
- Identifying analog years
- Market narrative framing
- Backtesting

Developed by **AD Fund Management LP**.
""")

st.title("📈 YTD Analog Year Correlation Explorer")

# === Inputs ===
ticker = st.text_input("Enter ticker symbol (e.g. ^GSPC, ^IXIC, AAPL)", value="^GSPC").upper()
top_n = st.slider("Top N analog years to show", 1, 10, 5)

@st.cache_data
def fetch_data(ticker):
    df = yf.download(ticker, start="1980-01-01", auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, axis=1, level=1, drop_level=True)
    df = df[['Close']].dropna()
    df['Year'] = df.index.year
    return df

# === Data Fetching & Validation ===
try:
    data = fetch_data(ticker)
except:
    st.error("Failed to download data. Check ticker symbol.")
    st.stop()

# === YTD Return Matrix Calculation ===
ytd_returns_by_year = {}
for year, group in data.groupby('Year'):
    if len(group) < 30:
        continue
    try:
        start_price = group['Close'].iloc[0]
        ytd = (group['Close'] / start_price) - 1
        ytd.index = group.index.dayofyear
        if ytd.isnull().any() or ytd.size < 30:
            continue
        ytd_returns_by_year[year] = ytd
    except:
        continue

ytd_df = pd.DataFrame(ytd_returns_by_year)
current_year = datetime.datetime.now().year

if current_year not in ytd_df.columns:
    st.warning(f"No valid YTD data for {current_year}")
    st.stop()

current_ytd = ytd_df[current_year].dropna()
correlations = {}

for year in ytd_df.columns:
    if year == current_year:
        continue
    past_ytd = ytd_df[year].dropna()
    min_len = min(len(current_ytd), len(past_ytd))
    if min_len < 30:
        continue
    corr = np.corrcoef(current_ytd[:min_len], past_ytd[:min_len])[0, 1]
    correlations[year] = corr

# === Top Analog Matches ===
top_matches = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:top_n]

st.subheader(f"Top {top_n} most correlated years to {current_year}")
for year, corr in top_matches:
    st.write(f"**{year}**: ρ = {corr:.4f}")

# === Plotting ===
fig, ax = plt.subplots(figsize=(14, 7))
days_current = list(range(1, len(current_ytd) + 1))
ax.plot(days_current, current_ytd, label=f"{current_year} (YTD)", linewidth=3, color='black')

for year, corr in top_matches:
    analog = ytd_df[year].dropna()
    ax.plot(analog.index, analog.values, linestyle='--', linewidth=1.8, label=f"{year} (ρ={corr:.2f})")

ax.set_title(f"{ticker} YTD ({current_year}) vs Full-Year Analogs", fontsize=15)
ax.set_xlabel("Trading Day of Year")
ax.set_ylabel("Cumulative Return")
ax.axhline(0, color='gray', linestyle='--', linewidth=1)
ax.set_xlim(1, 253)
ax.grid(True, linestyle=':', linewidth=0.6)
ax.legend(loc="upper left", fontsize=9)

# Format Y-axis nicely
ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))

st.pyplot(fig)
