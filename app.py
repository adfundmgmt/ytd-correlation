import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import datetime

# Set Streamlit page configuration
st.set_page_config(page_title="YTD Correlation Dashboard", layout="wide")

# Sidebar instructions
with st.sidebar:
    st.title("📘 How to Use This Tool")
    st.markdown("""
This dashboard compares the **current year-to-date (YTD) performance** of a chosen ticker with prior years, based on **correlation of daily return paths**.

---

### 🔧 Steps:
1. **Enter a ticker**  
   Examples:
   - `^GSPC` (S&P 500)
   - `^IXIC` (Nasdaq Composite)
   - `AAPL`, `TSLA`, etc.

2. **Adjust the Top N slider**  
   Choose how many analog years to show.

3. **Interpret the chart**  
   - **Black line** = current YTD  
   - **Dashed lines** = top correlated years  
   - **Legend** shows correlation ρ (Pearson)

---

### 💡 Use Cases:
- Analog-based market playbooks
- Historical quant framing
- Macro and technical overlays

Developed by **AD Fund Management LP**.
""")

# User input
st.title("📈 YTD Analog Year Correlation Explorer")
ticker = st.text_input("Enter ticker symbol (e.g. ^GSPC, ^IXIC, AAPL)", "^GSPC")
top_n = st.slider("Top N analog years to show", 1, 10, 5)

# Download historical data
hist = yf.download(ticker, start="1980-01-01", auto_adjust=False, progress=False)
hist = hist[['Close']].dropna()
hist['Year'] = hist.index.year
hist['DOY'] = hist.index.dayofyear

# Compute YTD return curves
ytd_returns_by_year = {}
for year, group in hist.groupby('Year'):
    if len(group) < 30:
        continue
    start_price = group.iloc[0]['Close']
    ytd_returns = group['Close'] / start_price - 1
    ytd_returns.index = group['DOY']
    if isinstance(ytd_returns, pd.Series) and ytd_returns.notna().sum() > 30:
        ytd_returns_by_year[year] = ytd_returns

# Assemble matrix
ytd_df = pd.DataFrame(ytd_returns_by_year)
current_year = datetime.datetime.now().year
if current_year not in ytd_df.columns or ytd_df[current_year].dropna().empty:
    valid_years = [y for y in sorted(ytd_df.columns, reverse=True) if ytd_df[y].dropna().size > 30]
    if valid_years:
        fallback_year = valid_years[0]
        st.info(f"No YTD data found for {current_year}. Showing data for {fallback_year} instead.")
        current_year = fallback_year
    else:
        st.error(f"No YTD return data available for {ticker.upper()}.")
        st.stop()


current_ytd = ytd_df[current_year].dropna()
correlations = {}
for year in ytd_df.columns:
    if year == current_year:
        continue
    other = ytd_df[year].dropna()
    min_len = min(len(current_ytd), len(other))
    if min_len < 30:
        continue
    correlations[year] = np.corrcoef(current_ytd[:min_len], other[:min_len])[0, 1]

top_matches = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:top_n]

# Show top correlations
st.markdown(f"""
### 📊 Top {top_n} Historical Analogs to **{current_year} YTD**
These years had the most similar return profiles so far. Use them to frame expectations, market setups, or macro parallels.
""")

cols = st.columns(len(top_matches))
for i, (year, corr) in enumerate(top_matches):
    with cols[i]:
        st.metric(label=f"**{year}**", value=f"{corr:.2f}", delta="ρ", delta_color="off")

# Full-year return function
def get_full_year_return(ticker, year):
    try:
        df = yf.download(ticker, start=f"{year}-01-01", end=f"{year}-12-31", progress=False)
        start_price = df['Close'].iloc[0]
        end_price = df['Close'].iloc[-1]
        return (end_price / start_price - 1) * 100
    except:
        return None

# Commentary
best_year = top_matches[0][0]
best_corr = top_matches[0][1]
analog_return = get_full_year_return(ticker, best_year)

insight = f"""
> 🧠 **{best_year}** stands out as the closest YTD analog to **{current_year}**, with a correlation of **{best_corr:.2f}**.
"""
if analog_return is not None:
    insight += f" That year ended with a total return of **{analog_return:.1f}%**."
insight += "\n\n> Use this as a framing tool for positioning, or scroll below to compare full-year return paths visually."

st.markdown(insight)

# CSV export
export_df = pd.DataFrame(top_matches[:top_n], columns=["Year", "Correlation"])
csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Export Analog Years to CSV", csv, f"top_{top_n}_analogs_{ticker}_{current_year}.csv", "text/csv")

# Plot full-year overlays
st.subheader(f"📈 Full-Year Performance: {current_year} vs. Top Analogs")
plt.figure(figsize=(14, 7))
plt.plot(current_ytd.values, label=f"{current_year} (YTD)", color="black", linewidth=2)

for year, _ in top_matches:
    if year in ytd_df.columns:
        full_series = ytd_df[year]
        plt.plot(full_series.values, label=f"{year}", linestyle='--')

plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Trading Day of Year")
plt.ylabel("Return")
plt.title(f"{ticker.upper()} — YTD and Analog Years ({current_year})")
plt.legend()
plt.grid(True, linestyle=":", linewidth=0.5)
st.pyplot(plt)

