import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import time
import requests # New: For Smart Search
from datetime import timedelta

# --- Configuration ---
st.set_page_config(page_title="Indian Stock Market Dashboard", layout="wide")

# Custom CSS
st.markdown("""
<style>
    div[data-testid="stMetricValue"] { font-size: 20px; }
    .big-font { font-size: 20px !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Pre-defined Indices
INDICES = {
    "Nifty 50": "^NSEI",
    "Nifty Bank": "^NSEBANK",
    "Nifty IT": "^CNXIT",
    "Nifty Auto": "^CNXAUTO",
    "Nifty Pharma": "^CNXPHARMA",
    "Nifty FMCG": "^CNXFMCG",
    "Nifty Metal": "^CNXMETAL",
    "Nifty Realty": "^CNXREALTY",
}

# --- State Management ---
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = None
if 'selected_name' not in st.session_state:
    st.session_state.selected_name = None
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""

# --- Helper Functions ---

def search_yahoo_finance(query):
    """
    Searches Yahoo Finance for the best matching Indian stock.
    Returns: (Symbol, Shortname) or (None, None)
    """
    try:
        # Yahoo Finance Search API
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {
            "q": query,
            "quotesCount": 10,
            "newsCount": 0,
            "region": "IN"
        }
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        
        if 'quotes' in data and len(data['quotes']) > 0:
            # Filter for Indian listings (NSE or BSE)
            # Prioritize NSE (.NS) over BSE (.BO)
            indian_stocks = [q for q in data['quotes'] if q.get('exchange') in ['NSI', 'BSE', 'NSE']]
            
            if not indian_stocks:
                # Fallback: Just take the top result if it looks like a ticker
                return data['quotes'][0]['symbol'], data['quotes'][0].get('shortname', query)
                
            # Look for NSE first
            for stock in indian_stocks:
                if stock['symbol'].endswith('.NS'):
                    return stock['symbol'], stock.get('shortname', stock['symbol'])
            
            # If no NSE, take the first Indian one (likely BSE)
            return indian_stocks[0]['symbol'], indian_stocks[0].get('shortname', indian_stocks[0]['symbol'])
            
        return None, None
    except Exception as e:
        print(f"Search Error: {e}")
        return None, None

def get_data(ticker_symbol):
    """Fetches data, calculates moving averages, and prepares dataframes."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="2y")
        
        if hist.empty:
            return None, None, None, None, None
            
        current_price = hist['Close'].iloc[-1]
        current_date = hist.index[-1]
        prev_close = hist['Close'].iloc[-2]
        change = current_price - prev_close
        pct_change = (change / prev_close) * 100
        
        # 1. Moving Averages Calculation
        past_data = hist['Close'].iloc[:-1]
        avgs = {
            "Yesterday": prev_close,
            "3-Day Avg": past_data.tail(3).mean(),
            "7-Day Avg": past_data.tail(7).mean(),
            "14-Day Avg": past_data.tail(14).mean(),
            "21-Day Avg": past_data.tail(21).mean(),
        }

        comparison_data = []
        for label, ref_value in avgs.items():
            if pd.isna(ref_value): continue
            diff = current_price - ref_value
            p_diff = (diff / ref_value) * 100
            comparison_data.append({
                "Metric": f"vs {label}",
                "Reference Price": ref_value,
                "Points Diff": diff,
                "% Diff": p_diff
            })
        
        comp_df = pd.DataFrame(comparison_data)

        # 2. Lows Analysis
        periods = [30, 45, 90, 150, 180, 240, 320, 360]
        lows_data = []
        
        for days in periods:
            cutoff_date = current_date - timedelta(days=days)
            mask = hist.index >= cutoff_date
            period_data = hist.loc[mask]
            
            if not period_data.empty:
                period_low = period_data['Low'].min()
                diff_from_low = current_price - period_low
                pct_above_low = (diff_from_low / period_low) * 100
                
                lows_data.append({
                    "Period": f"Last {days} Days",
                    "Lowest Price": period_low,
                    "Current Premium": diff_from_low,
                    "% Recovery": pct_above_low
                })
        
        lows_df = pd.DataFrame(lows_data)

        # 3. History DataFrame
        hist['Daily Change'] = hist['Close'].diff()
        hist['% Change'] = hist['Close'].pct_change() * 100
        display_df = hist[['Open', 'High', 'Low', 'Close', 'Daily Change', '% Change']].sort_index(ascending=False).head(30)
        
        return hist, (current_price, change, pct_change), display_df, comp_df, lows_df
        
    except Exception as e:
        return None, None, None, None, None

def color_change(val):
    if pd.isna(val): return ""
    color = '#00C805' if val > 0 else '#FF5000'
    return f'color: {color}'

def color_recovery(val):
    return f'color: #00C805; font-weight: bold;' 

def plot_sparkline(data):
    fig = go.Figure()
    subset = data.tail(30)
    start_val = subset['Close'].iloc[0]
    end_val = subset['Close'].iloc[-1]
    color = '#00C805' if end_val >= start_val else '#FF5000'
    
    fig.add_trace(go.Scatter(
        x=subset.index, y=subset['Close'], mode='lines',
        line=dict(color=color, width=2), fill='tozeroy',
        fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)"
    ))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=50, showlegend=False,
                      xaxis=dict(showgrid=False, showticklabels=False),
                      yaxis=dict(showgrid=False, showticklabels=False))
    return fig

def plot_full_chart(data, name):
    subset = data.tail(180) 
    fig = go.Figure(data=[go.Candlestick(
        x=subset.index,
        open=subset['Open'], high=subset['High'],
        low=subset['Low'], close=subset['Close'],
        name="Price"
    )])
    
    fig.update_layout(
        height=500,
        title=f"{name} - Price Movement (Last 180 Days)",
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )
    return fig

# --- Navigation & Search ---

def go_to_detail(name, ticker):
    st.session_state.selected_name = name
    st.session_state.selected_ticker = ticker
    st.rerun()

def go_back():
    st.session_state.selected_ticker = None
    st.session_state.selected_name = None
    st.rerun()

def handle_search():
    """Smart Search Handler"""
    query = st.session_state.search_query.strip()
    if query:
        with st.spinner(f"Searching for '{query}'..."):
            # 1. Try Smart Search First
            found_symbol, found_name = search_yahoo_finance(query)
            
            if found_symbol:
                st.session_state.selected_name = found_name
                st.session_state.selected_ticker = found_symbol
            else:
                # 2. Fallback: Blindly try adding .NS if search failed
                fallback_ticker = f"{query.upper()}.NS"
                st.session_state.selected_name = query.upper()
                st.session_state.selected_ticker = fallback_ticker

# --- Views ---

def show_sidebar():
    with st.sidebar:
        st.header("🔍 Smart Search")
        st.text_input(
            "Search Company (e.g., Zomato, Paytm)", 
            key="search_query", 
            on_change=handle_search,
            placeholder="Type name and hit Enter"
        )
        st.caption("✅ Now supports Company Names! Type 'Reliance', 'MRF', etc.")
        
        st.divider()
        st.markdown("**Quick Links:**")
        if st.button("🏠 Home Dashboard"):
            go_back()

def show_main_dashboard():
    st.title("🇮🇳 Indian Market Dashboard")
    st.caption(f"Live Data | Last Updated: {time.strftime('%H:%M:%S')}")
    
    if st.button('🔄 Refresh', type="primary"):
        st.rerun()

    st.subheader("Major Indices")
    cols = st.columns(3)
    
    for i, (name, ticker) in enumerate(INDICES.items()):
        with cols[i % 3]:
            hist_data, metrics, _, _, _ = get_data(ticker)
            
            if hist_data is not None:
                current, change, pct = metrics
                with st.container(border=True):
                    st.subheader(f"{name}")
                    st.metric(label="Current", value=f"{current:,.2f}", delta=f"{change:,.2f} ({pct:.2f}%)")
                    st.plotly_chart(plot_sparkline(hist_data), use_container_width=True, config={'displayModeBar': False})
                    st.button(f"Analyze {name}", key=f"btn_{ticker}", on_click=go_to_detail, args=(name, ticker))
            else:
                st.warning(f"Waiting for data: {name}")

def show_detailed_view():
    ticker = st.session_state.selected_ticker
    name = st.session_state.selected_name
    
    # Top Bar
    c1, c2 = st.columns([1, 8])
    with c1:
        st.button("← Back", on_click=go_back)
    with c2:
        st.header(f"📊 Analysis: {name}")
        st.caption(f"Symbol: {ticker}")

    # Fetch Data
    with st.spinner(f"Fetching deep data for {name}..."):
        hist_data, metrics, table_df, comp_df, lows_df = get_data(ticker)

    if hist_data is not None:
        current, change, pct = metrics
        
        # 1. Key Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Price", f"₹{current:,.2f}", delta=f"{change:,.2f} ({pct:.2f}%)")
        m2.metric("52-Week High", f"₹{hist_data.tail(252)['High'].max():,.2f}")
        m3.metric("52-Week Low", f"₹{hist_data.tail(252)['Low'].min():,.2f}")
        
        vol = hist_data.tail(30)['Volume'].mean()
        vol_str = f"{vol/100000:.1f} L" if vol > 0 else "N/A"
        m4.metric("Avg Vol (30d)", vol_str)
        
        st.divider()
        
        # 2. Analysis Tables
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("📉 Trend vs Moving Avgs")
            if not comp_df.empty:
                st.dataframe(
                    comp_df.style.format({
                        "Reference Price": "{:,.2f}",
                        "Points Diff": "{:+,.2f}",
                        "% Diff": "{:+.2f}%"
                    }).map(color_change, subset=['Points Diff', '% Diff']),
                    use_container_width=True,
                    hide_index=True
                )

        with col_right:
            st.subheader("🚀 Recovery from Lows")
            st.caption("Growth from lowest price in past X days")
            if not lows_df.empty:
                st.dataframe(
                    lows_df.style.format({
                        "Lowest Price": "{:,.2f}",
                        "Current Premium": "{:+,.2f}",
                        "% Recovery": "{:+.2f}%"
                    }).map(color_recovery, subset=['% Recovery']),
                    use_container_width=True,
                    hide_index=True
                )

        st.divider()

        # 3. Large Chart
        st.plotly_chart(plot_full_chart(hist_data, name), use_container_width=True)
        
        # 4. Data Table
        st.subheader("🗓️ Daily Historical Data")
        styled_df = table_df.style.format("{:,.2f}").map(color_change, subset=['Daily Change', '% Change'])
        st.dataframe(styled_df, use_container_width=True, height=400)
    else:
        st.error(f"❌ Could not load data for symbol: '{ticker}'.")
        st.info("Try searching for the exact Company Name again.")

# --- Main App Logic ---

show_sidebar()

if st.session_state.selected_ticker:
    show_detailed_view()
else:
    show_main_dashboard()