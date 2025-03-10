# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from agent import trading_agent, TradingState
from tools import TradingTools
from logger import logger
from config import Config
import yfinance as yf
import time
import json
import os

# Page configuration
st.set_page_config(
    page_title="AI Trading Agent Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_state():
    """Initialize the session state if it doesn't exist"""
    if 'trading_state' not in st.session_state:
        # Try to load initial market data
        try:
            initial_data = TradingTools.get_market_data('AAPL', period="1y")
        except Exception as e:
            st.warning(f"Could not load initial data: {e}")
            initial_data = pd.DataFrame()
            
        st.session_state.trading_state = {
            'symbol': 'AAPL',
            'portfolio_value': 1_000_000,
            'positions': {},
            'market_data': initial_data,
            'risk_metrics': {},
            'signals': {},
            'trades': [],
            'strategies': {
                'trend_following': True,
                'mean_reversion': True,
                'monte_carlo': True
            }
        }
    
    if 'history' not in st.session_state:
        st.session_state.history = {
            'portfolio_values': [1_000_000],  # Initial portfolio value
            'timestamps': [datetime.now()]
        }
        
    if 'logs' not in st.session_state:
        st.session_state.logs = []

def get_current_price(symbol):
    """Get the current price for a symbol"""
    try:
        ticker = yf.Ticker(symbol)
        todays_data = ticker.history(period='1d')
        if not todays_data.empty:
            return todays_data['Close'].iloc[-1]
    except Exception as e:
        logger.logger.warning(f"Error getting current price for {symbol}: {e}")
    
    # Return the last known price from our data
    data = st.session_state.trading_state['market_data']
    if not data.empty:
        return data['Close'].iloc[-1]
    return 0.0

def plot_market_data(data, positions):
    """Create an interactive plot of market data with technical indicators"""
    if data.empty:
        return None
    
    # Create subplots
    fig = make_subplots(rows=3, cols=1, 
                         shared_xaxes=True,
                         vertical_spacing=0.05,
                         row_heights=[0.6, 0.2, 0.2],
                         subplot_titles=("Price & Indicators", "Volume", "RSI"))
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="OHLC"
        ),
        row=1, col=1
    )
    
    # Add moving averages if they exist
    if 'sma_20' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['sma_20'],
                line=dict(color='blue', width=1),
                name="SMA 20"
            ),
            row=1, col=1
        )
    
    if 'sma_50' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['sma_50'],
                line=dict(color='red', width=1),
                name="SMA 50"
            ),
            row=1, col=1
        )
    
    # Add Bollinger Bands if they exist
    if 'bb_upper' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['bb_upper'],
                line=dict(color='rgba(0,128,0,0.3)', width=1),
                name="BB Upper"
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['bb_lower'],
                line=dict(color='rgba(0,128,0,0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(0,128,0,0.1)',
                name="BB Lower"
            ),
            row=1, col=1
        )
    
    # Add buy/sell markers for trades
    if 'trades' in st.session_state:
        for trade in st.session_state.trading_state['trades']:
            trade_date = trade['timestamp']
            # Find the closest date in our data
            if isinstance(data.index[0], pd.Timestamp):
                closest_idx = data.index.get_indexer([pd.Timestamp(trade_date)], method='nearest')[0]
                closest_date = data.index[closest_idx]
                
                marker_color = 'green' if trade['action'] in ['BUY'] else 'red'
                marker_symbol = 'triangle-up' if trade['action'] in ['BUY'] else 'triangle-down'
                
                fig.add_trace(
                    go.Scatter(
                        x=[closest_date],
                        y=[trade['price']],
                        mode='markers',
                        marker=dict(color=marker_color, size=12, symbol=marker_symbol),
                        name=f"{trade['action']} {trade['quantity']} @ {trade['price']:.2f}"
                    ),
                    row=1, col=1
                )
    
    # Add volume chart
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            marker_color='rgba(0,0,255,0.5)',
            name="Volume"
        ),
        row=2, col=1
    )
    
    # Add RSI if it exists
    if 'rsi' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['rsi'],
                line=dict(color='purple', width=1),
                name="RSI"
            ),
            row=3, col=1
        )
        
        # Add RSI guide lines
        fig.add_trace(
            go.Scatter(
                x=[data.index[0], data.index[-1]],
                y=[70, 70],
                line=dict(color='red', width=1, dash='dash'),
                name="RSI Overbought"
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[data.index[0], data.index[-1]],
                y=[30, 30],
                line=dict(color='green', width=1, dash='dash'),
                name="RSI Oversold"
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text=f"Technical Analysis for {st.session_state.trading_state['symbol']}",
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )
    
    # Update y-axis for RSI
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    
    return fig

def plot_portfolio_performance():
    """Plot the portfolio performance over time"""
    history = st.session_state.history
    
    if not history['portfolio_values'] or len(history['portfolio_values']) < 2:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=history['timestamps'],
            y=history['portfolio_values'],
            mode='lines',
            name="Portfolio Value",
            line=dict(color='blue', width=2)
        )
    )
    
    # Calculate and add a benchmark (e.g., SPY)
    if 'benchmark_values' in history:
        fig.add_trace(
            go.Scatter(
                x=history['timestamps'],
                y=history['benchmark_values'],
                mode='lines',
                name="S&P 500 (Benchmark)",
                line=dict(color='gray', width=1, dash='dash')
            )
        )
    
    fig.update_layout(
        title="Portfolio Performance",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        template="plotly_white",
        height=400,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def run_agent():
    """Execute one iteration of the trading agent"""
    # Get the current state
    state = st.session_state.trading_state.copy()
    
    # Execute the agent
    new_state = trading_agent.invoke(state)
    
    # Update the session state
    st.session_state.trading_state = new_state
    
    # Update portfolio history
    st.session_state.history['portfolio_values'].append(new_state['portfolio_value'])
    st.session_state.history['timestamps'].append(datetime.now())
    
    # Log portfolio update
    logger.log_portfolio_update(new_state['portfolio_value'], new_state['positions'])
    
    return new_state

def update_market_data():
    """Update market data for the selected symbol and period"""
    symbol = st.session_state.trading_state['symbol']
    period = st.session_state.period if 'period' in st.session_state else "1y"
    
    try:
        data = TradingTools.get_market_data(symbol, period=period)
        if data.empty:
            st.error(f"No data available for {symbol}")
            return False
        
        st.session_state.trading_state['market_data'] = data
        return True
    except Exception as e:
        st.error(f"Error updating market data: {e}")
        logger.logger.error(f"Error updating market data: {e}")
        return False

def format_position_table(positions):
    """Format positions for display in a table"""
    if not positions:
        return pd.DataFrame()
    
    position_data = []
    for symbol, details in positions.items():
        current_price = get_current_price(symbol)
        market_value = details['quantity'] * current_price
        profit_loss = (current_price - details['avg_price']) * details['quantity']
        profit_loss_pct = (current_price / details['avg_price'] - 1) * 100 if details['avg_price'] > 0 else 0
        
        position_data.append({
            'Symbol': symbol,
            'Quantity': details['quantity'],
            'Avg Price': f"${details['avg_price']:.2f}",
            'Current Price': f"${current_price:.2f}",
            'Market Value': f"${market_value:.2f}",
            'P/L': f"${profit_loss:.2f}",
            'P/L %': f"{profit_loss_pct:.2f}%",
            'Stop Loss': f"${details.get('stop_loss', 0):.2f}",
            'Take Profit': f"${details.get('take_profit', 0):.2f}"
        })
    
    return pd.DataFrame(position_data)

def format_trades_table(trades):
    """Format trades for display in a table"""
    if not trades:
        return pd.DataFrame()
    
    trade_data = []
    for trade in trades:
        trade_data.append({
            'Date': trade['timestamp'].strftime('%Y-%m-%d %H:%M'),
            'Symbol': trade['symbol'],
            'Action': trade['action'],
            'Quantity': trade['quantity'],
            'Price': f"${trade['price']:.2f}",
            'Value': f"${trade['value']:.2f}"
        })
    
    return pd.DataFrame(trade_data)

def update_benchmark_data():
    """Update benchmark (S&P 500) data for comparison"""
    try:
        benchmark_data = TradingTools.get_market_data('SPY', period="1y")
        if benchmark_data.empty:
            logger.logger.warning("No benchmark data available")
            return
        
        # Normalize both series to start at the same value
        first_portfolio_value = st.session_state.history['portfolio_values'][0]
        first_benchmark_price = benchmark_data['Close'].iloc[0]
        benchmark_scale = first_portfolio_value / first_benchmark_price
        
        # Calculate benchmark values proportional to portfolio
        benchmark_prices = benchmark_data['Close'] * benchmark_scale
        
        # Interpolate benchmark values to match portfolio timestamps
        benchmark_df = pd.DataFrame(benchmark_prices)
        benchmark_df.columns = ['value']
        
        # Create a DataFrame with portfolio timestamps
        portfolio_df = pd.DataFrame({
            'timestamp': st.session_state.history['timestamps'],
            'value': st.session_state.history['portfolio_values']
        })
        portfolio_df.set_index('timestamp', inplace=True)
        
        # Combine and interpolate
        if len(benchmark_df) > 1:
            # Resample benchmark data to daily
            benchmark_daily = benchmark_df.resample('D').last().interpolate()
            
            # For each portfolio timestamp, find the closest benchmark value
            benchmark_values = []
            for ts in st.session_state.history['timestamps']:
                # Find closest date in benchmark data
                closest_date = benchmark_daily.index[benchmark_daily.index.get_indexer([ts], method='nearest')[0]]
                benchmark_values.append(benchmark_daily.loc[closest_date, 'value'])
            
            st.session_state.history['benchmark_values'] = benchmark_values
    
    except Exception as e:
        logger.logger.warning(f"Error updating benchmark data: {e}")

def calculate_performance_metrics():
    """Calculate performance metrics for the portfolio"""
    if len(st.session_state.history['portfolio_values']) < 2:
        return {}
    
    portfolio_values = st.session_state.history['portfolio_values']
    initial_value = portfolio_values[0]
    current_value = portfolio_values[-1]
    
    # Calculate returns
    total_return = (current_value / initial_value - 1) * 100
    
    # Calculate daily returns
    returns = []
    for i in range(1, len(portfolio_values)):
        daily_return = (portfolio_values[i] / portfolio_values[i-1]) - 1
        returns.append(daily_return)
    
    # Calculate sharpe ratio if we have enough data
    sharpe_ratio = 0
    if len(returns) > 1:
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return > 0:
            # Annualized Sharpe ratio (assuming daily returns)
            sharpe_ratio = (avg_return / std_return) * np.sqrt(252)
    
    # Calculate max drawdown
    max_drawdown = 0
    peak = initial_value
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # Benchmark comparison
    benchmark_comparison = "N/A"
    if 'benchmark_values' in st.session_state.history:
        benchmark_values = st.session_state.history['benchmark_values']
        if len(benchmark_values) > 1:
            benchmark_return = (benchmark_values[-1] / benchmark_values[0] - 1) * 100
            benchmark_comparison = f"{total_return - benchmark_return:.2f}%"
    
    # Trading activity
    num_trades = len(st.session_state.trading_state['trades'])
    win_rate = 0
    if num_trades > 0:
        profitable_trades = 0
        for trade in st.session_state.trading_state['trades']:
            if trade['action'] in ['SELL', 'TAKE_PROFIT']:
                # Simplified win calculation
                profitable_trades += 1
        
        if num_trades > 0:
            win_rate = (profitable_trades / num_trades) * 100
    
    return {
        'Total Return': f"{total_return:.2f}%",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Max Drawdown': f"{max_drawdown:.2f}%",
        'Num Trades': num_trades,
        'Win Rate': f"{win_rate:.1f}%",
        'Alpha vs. S&P 500': benchmark_comparison
    }

def main():
    """Main Streamlit application"""
    # Initialize session state
    initialize_state()
    
    # Set up the sidebar
    with st.sidebar:
        st.title("AI Trading Agent")
        
        # Symbol selector
        new_symbol = st.text_input("Symbol", value=st.session_state.trading_state['symbol'])
        if new_symbol != st.session_state.trading_state['symbol']:
            st.session_state.trading_state['symbol'] = new_symbol
            update_market_data()
        
        # Data period selector
        period_options = {
            "1 Month": "1mo",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y"
        }
        selected_period = st.selectbox(
            "Data Period", 
            options=list(period_options.keys()),
            index=3  # Default to 1 Year
        )
        st.session_state.period = period_options[selected_period]
        
        # Trading strategy toggles
        st.header("Trading Strategies")
        strategies = st.session_state.trading_state['strategies']
        
        strategies['trend_following'] = st.checkbox(
            "Trend Following", 
            value=strategies.get('trend_following', True),
            help="Uses moving averages and MACD to identify trends"
        )
        
        strategies['mean_reversion'] = st.checkbox(
            "Mean Reversion", 
            value=strategies.get('mean_reversion', True),
            help="Uses RSI and Bollinger Bands to identify overbought/oversold conditions"
        )
        
        strategies['monte_carlo'] = st.checkbox(
            "Monte Carlo", 
            value=strategies.get('monte_carlo', True),
            help="Uses Monte Carlo simulations to forecast price movements"
        )
        
        # Risk parameters
        st.header("Risk Parameters")
        
        max_position = st.slider(
            "Max Position Size (%)", 
            min_value=1, 
            max_value=50, 
            value=int(Config.MAX_POSITION_SIZE * 100),
            help="Maximum percentage of portfolio allocated to a single position"
        )
        Config.MAX_POSITION_SIZE = max_position / 100
        
        max_kelly = st.slider(
            "Max Kelly Fraction", 
            min_value=0.1, 
            max_value=1.0, 
            value=Config.MAX_KELLY_FRACTION,
            help="Maximum Kelly criterion fraction for position sizing"
        )
        Config.MAX_KELLY_FRACTION = max_kelly
        
        stop_loss = st.slider(
            "Stop Loss (%)", 
            min_value=1, 
            max_value=20, 
            value=int(Config.STOP_LOSS_PCT * 100),
            help="Percentage below entry price to place stop loss"
        )
        Config.STOP_LOSS_PCT = stop_loss / 100
        
        take_profit = st.slider(
            "Take Profit (%)", 
            min_value=1, 
            max_value=50, 
            value=int(Config.TAKE_PROFIT_PCT * 100),
            help="Percentage above entry price to place take profit"
        )
        Config.TAKE_PROFIT_PCT = take_profit / 100
        
        # Actions
        st.header("Actions")
        if st.button("Update Market Data"):
            with st.spinner("Updating market data..."):
                success = update_market_data()
                if success:
                    st.success("Market data updated!")
        
        if st.button("Execute Agent"):
            with st.spinner("Running trading agent..."):
                run_agent()
                update_benchmark_data()
                st.success("Trading cycle completed!")
                st.experimental_rerun()
    
    # Main content area
    st.title("AI Trading Agent Dashboard")
    
    # Portfolio summary
    col1, col2, col3 = st.columns(3)
    portfolio_value = st.session_state.trading_state['portfolio_value']
    
    with col1:
        st.metric(
            label="Portfolio Value",
            value=f"${portfolio_value:,.2f}",
            delta=f"${portfolio_value - 1_000_000:,.2f}"
        )
    
    with col2:
        # Calculate cash position
        positions = st.session_state.trading_state['positions']
        invested_value = 0
        for symbol, position in positions.items():
            invested_value += position['quantity'] * get_current_price(symbol)
        cash = portfolio_value - invested_value
        
        st.metric(
            label="Cash Available",
            value=f"${cash:,.2f}",
            delta=f"{cash/portfolio_value*100:.1f}%"
        )
    
    with col3:
        metrics = calculate_performance_metrics()
        if metrics:
            st.metric(
                label="Total Return",
                value=metrics['Total Return'],
                delta=metrics.get('Alpha vs. S&P 500', "N/A")
            )
    
    # Portfolio performance graph
    st.header("Portfolio Performance")
    performance_fig = plot_portfolio_performance()
    if performance_fig:
        st.plotly_chart(performance_fig, use_container_width=True)
    else:
        st.info("Not enough data to plot portfolio performance yet.")
    
    # Performance metrics
    if metrics:
        st.subheader("Performance Metrics")
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.metric("Sharpe Ratio", metrics['Sharpe Ratio'])
        
        with metrics_col2:
            st.metric("Max Drawdown", metrics['Max Drawdown'])
        
        with metrics_col3:
            st.metric("Number of Trades", metrics['Num Trades'])
        
        with metrics_col4:
            st.metric("Win Rate", metrics['Win Rate'])
    
    # Market data display
    st.header("Market Analysis")
    market_data = st.session_state.trading_state['market_data']
    if not market_data.empty:
        fig = plot_market_data(market_data, st.session_state.trading_state['positions'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No market data available for {st.session_state.trading_state['symbol']}. Please update market data.")
    
    # Display current positions and trade signals
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Positions")
        positions_df = format_position_table(st.session_state.trading_state['positions'])
        if not positions_df.empty:
            st.dataframe(positions_df, use_container_width=True)
        else:
            st.info("No open positions")
    
    with col2:
        st.subheader("Trading Signals")
        signals = st.session_state.trading_state['signals']
        if signals:
            signals_df = pd.DataFrame({
                'Signal': list(signals.keys()),
                'Value': list(signals.values())
            })
            st.dataframe(signals_df, use_container_width=True)
            
            # Highlight trade decision if available
            if 'trade_decision' in signals:
                decision = signals['trade_decision']
                confidence = signals.get('confidence', 0)
                
                decision_color = "gray"
                if decision == "BUY":
                    decision_color = "green"
                elif decision == "SELL":
                    decision_color = "red"
                
                st.markdown(f"<h3 style='color: {decision_color};'>Decision: {decision} (Confidence: {confidence:.2f})</h3>", unsafe_allow_html=True)
        else:
            st.info("No signals generated yet")
    
    # Recent trades
    st.header("Recent Trades")
    trades_df = format_trades_table(st.session_state.trading_state['trades'])
    if not trades_df.empty:
        st.dataframe(trades_df, use_container_width=True)
    else:
        st.info("No trades executed yet")
    
    # Activity Log
    st.header("Activity Log")
    if st.session_state.logs:
        logs = "\n".join(st.session_state.logs)
        st.text_area("Recent activity", logs, height=200)
    else:
        st.info("No activity logged yet")

if __name__ == "__main__":
    main()