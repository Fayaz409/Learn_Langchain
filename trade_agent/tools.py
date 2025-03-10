# tools.py
import yfinance as yf
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from config import Config

class TradingTools:
    @staticmethod
    def get_market_data(symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch and preprocess market data
        
        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame with OHLCV data and additional technical indicators
        """
        try:
            data = yf.download(symbol, period=period, progress=False)
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
                
            # Keep only required columns and ensure proper column names
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Handle missing data
            data = data.ffill().bfill()
            
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value-at-Risk using historical simulation
        
        Args:
            returns: Series of historical returns
            confidence: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            VaR value as a positive percentage
        """
        if returns.empty:
            return 0.0
        
        return abs(np.percentile(returns, 100 * (1 - confidence)))
    
    @staticmethod
    def monte_carlo_sim(returns: pd.Series, days: int = 5, sims: int = 1000) -> np.ndarray:
        """Monte Carlo simulation for future returns
        
        Args:
            returns: Series of historical returns
            days: Number of days to simulate
            sims: Number of simulations
            
        Returns:
            Array of simulated future prices
        """
        if returns.empty:
            return np.array([1.0])
        
        mu = returns.mean()
        sigma = returns.std()
        
        # Generate random daily returns
        daily_returns = np.random.normal(mu, sigma, (days, sims))
        
        # Calculate cumulative returns
        return np.cumprod(1 + daily_returns, axis=0)
    
    @staticmethod
    def calculate_position_size(portfolio_value: float, price: float, kelly_fraction: float = None) -> int:
        """Risk-aware position sizing with optional Kelly criterion
        
        Args:
            portfolio_value: Current portfolio value
            price: Current asset price
            kelly_fraction: Optional Kelly criterion fraction (0-1)
            
        Returns:
            Number of shares to trade
        """
        if not kelly_fraction:
            # Default position sizing based on portfolio percentage
            max_risk = portfolio_value * Config.MAX_POSITION_SIZE
        else:
            # Kelly-based position sizing
            max_risk = portfolio_value * kelly_fraction * Config.MAX_POSITION_SIZE
        
        # Calculate number of shares and round down
        return int(max_risk // price) if price > 0 else 0
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index
        
        Args:
            prices: Series of price data
            window: RSI window period
            
        Returns:
            Series containing RSI values
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.copy()
        gain[gain < 0] = 0
        loss = -delta.copy()
        loss[loss < 0] = 0
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate RS and RSI (handle division by zero)
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands
        
        Args:
            prices: Series of price data
            window: Moving average window
            num_std: Number of standard deviations
            
        Returns:
            Tuple of (upper band, middle band, lower band)
        """
        # Calculate middle band (SMA)
        middle_band = prices.rolling(window=window).mean()
        
        # Calculate standard deviation
        rolling_std = prices.rolling(window=window).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (rolling_std * num_std)
        lower_band = middle_band - (rolling_std * num_std)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: Series of price data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Tuple of (MACD line, signal line, histogram)
        """
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """Calculate maximum drawdown percentage
        
        Args:
            prices: Series of price data
            
        Returns:
            Maximum drawdown as a positive percentage
        """
        if len(prices) < 2:
            return 0.0
            
        # Calculate cumulative maximum
        roll_max = prices.cummax()
        
        # Calculate drawdown
        drawdown = (prices / roll_max - 1.0)
        
        # Return maximum drawdown as positive value
        return abs(drawdown.min())
    
    @staticmethod
    def calculate_sharpe_ratio(prices: pd.Series, risk_free_rate: float = 0.02, annualization: int = 252) -> float:
        """Calculate Sharpe ratio
        
        Args:
            prices: Series of price data
            risk_free_rate: Annual risk-free rate (default 2%)
            annualization: Annualization factor (252 trading days)
            
        Returns:
            Sharpe ratio
        """
        if len(prices) < 2:
            return 0.0
            
        # Calculate daily returns
        returns = prices.pct_change().dropna()
        
        if returns.empty:
            return 0.0
            
        # Calculate annualized return and volatility
        daily_rf = risk_free_rate / annualization
        excess_returns = returns - daily_rf
        
        # Calculate Sharpe ratio (handle zero standard deviation)
        std = returns.std()
        if std == 0:
            return 0.0
            
        sharpe = np.sqrt(annualization) * excess_returns.mean() / std
        
        return sharpe
    
    @staticmethod
    def backtest_strategy(prices: pd.Series, signals: pd.Series, initial_capital: float = 100000.0) -> Dict:
        """Backtest a trading strategy
        
        Args:
            prices: Series of price data
            signals: Series of trading signals (1 for buy, -1 for sell, 0 for hold)
            initial_capital: Starting capital
            
        Returns:
            Dictionary of performance metrics
        """
        # Create positions series (1 or -1)
        positions = signals.shift(1).fillna(0).astype(int)
        
        # Calculate returns
        returns = prices.pct_change().fillna(0)
        
        # Calculate strategy returns
        strategy_returns = positions * returns
        
        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # Calculate metrics
        total_return = (cumulative_returns.iloc[-1] - 1) * 100
        annual_return = ((cumulative_returns.iloc[-1]) ** (252 / len(cumulative_returns)) - 1) * 100
        
        # Calculate drawdown
        drawdown = (cumulative_returns / cumulative_returns.cummax() - 1) * 100
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        
        # Calculate number of trades
        trades = positions.diff().fillna(0).abs().sum() / 2
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'trades': trades
        }