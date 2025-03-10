# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
    ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
    
    # Risk Parameters
    MAX_EXPOSURE = 1000000  # $1M
    MAX_POSITION_SIZE = 0.1  # 10% of portfolio
    STOP_LOSS_PCT = 0.05  # 5% stop loss
    TAKE_PROFIT_PCT = 0.10  # 10% take profit
    MAX_KELLY_FRACTION = 0.5  # Maximum Kelly criterion fraction
    
    # Data Settings
    HISTORICAL_WINDOW = 252  # 1 year of trading days
    FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Trading Parameters
    TRADING_FREQUENCY = "daily"  # daily, hourly, etc.
    
    # Technical Analysis Parameters
    RSI_WINDOW = 14
    BOLLINGER_WINDOW = 20
    BOLLINGER_STD = 2.0
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    # Backtesting Parameters
    COMMISSION_RATE = 0.001  # 0.1% commission
    SLIPPAGE = 0.0005  # 5 basis points slippage
    
    # Paths
    LOG_PATH = "logs/"
    DATA_PATH = "data/"
    
    # Performance Metrics
    BENCHMARK = "SPY"  # Benchmark symbol for comparison
    RISK_FREE_RATE = 0.02  # 2% risk-free rate assumption