# logger.py
import logging
from datetime import datetime
import streamlit as st
import os
import json

class TradingLogger:
    def __init__(self):
        self.logger = logging.getLogger("trading_agent")
        self.logger.setLevel(logging.DEBUG)
        
        # Ensure we don't duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
        
    def _setup_handlers(self):
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # File handler
        log_filename = f"logs/trading_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_filename)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(file_format)
        console_handler.setLevel(logging.INFO)
        
        # Streamlit handler
        class StreamlitHandler(logging.Handler):
            def emit(self, record):
                log_message = self.format(record)
                if 'logs' in st.session_state:
                    st.session_state.logs.append(log_message)
                    # Keep only the last 100 logs
                    if len(st.session_state.logs) > 100:
                        st.session_state.logs = st.session_state.logs[-100:]
        
        streamlit_handler = StreamlitHandler()
        streamlit_handler.setFormatter(file_format)
        streamlit_handler.setLevel(logging.INFO)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(streamlit_handler)
    
    def log_trade(self, action: str, symbol: str, quantity: int, price: float) -> dict:
        """Log a trade execution and return trade data"""
        timestamp = datetime.now()
        
        trade_data = {
            'timestamp': timestamp,
            'action': action,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'value': quantity * price
        }
        
        # Convert to JSON-serializable format for logging
        log_data = {**trade_data, 'timestamp': timestamp.isoformat()}
        self.logger.info(f"TRADE: {json.dumps(log_data)}")
        
        return trade_data
    
    def log_portfolio_update(self, portfolio_value: float, positions: dict) -> None:
        """Log portfolio updates"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': portfolio_value,
            'positions': positions
        }
        self.logger.info(f"PORTFOLIO: {json.dumps(log_data)}")
    
    def log_strategy_signal(self, strategy: str, signal: str, data: dict) -> None:
        """Log strategy signals"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'signal': signal,
            'data': data
        }
        self.logger.info(f"SIGNAL: {json.dumps(log_data)}")

# Initialize logger
if 'logs' not in st.session_state:
    st.session_state.logs = []

logger = TradingLogger()