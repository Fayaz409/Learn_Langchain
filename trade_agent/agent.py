# agent.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Literal, Union, Annotated
import pandas as pd
from tools import TradingTools
import numpy as np
from logger import logger
from config import Config

class TradingState(TypedDict):
    symbol: str
    portfolio_value: float
    positions: Dict[str, Dict[str, Union[int, float]]]  # Enhanced position tracking
    market_data: pd.DataFrame
    risk_metrics: Dict[str, float]
    signals: Dict[str, float]
    trades: List[Dict]
    strategies: Dict[str, bool]  # Which strategies are active

def analyze_market(state: TradingState) -> TradingState:
    """Market analysis node - generates signals based on technical analysis"""
    data = state['market_data']
    
    if data.empty:
        logger.logger.warning("No market data available for analysis")
        return state
    
    # Calculate returns
    returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    
    # Technical indicators
    data['sma_20'] = data['Close'].rolling(window=20).mean()
    data['sma_50'] = data['Close'].rolling(window=50).mean()
    data['rsi'] = TradingTools.calculate_rsi(data['Close'])
    data['bb_upper'], data['bb_middle'], data['bb_lower'] = TradingTools.calculate_bollinger_bands(data['Close'])
    data['macd'], data['macd_signal'], data['macd_hist'] = TradingTools.calculate_macd(data['Close'])
    
    # Generate signals
    current_price = data['Close'].iloc[-1]
    signals = {
        'current_price': current_price,
        'var_95': TradingTools.calculate_var(returns),
        'monte_carlo': TradingTools.monte_carlo_sim(returns).mean(),
        'trend': 1 if data['sma_20'].iloc[-1] > data['sma_50'].iloc[-1] else -1,
        'rsi': data['rsi'].iloc[-1],
        'macd_signal': 1 if data['macd'].iloc[-1] > data['macd_signal'].iloc[-1] else -1,
        'bb_position': (current_price - data['bb_lower'].iloc[-1]) / (data['bb_upper'].iloc[-1] - data['bb_lower'].iloc[-1])
    }
    
    logger.logger.info(f"Generated signals: {signals}")
    
    # Update state with enhanced market data and signals
    state['market_data'] = data
    state['signals'] = signals
    return state

def generate_trading_decision(state: TradingState) -> TradingState:
    """Decide whether to buy, sell, or hold based on active strategies"""
    signals = state['signals']
    strategies = state['strategies']
    symbol = state['symbol']
    decision = "HOLD"
    confidence = 0
    
    # Check if we have all necessary data
    if not signals or 'current_price' not in signals:
        logger.logger.warning("Insufficient signal data for trading decision")
        return state
    
    # Trend following strategy
    if strategies.get('trend_following', False):
        trend_signal = signals['trend']
        macd_signal = signals['macd_signal']
        
        if trend_signal > 0 and macd_signal > 0:
            decision = "BUY"
            confidence += 0.4
        elif trend_signal < 0 and macd_signal < 0:
            decision = "SELL"
            confidence += 0.4
    
    # Mean reversion strategy
    if strategies.get('mean_reversion', False):
        rsi = signals['rsi']
        bb_position = signals['bb_position']
        
        if rsi < 30 and bb_position < 0.2:
            decision = "BUY"
            confidence += 0.3
        elif rsi > 70 and bb_position > 0.8:
            decision = "SELL"
            confidence += 0.3
    
    # Monte Carlo simulation strategy
    if strategies.get('monte_carlo', False):
        monte_carlo = signals['monte_carlo']
        current_price = signals['current_price']
        
        if monte_carlo > current_price * 1.05:  # 5% expected gain
            decision = "BUY"
            confidence += 0.3
        elif monte_carlo < current_price * 0.95:  # 5% expected loss
            decision = "SELL"
            confidence += 0.3
    
    # Store the decision in signals
    signals['trade_decision'] = decision
    signals['confidence'] = confidence
    
    logger.logger.info(f"Trading decision: {decision} with confidence {confidence}")
    state['signals'] = signals
    return state

def risk_assessment(state: TradingState) -> TradingState:
    """Risk management node - calculate position sizes and set stop losses"""
    signals = state['signals']
    current_value = state['portfolio_value']
    
    if not signals or 'current_price' not in signals:
        logger.logger.warning("Insufficient data for risk assessment")
        return state
    
    price = signals['current_price']
    decision = signals.get('trade_decision', 'HOLD')
    confidence = signals.get('confidence', 0)
    
    # Position sizing based on Kelly Criterion and confidence
    if decision != "HOLD" and confidence > 0:
        kelly_fraction = min(confidence, Config.MAX_KELLY_FRACTION)
        position_size = TradingTools.calculate_position_size(
            current_value, 
            price, 
            kelly_fraction=kelly_fraction
        )
    else:
        position_size = 0
    
    # Risk checks
    risk_metrics = {
        'max_position_size': position_size,
        'stop_loss': price * (1 - Config.STOP_LOSS_PCT),
        'take_profit': price * (1 + Config.TAKE_PROFIT_PCT),
        'var_95': signals['var_95'],
        'max_drawdown': TradingTools.calculate_max_drawdown(state['market_data']['Close']),
        'sharpe_ratio': TradingTools.calculate_sharpe_ratio(state['market_data']['Close'])
    }
    
    logger.logger.info(f"Risk metrics: {risk_metrics}")
    state['risk_metrics'] = risk_metrics
    return state

def execute_trades(state: TradingState) -> TradingState:
    """Execution node - execute trades based on signals and risk assessment"""
    symbol = state['symbol']
    signals = state['signals']
    risk_metrics = state['risk_metrics']
    current_positions = state['positions']
    
    if not signals or 'trade_decision' not in signals:
        logger.logger.warning("No trading decision available")
        return state
    
    decision = signals['trade_decision']
    price = signals['current_price']
    
    # Get current position for this symbol
    current_position = current_positions.get(symbol, {'quantity': 0, 'avg_price': 0})
    current_quantity = current_position.get('quantity', 0)
    
    # Determine quantity based on decision
    if decision == "BUY":
        target_quantity = risk_metrics['max_position_size']
        quantity = max(0, target_quantity - current_quantity)  # Only buy what we don't have
        action = "BUY"
    elif decision == "SELL":
        if current_quantity > 0:
            quantity = current_quantity  # Sell all we have
            action = "SELL"
        else:
            # No position to sell
            logger.logger.info(f"No {symbol} position to sell")
            return state
    else:  # HOLD
        logger.logger.info(f"Holding {symbol} position")
        return state
    
    if quantity == 0:
        logger.logger.info(f"No trade to execute for {symbol}")
        return state
    
    # Execute trade
    trade = logger.log_trade(action, symbol, quantity, price)
    
    # Update positions
    new_positions = state['positions'].copy()
    
    if action == "BUY":
        # Update average price and quantity
        total_quantity = current_quantity + quantity
        total_cost = (current_quantity * current_position.get('avg_price', 0)) + (quantity * price)
        new_avg_price = total_cost / total_quantity if total_quantity > 0 else 0
        
        new_positions[symbol] = {
            'quantity': total_quantity,
            'avg_price': new_avg_price,
            'stop_loss': risk_metrics['stop_loss'],
            'take_profit': risk_metrics['take_profit']
        }
    else:  # SELL
        new_positions.pop(symbol, None)  # Remove the position completely
    
    # Update portfolio value
    trade_value = quantity * price
    new_value = state['portfolio_value'] - trade_value if action == "BUY" else state['portfolio_value'] + trade_value
    
    # Update state
    state['positions'] = new_positions
    state['portfolio_value'] = new_value
    state['trades'] = state['trades'] + [trade]
    
    return state

def router(state: TradingState) -> Literal["end", "monitor"]:
    """Route to either monitor or end based on whether any trades were executed"""
    if state['signals'].get('trade_decision', 'HOLD') != 'HOLD':
        return "monitor"
    return "end"

def monitor_positions(state: TradingState) -> TradingState:
    """Monitor existing positions for stop loss/take profit conditions"""
    positions = state['positions']
    market_data = state['market_data']
    
    if market_data.empty or not positions:
        return state
    
    current_price = market_data['Close'].iloc[-1]
    symbol = state['symbol']
    
    if symbol in positions:
        position = positions[symbol]
        stop_loss = position.get('stop_loss', 0)
        take_profit = position.get('take_profit', 0)
        
        # Check stop loss
        if current_price <= stop_loss and stop_loss > 0:
            logger.logger.info(f"Stop loss triggered for {symbol} at {current_price}")
            quantity = position['quantity']
            trade = logger.log_trade("STOP_LOSS", symbol, quantity, current_price)
            
            # Update portfolio and positions
            new_positions = positions.copy()
            new_positions.pop(symbol, None)
            
            state['positions'] = new_positions
            state['portfolio_value'] += quantity * current_price
            state['trades'].append(trade)
        
        # Check take profit
        elif current_price >= take_profit and take_profit > 0:
            logger.logger.info(f"Take profit triggered for {symbol} at {current_price}")
            quantity = position['quantity']
            trade = logger.log_trade("TAKE_PROFIT", symbol, quantity, current_price)
            
            # Update portfolio and positions
            new_positions = positions.copy()
            new_positions.pop(symbol, None)
            
            state['positions'] = new_positions
            state['portfolio_value'] += quantity * current_price
            state['trades'].append(trade)
    
    return state

def create_trading_agent():
    """Create and configure the trading agent workflow"""
    builder = StateGraph(TradingState)
    
    # Add nodes
    builder.add_node("market_analysis", analyze_market)
    builder.add_node("generate_trading_decision", generate_trading_decision)
    builder.add_node("risk_assessment", risk_assessment)
    builder.add_node("execute_trades", execute_trades)
    builder.add_node("monitor_positions", monitor_positions)
    
    # Add edges and conditional routing
    builder.add_edge("market_analysis", "generate_trading_decision")
    builder.add_edge("generate_trading_decision", "risk_assessment")
    builder.add_edge("risk_assessment", "execute_trades")
    builder.add_edge("execute_trades", router)
    builder.add_conditional_edges(
        "execute_trades",
        router,
        {
            "monitor": "monitor_positions",
            "end": END
        }
    )
    builder.add_edge("monitor_positions", END)
    
    # Set entry point
    builder.set_entry_point("market_analysis")
    return builder.compile()

trading_agent = create_trading_agent()