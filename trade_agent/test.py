# test/test_agent.py
import pytest
from agent import trading_agent
from tools import TradingTools

def test_trading_cycle():
    initial_state = {
        'symbol': 'AAPL',
        'portfolio_value': 1_000_000,
        'positions': {},
        'market_data': TradingTools.get_market_data('AAPL'),
        'risk_metrics': {},
        'signals': {},
        'trades': []
    }
    
    result = trading_agent.invoke(initial_state)
    
    assert 'positions' in result
    assert 'portfolio_value' in result
    assert len(result['trades']) == 1
    assert result['positions']['AAPL'] > 0