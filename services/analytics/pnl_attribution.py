"""
P&L Attribution Engine - Real-time P&L calculation and attribution analysis
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PnLAttributionEngine:
    """
    Engine for calculating and attributing P&L across strategies, assets, and time periods
    """
    
    def __init__(self):
        self.price_cache = {}
        self.attribution_cache = {}
    
    async def calculate_real_time_pnl(
        self,
        positions: List[Dict[str, Any]],
        recent_trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate real-time P&L from current positions and recent trades"""
        try:
            total_unrealized_pnl = Decimal('0')
            total_realized_pnl = Decimal('0')
            
            # Calculate unrealized P&L from positions
            position_pnl = {}
            for position in positions:
                symbol = position['symbol']
                current_price = await self._get_current_price(symbol, position['exchange'])
                
                if current_price:
                    entry_price = Decimal(str(position['entry_price']))
                    quantity = Decimal(str(position['quantity']))
                    side = position['side']
                    
                    # Calculate unrealized P&L
                    if side == 'LONG':
                        unrealized = (current_price - entry_price) * quantity
                    else:  # SHORT
                        unrealized = (entry_price - current_price) * quantity
                    
                    position_pnl[f"{symbol}_{position['id']}"] = {
                        'symbol': symbol,
                        'side': side,
                        'quantity': float(quantity),
                        'entry_price': float(entry_price),
                        'current_price': float(current_price),
                        'unrealized_pnl': float(unrealized),
                        'strategy_id': position.get('strategy_id')
                    }
                    
                    total_unrealized_pnl += unrealized
            
            # Calculate realized P&L from recent trades
            trade_pnl = {}
            for trade in recent_trades:
                trade_id = trade['id']
                pnl = Decimal(str(trade.get('pnl', 0)))
                
                trade_pnl[trade_id] = {
                    'symbol': trade['symbol'],
                    'side': trade['side'],
                    'quantity': float(trade['quantity']),
                    'price': float(trade['price']),
                    'pnl': float(pnl),
                    'fees': float(trade.get('fees', 0)),
                    'strategy_id': trade.get('strategy_id'),
                    'executed_at': trade['executed_at']
                }
                
                total_realized_pnl += pnl
            
            # Calculate attribution by strategy
            strategy_attribution = await self._calculate_strategy_attribution(
                position_pnl, trade_pnl
            )
            
            # Calculate attribution by asset
            asset_attribution = await self._calculate_asset_attribution(
                position_pnl, trade_pnl
            )
            
            return {
                'total_unrealized_pnl': float(total_unrealized_pnl),
                'total_realized_pnl': float(total_realized_pnl),
                'total_pnl': float(total_unrealized_pnl + total_realized_pnl),
                'position_pnl': position_pnl,
                'trade_pnl': trade_pnl,
                'strategy_attribution': strategy_attribution,
                'asset_attribution': asset_attribution,
                'calculated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating real-time P&L: {e}")
            raise
    
    async def calculate_attribution(
        self,
        trades: List[Dict[str, Any]],
        positions: List[Dict[str, Any]],
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Calculate detailed P&L attribution for a time period"""
        try:
            # Convert to DataFrame for easier analysis
            trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
            positions_df = pd.DataFrame(positions) if positions else pd.DataFrame()
            
            attribution_data = {
                'period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'duration_hours': (end_time - start_time).total_seconds() / 3600
                },
                'summary': {},
                'by_strategy': {},
                'by_asset': {},
                'by_time': {},
                'risk_attribution': {}
            }
            
            if not trades_df.empty:
                # Overall summary
                total_pnl = trades_df['pnl'].sum()
                total_fees = trades_df['fees'].sum()
                net_pnl = total_pnl - total_fees
                
                attribution_data['summary'] = {
                    'total_pnl': float(total_pnl),
                    'total_fees': float(total_fees),
                    'net_pnl': float(net_pnl),
                    'trade_count': len(trades_df),
                    'win_rate': float((trades_df['pnl'] > 0).mean()),
                    'avg_trade_pnl': float(trades_df['pnl'].mean()),
                    'best_trade': float(trades_df['pnl'].max()),
                    'worst_trade': float(trades_df['pnl'].min())
                }
                
                # Attribution by strategy
                if 'strategy_id' in trades_df.columns:
                    strategy_groups = trades_df.groupby('strategy_id')
                    for strategy_id, group in strategy_groups:
                        if pd.notna(strategy_id):
                            attribution_data['by_strategy'][str(strategy_id)] = {
                                'pnl': float(group['pnl'].sum()),
                                'fees': float(group['fees'].sum()),
                                'trade_count': len(group),
                                'win_rate': float((group['pnl'] > 0).mean()),
                                'avg_trade_pnl': float(group['pnl'].mean()),
                                'contribution_pct': float((group['pnl'].sum() / total_pnl) * 100) if total_pnl != 0 else 0
                            }
                
                # Attribution by asset
                if 'symbol' in trades_df.columns:
                    asset_groups = trades_df.groupby('symbol')
                    for symbol, group in asset_groups:
                        attribution_data['by_asset'][symbol] = {
                            'pnl': float(group['pnl'].sum()),
                            'fees': float(group['fees'].sum()),
                            'trade_count': len(group),
                            'win_rate': float((group['pnl'] > 0).mean()),
                            'avg_trade_pnl': float(group['pnl'].mean()),
                            'contribution_pct': float((group['pnl'].sum() / total_pnl) * 100) if total_pnl != 0 else 0
                        }
                
                # Attribution by time (hourly buckets)
                if 'executed_at' in trades_df.columns:
                    trades_df['executed_at'] = pd.to_datetime(trades_df['executed_at'])
                    trades_df['hour'] = trades_df['executed_at'].dt.floor('H')
                    time_groups = trades_df.groupby('hour')
                    
                    for hour, group in time_groups:
                        attribution_data['by_time'][hour.isoformat()] = {
                            'pnl': float(group['pnl'].sum()),
                            'trade_count': len(group),
                            'avg_trade_pnl': float(group['pnl'].mean())
                        }
            
            # Risk attribution (if positions data available)
            if not positions_df.empty:
                risk_metrics = await self._calculate_risk_attribution(positions_df)
                attribution_data['risk_attribution'] = risk_metrics
            
            return attribution_data
            
        except Exception as e:
            logger.error(f"Error calculating P&L attribution: {e}")
            raise
    
    async def _calculate_strategy_attribution(
        self,
        position_pnl: Dict[str, Any],
        trade_pnl: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate P&L attribution by strategy"""
        strategy_attribution = {}
        
        # Aggregate by strategy from positions
        for pos_id, pos_data in position_pnl.items():
            strategy_id = pos_data.get('strategy_id', 'unassigned')
            if strategy_id not in strategy_attribution:
                strategy_attribution[strategy_id] = {
                    'unrealized_pnl': 0,
                    'realized_pnl': 0,
                    'total_pnl': 0,
                    'position_count': 0,
                    'trade_count': 0
                }
            
            strategy_attribution[strategy_id]['unrealized_pnl'] += pos_data['unrealized_pnl']
            strategy_attribution[strategy_id]['position_count'] += 1
        
        # Aggregate by strategy from trades
        for trade_id, trade_data in trade_pnl.items():
            strategy_id = trade_data.get('strategy_id', 'unassigned')
            if strategy_id not in strategy_attribution:
                strategy_attribution[strategy_id] = {
                    'unrealized_pnl': 0,
                    'realized_pnl': 0,
                    'total_pnl': 0,
                    'position_count': 0,
                    'trade_count': 0
                }
            
            strategy_attribution[strategy_id]['realized_pnl'] += trade_data['pnl']
            strategy_attribution[strategy_id]['trade_count'] += 1
        
        # Calculate totals
        for strategy_id in strategy_attribution:
            strategy_data = strategy_attribution[strategy_id]
            strategy_data['total_pnl'] = (
                strategy_data['unrealized_pnl'] + strategy_data['realized_pnl']
            )
        
        return strategy_attribution
    
    async def _calculate_asset_attribution(
        self,
        position_pnl: Dict[str, Any],
        trade_pnl: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate P&L attribution by asset"""
        asset_attribution = {}
        
        # Aggregate by asset from positions
        for pos_id, pos_data in position_pnl.items():
            symbol = pos_data['symbol']
            if symbol not in asset_attribution:
                asset_attribution[symbol] = {
                    'unrealized_pnl': 0,
                    'realized_pnl': 0,
                    'total_pnl': 0,
                    'position_count': 0,
                    'trade_count': 0
                }
            
            asset_attribution[symbol]['unrealized_pnl'] += pos_data['unrealized_pnl']
            asset_attribution[symbol]['position_count'] += 1
        
        # Aggregate by asset from trades
        for trade_id, trade_data in trade_pnl.items():
            symbol = trade_data['symbol']
            if symbol not in asset_attribution:
                asset_attribution[symbol] = {
                    'unrealized_pnl': 0,
                    'realized_pnl': 0,
                    'total_pnl': 0,
                    'position_count': 0,
                    'trade_count': 0
                }
            
            asset_attribution[symbol]['realized_pnl'] += trade_data['pnl']
            asset_attribution[symbol]['trade_count'] += 1
        
        # Calculate totals
        for symbol in asset_attribution:
            asset_data = asset_attribution[symbol]
            asset_data['total_pnl'] = (
                asset_data['unrealized_pnl'] + asset_data['realized_pnl']
            )
        
        return asset_attribution
    
    async def _calculate_risk_attribution(
        self,
        positions_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate risk-based P&L attribution"""
        try:
            risk_attribution = {}
            
            if 'unrealized_pnl' in positions_df.columns and 'quantity' in positions_df.columns:
                # Calculate position sizes and risk contributions
                positions_df['position_value'] = positions_df['quantity'] * positions_df['entry_price']
                total_position_value = positions_df['position_value'].sum()
                
                if total_position_value > 0:
                    positions_df['weight'] = positions_df['position_value'] / total_position_value
                    positions_df['risk_contribution'] = positions_df['weight'] * positions_df['unrealized_pnl']
                    
                    # Group by symbol for risk attribution
                    symbol_groups = positions_df.groupby('symbol')
                    for symbol, group in symbol_groups:
                        risk_attribution[symbol] = {
                            'weight': float(group['weight'].sum()),
                            'risk_contribution': float(group['risk_contribution'].sum()),
                            'position_count': len(group)
                        }
            
            return risk_attribution
            
        except Exception as e:
            logger.error(f"Error calculating risk attribution: {e}")
            return {}
    
    async def _get_current_price(self, symbol: str, exchange: str) -> Optional[Decimal]:
        """Get current market price for a symbol"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{exchange}"
            if cache_key in self.price_cache:
                price, timestamp = self.price_cache[cache_key]
                if (datetime.utcnow() - timestamp).total_seconds() < 10:  # 10 second cache
                    return price
            
            # In a real implementation, this would fetch from market data service
            # For now, return a mock price
            mock_price = Decimal('50000.00')  # Mock BTC price
            
            # Cache the price
            self.price_cache[cache_key] = (mock_price, datetime.utcnow())
            
            return mock_price
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None