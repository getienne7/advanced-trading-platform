"""Arbitrage Detection Engine
Detects and analyzes arbitrage opportunities across multiple exchanges.
"""
import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import structlog
import numpy as np
from collections import defaultdict, deque

try:
    from exchange_abstraction import ExchangeManager, OrderBook, Ticker
except ImportError:
    # For testing purposes, create mock classes
    class ExchangeManager:
        pass
    class OrderBook:
        pass
    class Ticker:
        pass

logger = structlog.get_logger("arbitrage-engine")


class ArbitrageType(str, Enum):
    """Types of arbitrage opportunities."""
    SIMPLE = "simple"  # Buy on one exchange, sell on another
    TRIANGULAR = "triangular"  # Three-way arbitrage within same exchange
    CROSS_EXCHANGE_TRIANGULAR = "cross_exchange_triangular"  # Triangular across exchanges
    FUNDING_RATE = "funding_rate"  # Funding rate arbitrage
    STATISTICAL = "statistical"  # Statistical arbitrage based on price divergence


@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity."""
    type: ArbitrageType
    symbol: str
    profit_pct: Decimal
    profit_absolute: Decimal
    confidence_score: Decimal
    timestamp: datetime
    expires_at: datetime
    min_capital: Decimal
    max_capital: Decimal
    
    # Simple arbitrage fields
    buy_exchange: Optional[str] = None
    sell_exchange: Optional[str] = None
    buy_price: Optional[Decimal] = None
    sell_price: Optional[Decimal] = None
    
    # Triangular arbitrage fields
    path: Optional[List[str]] = None  # Trading path
    exchanges: Optional[List[str]] = None  # Exchanges involved
    prices: Optional[List[Decimal]] = None  # Prices at each step
    
    # Additional metadata
    liquidity_score: Decimal = Decimal('0')
    risk_score: Decimal = Decimal('0')
    execution_complexity: int = 1
    estimated_execution_time: float = 0.0
    fees_estimate: Decimal = Decimal('0')
    slippage_estimate: Decimal = Decimal('0')
    
    @property
    def net_profit_pct(self) -> Decimal:
        """Calculate net profit after fees and slippage."""
        return self.profit_pct - self.fees_estimate - self.slippage_estimate
    
    @property
    def is_profitable(self) -> bool:
        """Check if opportunity is still profitable after costs."""
        return self.net_profit_pct > Decimal('0.1')  # Minimum 0.1% profit
    
    @property
    def is_expired(self) -> bool:
        """Check if opportunity has expired."""
        return datetime.now() > self.expires_at


@dataclass
class MarketData:
    """Market data for arbitrage analysis."""
    exchange: str
    symbol: str
    ticker: Ticker
    order_book: OrderBook
    timestamp: datetime
    
    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        return self.ticker.ask_price - self.ticker.bid_price if self.ticker.ask_price and self.ticker.bid_price else Decimal('0')
    
    @property
    def spread_pct(self) -> Decimal:
        """Calculate spread as percentage of mid price."""
        if self.ticker.bid_price and self.ticker.ask_price:
            mid_price = (self.ticker.bid_price + self.ticker.ask_price) / 2
            return (self.spread / mid_price) * 100 if mid_price > 0 else Decimal('0')
        return Decimal('0')


class PriceHistory:
    """Maintains price history for statistical analysis."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.prices: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=max_size)))
        self.timestamps: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=max_size)))
    
    def add_price(self, exchange: str, symbol: str, price: Decimal, timestamp: datetime):
        """Add a price point to history."""
        self.prices[exchange][symbol].append(float(price))
        self.timestamps[exchange][symbol].append(timestamp)
    
    def get_price_series(self, exchange: str, symbol: str, lookback_minutes: int = 60) -> List[float]:
        """Get price series for the last N minutes."""
        if exchange not in self.prices or symbol not in self.prices[exchange]:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
        prices = []
        timestamps = list(self.timestamps[exchange][symbol])
        price_values = list(self.prices[exchange][symbol])
        
        for i, ts in enumerate(timestamps):
            if ts >= cutoff_time:
                prices.append(price_values[i])
        
        return prices
    
    def calculate_volatility(self, exchange: str, symbol: str, lookback_minutes: int = 60) -> float:
        """Calculate price volatility."""
        prices = self.get_price_series(exchange, symbol, lookback_minutes)
        if len(prices) < 2:
            return 0.0
        
        returns = np.diff(np.log(prices))
        return float(np.std(returns) * np.sqrt(len(returns)))
    
    def calculate_correlation(self, exchange1: str, exchange2: str, symbol: str, lookback_minutes: int = 60) -> float:
        """Calculate price correlation between exchanges."""
        prices1 = self.get_price_series(exchange1, symbol, lookback_minutes)
        prices2 = self.get_price_series(exchange2, symbol, lookback_minutes)
        
        if len(prices1) < 10 or len(prices2) < 10:
            return 0.0
        
        # Align the series by taking the minimum length
        min_len = min(len(prices1), len(prices2))
        prices1 = prices1[-min_len:]
        prices2 = prices2[-min_len:]
        
        correlation = np.corrcoef(prices1, prices2)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0


class ArbitrageDetectionEngine:
    """Main arbitrage detection engine."""
    
    def __init__(self, exchange_manager: ExchangeManager):
        self.exchange_manager = exchange_manager
        self.price_history = PriceHistory()
        self.opportunities: List[ArbitrageOpportunity] = []
        self.market_data: Dict[str, Dict[str, MarketData]] = {}  # {exchange: {symbol: data}}
        
        # Configuration
        self.min_profit_pct = Decimal('0.5')  # Minimum 0.5% profit
        self.max_opportunity_age = timedelta(seconds=30)  # Opportunities expire after 30 seconds
        self.update_interval = 1.0  # Update every second
        self.max_opportunities = 100  # Maximum opportunities to track
        
        # Fee estimates (exchange-specific)
        self.exchange_fees = {
            'binance': {'maker': Decimal('0.1'), 'taker': Decimal('0.1')},
            'coinbase': {'maker': Decimal('0.5'), 'taker': Decimal('0.5')},
            'kraken': {'maker': Decimal('0.16'), 'taker': Decimal('0.26')}
        }
        
        # Running flag
        self.running = False
        self.update_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the arbitrage detection engine."""
        if self.running:
            return
        
        self.running = True
        self.update_task = asyncio.create_task(self._update_loop())
        logger.info("Arbitrage detection engine started")
    
    async def stop(self):
        """Stop the arbitrage detection engine."""
        self.running = False
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        logger.info("Arbitrage detection engine stopped")
    
    async def _update_loop(self):
        """Main update loop for detecting arbitrage opportunities."""
        while self.running:
            try:
                start_time = time.time()
                
                # Update market data
                await self._update_market_data()
                
                # Detect opportunities
                await self._detect_opportunities()
                
                # Clean up expired opportunities
                self._cleanup_expired_opportunities()
                
                # Calculate loop time and sleep
                loop_time = time.time() - start_time
                sleep_time = max(0, self.update_interval - loop_time)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    logger.warning(f"Arbitrage detection loop took {loop_time:.3f}s, longer than interval {self.update_interval}s")
                
            except Exception as e:
                logger.error(f"Error in arbitrage detection loop: {str(e)}")
                await asyncio.sleep(1)  # Brief pause before retrying
    
    async def _update_market_data(self):
        """Update market data from all exchanges."""
        active_exchanges = self.exchange_manager.get_active_exchanges()
        if len(active_exchanges) < 2:
            return  # Need at least 2 exchanges for arbitrage
        
        # Common trading pairs to monitor
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
        
        for symbol in symbols:
            # Get tickers from all exchanges
            tickers = await self.exchange_manager.get_all_tickers(symbol)
            order_books = await self.exchange_manager.get_all_order_books(symbol)
            
            for exchange_name in active_exchanges:
                if exchange_name in tickers and exchange_name in order_books:
                    ticker = tickers[exchange_name]
                    order_book = order_books[exchange_name]
                    
                    # Store market data
                    if exchange_name not in self.market_data:
                        self.market_data[exchange_name] = {}
                    
                    self.market_data[exchange_name][symbol] = MarketData(
                        exchange=exchange_name,
                        symbol=symbol,
                        ticker=ticker,
                        order_book=order_book,
                        timestamp=datetime.now()
                    )
                    
                    # Update price history
                    if ticker.close_price:
                        self.price_history.add_price(
                            exchange_name, symbol, ticker.close_price, datetime.now()
                        )
    
    async def _detect_opportunities(self):
        """Detect all types of arbitrage opportunities."""
        # Simple arbitrage
        await self._detect_simple_arbitrage()
        
        # Triangular arbitrage
        await self._detect_triangular_arbitrage()
        
        # Statistical arbitrage
        await self._detect_statistical_arbitrage()
        
        # Funding rate arbitrage (placeholder for futures)
        # await self._detect_funding_rate_arbitrage()
    
    async def _detect_simple_arbitrage(self):
        """Detect simple arbitrage opportunities between exchanges."""
        active_exchanges = self.exchange_manager.get_active_exchanges()
        
        for symbol in self.market_data.get(active_exchanges[0], {}).keys():
            exchange_data = []
            
            # Collect data from all exchanges
            for exchange in active_exchanges:
                if exchange in self.market_data and symbol in self.market_data[exchange]:
                    data = self.market_data[exchange][symbol]
                    if data.ticker.bid_price and data.ticker.ask_price:
                        exchange_data.append(data)
            
            if len(exchange_data) < 2:
                continue
            
            # Find best bid and ask across exchanges
            best_bid_data = max(exchange_data, key=lambda x: x.ticker.bid_price)
            best_ask_data = min(exchange_data, key=lambda x: x.ticker.ask_price)
            
            if best_bid_data.exchange == best_ask_data.exchange:
                continue  # Same exchange, no arbitrage
            
            # Calculate profit
            buy_price = best_ask_data.ticker.ask_price
            sell_price = best_bid_data.ticker.bid_price
            profit_absolute = sell_price - buy_price
            profit_pct = (profit_absolute / buy_price) * 100
            
            if profit_pct >= self.min_profit_pct:
                # Calculate additional metrics
                liquidity_score = self._calculate_liquidity_score(best_ask_data, best_bid_data)
                risk_score = self._calculate_risk_score(best_ask_data, best_bid_data)
                fees_estimate = self._estimate_fees(best_ask_data.exchange, best_bid_data.exchange)
                
                opportunity = ArbitrageOpportunity(
                    type=ArbitrageType.SIMPLE,
                    symbol=symbol,
                    profit_pct=profit_pct,
                    profit_absolute=profit_absolute,
                    confidence_score=self._calculate_confidence_score(profit_pct, liquidity_score, risk_score),
                    timestamp=datetime.now(),
                    expires_at=datetime.now() + self.max_opportunity_age,
                    min_capital=Decimal('100'),  # Minimum $100
                    max_capital=self._calculate_max_capital(best_ask_data, best_bid_data),
                    buy_exchange=best_ask_data.exchange,
                    sell_exchange=best_bid_data.exchange,
                    buy_price=buy_price,
                    sell_price=sell_price,
                    liquidity_score=liquidity_score,
                    risk_score=risk_score,
                    execution_complexity=2,  # Buy + Sell
                    estimated_execution_time=2.0,  # 2 seconds
                    fees_estimate=fees_estimate,
                    slippage_estimate=self._estimate_slippage(best_ask_data, best_bid_data)
                )
                
                self._add_opportunity(opportunity)
    
    async def _detect_triangular_arbitrage(self):
        """Detect triangular arbitrage opportunities."""
        active_exchanges = self.exchange_manager.get_active_exchanges()
        
        # Common triangular paths
        triangular_paths = [
            ['BTCUSDT', 'ETHBTC', 'ETHUSDT'],
            ['ETHUSDT', 'ADAETH', 'ADAUSDT'],
            ['BTCUSDT', 'ADABTC', 'ADAUSDT']
        ]
        
        for exchange in active_exchanges:
            if exchange not in self.market_data:
                continue
            
            for path in triangular_paths:
                # Check if all symbols in path are available
                if not all(symbol in self.market_data[exchange] for symbol in path):
                    continue
                
                # Calculate triangular arbitrage
                opportunity = self._calculate_triangular_arbitrage(exchange, path)
                if opportunity:
                    self._add_opportunity(opportunity)
    
    def _calculate_triangular_arbitrage(self, exchange: str, path: List[str]) -> Optional[ArbitrageOpportunity]:
        """Calculate triangular arbitrage for a given path."""
        try:
            # Get market data for all symbols in path
            market_data = [self.market_data[exchange][symbol] for symbol in path]
            
            # Calculate forward path (A -> B -> C -> A)
            forward_result = Decimal('1')
            forward_prices = []
            
            # A -> B (buy B with A)
            price_ab = market_data[0].ticker.ask_price  # Buy B, pay in A
            if not price_ab:
                return None
            forward_result /= price_ab
            forward_prices.append(price_ab)
            
            # B -> C (buy C with B)  
            price_bc = market_data[1].ticker.ask_price  # Buy C, pay in B
            if not price_bc:
                return None
            forward_result /= price_bc
            forward_prices.append(price_bc)
            
            # C -> A (sell C for A)
            price_ca = market_data[2].ticker.bid_price  # Sell C, get A
            if not price_ca:
                return None
            forward_result *= price_ca
            forward_prices.append(price_ca)
            
            # Calculate profit
            profit_absolute = forward_result - Decimal('1')
            profit_pct = profit_absolute * 100
            
            if profit_pct >= self.min_profit_pct:
                # Calculate metrics
                liquidity_score = min(data.order_book.best_bid[1] if data.order_book.best_bid else Decimal('0') 
                                    for data in market_data)
                risk_score = Decimal('5')  # Higher risk for triangular arbitrage
                fees_estimate = self._estimate_fees(exchange, exchange) * 3  # Three trades
                
                return ArbitrageOpportunity(
                    type=ArbitrageType.TRIANGULAR,
                    symbol=path[0],  # Base symbol
                    profit_pct=profit_pct,
                    profit_absolute=profit_absolute,
                    confidence_score=self._calculate_confidence_score(profit_pct, liquidity_score, risk_score),
                    timestamp=datetime.now(),
                    expires_at=datetime.now() + timedelta(seconds=15),  # Shorter expiry for triangular
                    min_capital=Decimal('500'),  # Higher minimum for triangular
                    max_capital=liquidity_score * Decimal('100'),
                    path=path,
                    exchanges=[exchange] * 3,
                    prices=forward_prices,
                    liquidity_score=liquidity_score,
                    risk_score=risk_score,
                    execution_complexity=3,  # Three trades
                    estimated_execution_time=5.0,  # 5 seconds
                    fees_estimate=fees_estimate,
                    slippage_estimate=Decimal('0.2')  # Higher slippage for multiple trades
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating triangular arbitrage: {str(e)}")
            return None
    
    async def _detect_statistical_arbitrage(self):
        """Detect statistical arbitrage opportunities based on price divergence."""
        active_exchanges = self.exchange_manager.get_active_exchanges()
        
        if len(active_exchanges) < 2:
            return
        
        for symbol in self.market_data.get(active_exchanges[0], {}).keys():
            # Calculate price divergence between exchanges
            prices = {}
            for exchange in active_exchanges:
                if exchange in self.market_data and symbol in self.market_data[exchange]:
                    ticker = self.market_data[exchange][symbol]
                    if ticker.ticker.close_price:
                        prices[exchange] = float(ticker.ticker.close_price)
            
            if len(prices) < 2:
                continue
            
            # Calculate mean and standard deviation
            price_values = list(prices.values())
            mean_price = np.mean(price_values)
            std_price = np.std(price_values)
            
            if std_price == 0:
                continue
            
            # Find exchanges with significant price divergence
            for exchange, price in prices.items():
                z_score = abs(price - mean_price) / std_price
                
                if z_score > 2.0:  # 2 standard deviations
                    # Calculate correlation with other exchanges
                    correlations = []
                    for other_exchange in active_exchanges:
                        if other_exchange != exchange:
                            corr = self.price_history.calculate_correlation(
                                exchange, other_exchange, symbol, 60
                            )
                            correlations.append(corr)
                    
                    avg_correlation = np.mean(correlations) if correlations else 0
                    
                    if avg_correlation > 0.8:  # High correlation suggests mean reversion
                        # Determine direction
                        is_overpriced = price > mean_price
                        
                        profit_pct = Decimal(str(abs(price - mean_price) / mean_price * 100))
                        
                        if profit_pct >= self.min_profit_pct:
                            opportunity = ArbitrageOpportunity(
                                type=ArbitrageType.STATISTICAL,
                                symbol=symbol,
                                profit_pct=profit_pct,
                                profit_absolute=Decimal(str(abs(price - mean_price))),
                                confidence_score=Decimal(str(avg_correlation * 10)),
                                timestamp=datetime.now(),
                                expires_at=datetime.now() + timedelta(minutes=5),  # Longer expiry
                                min_capital=Decimal('1000'),  # Higher minimum for stat arb
                                max_capital=Decimal('10000'),
                                buy_exchange=exchange if not is_overpriced else None,
                                sell_exchange=exchange if is_overpriced else None,
                                buy_price=Decimal(str(price)) if not is_overpriced else None,
                                sell_price=Decimal(str(price)) if is_overpriced else None,
                                liquidity_score=Decimal('5'),
                                risk_score=Decimal('3'),  # Medium risk
                                execution_complexity=1,
                                estimated_execution_time=1.0,
                                fees_estimate=self._estimate_fees(exchange, exchange),
                                slippage_estimate=Decimal('0.1')
                            )
                            
                            self._add_opportunity(opportunity)
    
    def _calculate_liquidity_score(self, ask_data: MarketData, bid_data: MarketData) -> Decimal:
        """Calculate liquidity score based on order book depth."""
        ask_liquidity = sum(qty for _, qty in ask_data.order_book.asks[:5]) if ask_data.order_book.asks else Decimal('0')
        bid_liquidity = sum(qty for _, qty in bid_data.order_book.bids[:5]) if bid_data.order_book.bids else Decimal('0')
        
        return min(ask_liquidity, bid_liquidity)
    
    def _calculate_risk_score(self, ask_data: MarketData, bid_data: MarketData) -> Decimal:
        """Calculate risk score based on spreads and volatility."""
        ask_spread_pct = ask_data.spread_pct
        bid_spread_pct = bid_data.spread_pct
        
        # Higher spreads = higher risk
        avg_spread = (ask_spread_pct + bid_spread_pct) / 2
        
        # Add volatility component
        ask_volatility = self.price_history.calculate_volatility(ask_data.exchange, ask_data.symbol)
        bid_volatility = self.price_history.calculate_volatility(bid_data.exchange, bid_data.symbol)
        avg_volatility = (ask_volatility + bid_volatility) / 2
        
        return Decimal(str(avg_spread + avg_volatility * 100))
    
    def _calculate_confidence_score(self, profit_pct: Decimal, liquidity_score: Decimal, risk_score: Decimal) -> Decimal:
        """Calculate confidence score for the opportunity."""
        # Higher profit and liquidity = higher confidence
        # Higher risk = lower confidence
        base_score = profit_pct * 2 + liquidity_score / 10
        risk_penalty = risk_score / 2
        
        confidence = max(Decimal('0'), min(Decimal('10'), base_score - risk_penalty))
        return confidence
    
    def _estimate_fees(self, buy_exchange: str, sell_exchange: str) -> Decimal:
        """Estimate trading fees for the arbitrage."""
        buy_fee = self.exchange_fees.get(buy_exchange, {}).get('taker', Decimal('0.1'))
        sell_fee = self.exchange_fees.get(sell_exchange, {}).get('taker', Decimal('0.1'))
        
        return buy_fee + sell_fee
    
    def _estimate_slippage(self, ask_data: MarketData, bid_data: MarketData) -> Decimal:
        """Estimate slippage based on order book depth."""
        # Simple slippage model based on spreads
        ask_spread_pct = ask_data.spread_pct
        bid_spread_pct = bid_data.spread_pct
        
        # Assume slippage is proportional to spread
        avg_spread = (ask_spread_pct + bid_spread_pct) / 2
        return avg_spread * Decimal('0.5')  # 50% of spread as slippage estimate
    
    def _calculate_max_capital(self, ask_data: MarketData, bid_data: MarketData) -> Decimal:
        """Calculate maximum capital that can be deployed."""
        ask_liquidity = sum(qty for _, qty in ask_data.order_book.asks[:10]) if ask_data.order_book.asks else Decimal('0')
        bid_liquidity = sum(qty for _, qty in bid_data.order_book.bids[:10]) if bid_data.order_book.bids else Decimal('0')
        
        # Use 50% of available liquidity to avoid excessive slippage
        max_qty = min(ask_liquidity, bid_liquidity) * Decimal('0.5')
        
        # Convert to capital (assuming average price)
        if ask_data.ticker.ask_price:
            max_capital = max_qty * ask_data.ticker.ask_price
            return min(max_capital, Decimal('100000'))  # Cap at $100k
        
        return Decimal('1000')  # Default
    
    def _add_opportunity(self, opportunity: ArbitrageOpportunity):
        """Add opportunity to the list, avoiding duplicates."""
        # Remove similar existing opportunities
        self.opportunities = [
            opp for opp in self.opportunities 
            if not self._is_similar_opportunity(opp, opportunity)
        ]
        
        # Add new opportunity
        self.opportunities.append(opportunity)
        
        # Keep only the best opportunities
        self.opportunities.sort(key=lambda x: x.net_profit_pct, reverse=True)
        self.opportunities = self.opportunities[:self.max_opportunities]
        
        logger.info(f"New {opportunity.type} arbitrage opportunity: {opportunity.symbol} "
                   f"{opportunity.profit_pct:.2f}% profit on {opportunity.buy_exchange}->{opportunity.sell_exchange}")
    
    def _is_similar_opportunity(self, opp1: ArbitrageOpportunity, opp2: ArbitrageOpportunity) -> bool:
        """Check if two opportunities are similar."""
        return (opp1.type == opp2.type and 
                opp1.symbol == opp2.symbol and
                opp1.buy_exchange == opp2.buy_exchange and
                opp1.sell_exchange == opp2.sell_exchange)
    
    def _cleanup_expired_opportunities(self):
        """Remove expired opportunities."""
        before_count = len(self.opportunities)
        self.opportunities = [opp for opp in self.opportunities if not opp.is_expired]
        
        removed_count = before_count - len(self.opportunities)
        if removed_count > 0:
            logger.debug(f"Removed {removed_count} expired arbitrage opportunities")
    
    def get_opportunities(self, 
                         symbol: Optional[str] = None,
                         min_profit_pct: Optional[Decimal] = None,
                         max_risk_score: Optional[Decimal] = None,
                         arbitrage_type: Optional[ArbitrageType] = None) -> List[ArbitrageOpportunity]:
        """Get filtered arbitrage opportunities."""
        opportunities = self.opportunities.copy()
        
        # Apply filters
        if symbol:
            opportunities = [opp for opp in opportunities if opp.symbol == symbol]
        
        if min_profit_pct:
            opportunities = [opp for opp in opportunities if opp.net_profit_pct >= min_profit_pct]
        
        if max_risk_score:
            opportunities = [opp for opp in opportunities if opp.risk_score <= max_risk_score]
        
        if arbitrage_type:
            opportunities = [opp for opp in opportunities if opp.type == arbitrage_type]
        
        # Filter only profitable opportunities
        opportunities = [opp for opp in opportunities if opp.is_profitable]
        
        return sorted(opportunities, key=lambda x: x.net_profit_pct, reverse=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get arbitrage engine statistics."""
        total_opportunities = len(self.opportunities)
        profitable_opportunities = len([opp for opp in self.opportunities if opp.is_profitable])
        
        if total_opportunities > 0:
            avg_profit = sum(opp.profit_pct for opp in self.opportunities) / total_opportunities
            max_profit = max(opp.profit_pct for opp in self.opportunities)
            
            type_distribution = {}
            for opp_type in ArbitrageType:
                count = len([opp for opp in self.opportunities if opp.type == opp_type])
                type_distribution[opp_type.value] = count
        else:
            avg_profit = Decimal('0')
            max_profit = Decimal('0')
            type_distribution = {}
        
        return {
            'total_opportunities': total_opportunities,
            'profitable_opportunities': profitable_opportunities,
            'avg_profit_pct': float(avg_profit),
            'max_profit_pct': float(max_profit),
            'type_distribution': type_distribution,
            'active_exchanges': len(self.exchange_manager.get_active_exchanges()),
            'last_update': datetime.now().isoformat()
        }