"""Test suite for the Arbitrage Detection Engine."""
import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock

import sys
import os
sys.path.append(os.path.dirname(__file__))

from arbitrage_engine import (
    ArbitrageDetectionEngine, ArbitrageOpportunity, ArbitrageType,
    MarketData, PriceHistory
)
from exchange_abstraction import ExchangeManager, Ticker, OrderBook


class TestPriceHistory:
    """Test price history functionality."""
    
    def test_add_and_get_prices(self):
        """Test adding and retrieving price data."""
        history = PriceHistory(max_size=100)
        
        # Add some prices
        now = datetime.now()
        history.add_price("binance", "BTCUSDT", Decimal("50000"), now)
        history.add_price("binance", "BTCUSDT", Decimal("50100"), now + timedelta(minutes=1))
        history.add_price("binance", "BTCUSDT", Decimal("49900"), now + timedelta(minutes=2))
        
        # Get price series
        prices = history.get_price_series("binance", "BTCUSDT", lookback_minutes=5)
        assert len(prices) == 3
        assert prices == [50000.0, 50100.0, 49900.0]
    
    def test_volatility_calculation(self):
        """Test volatility calculation."""
        history = PriceHistory()
        
        # Add prices with some volatility
        now = datetime.now()
        prices = [50000, 50500, 49500, 51000, 48000, 52000]
        for i, price in enumerate(prices):
            history.add_price("binance", "BTCUSDT", Decimal(str(price)), 
                            now + timedelta(minutes=i))
        
        volatility = history.calculate_volatility("binance", "BTCUSDT", lookback_minutes=10)
        assert volatility > 0
        assert isinstance(volatility, float)
    
    def test_correlation_calculation(self):
        """Test correlation calculation between exchanges."""
        history = PriceHistory()
        
        # Add correlated prices
        now = datetime.now()
        for i in range(20):
            base_price = 50000 + i * 100
            history.add_price("binance", "BTCUSDT", Decimal(str(base_price)), 
                            now + timedelta(minutes=i))
            history.add_price("coinbase", "BTCUSDT", Decimal(str(base_price + 50)), 
                            now + timedelta(minutes=i))
        
        correlation = history.calculate_correlation("binance", "coinbase", "BTCUSDT", 30)
        assert correlation > 0.9  # Should be highly correlated


class TestArbitrageDetectionEngine:
    """Test arbitrage detection engine."""
    
    @pytest.fixture
    def mock_exchange_manager(self):
        """Create a mock exchange manager."""
        manager = Mock(spec=ExchangeManager)
        manager.get_active_exchanges.return_value = ["binance", "coinbase"]
        return manager
    
    @pytest.fixture
    def arbitrage_engine(self, mock_exchange_manager):
        """Create arbitrage detection engine."""
        return ArbitrageDetectionEngine(mock_exchange_manager)
    
    def test_initialization(self, arbitrage_engine):
        """Test engine initialization."""
        assert not arbitrage_engine.running
        assert arbitrage_engine.min_profit_pct == Decimal('0.5')
        assert len(arbitrage_engine.opportunities) == 0
    
    def test_liquidity_score_calculation(self, arbitrage_engine):
        """Test liquidity score calculation."""
        # Create mock market data
        ask_order_book = OrderBook(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            bids=[],
            asks=[(Decimal("50000"), Decimal("1")), (Decimal("50010"), Decimal("2"))]
        )
        
        bid_order_book = OrderBook(
            symbol="BTCUSDT", 
            timestamp=datetime.now(),
            bids=[(Decimal("49990"), Decimal("1.5")), (Decimal("49980"), Decimal("1"))],
            asks=[]
        )
        
        ask_ticker = Ticker(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            open_price=Decimal("49000"),
            high_price=Decimal("51000"),
            low_price=Decimal("48000"),
            close_price=Decimal("50000"),
            volume=Decimal("100"),
            quote_volume=Decimal("5000000"),
            price_change=Decimal("1000"),
            price_change_percent=Decimal("2"),
            ask_price=Decimal("50000")
        )
        
        bid_ticker = Ticker(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            open_price=Decimal("49000"),
            high_price=Decimal("51000"),
            low_price=Decimal("48000"),
            close_price=Decimal("49990"),
            volume=Decimal("100"),
            quote_volume=Decimal("4999000"),
            price_change=Decimal("990"),
            price_change_percent=Decimal("2"),
            bid_price=Decimal("49990")
        )
        
        ask_data = MarketData("binance", "BTCUSDT", ask_ticker, ask_order_book, datetime.now())
        bid_data = MarketData("coinbase", "BTCUSDT", bid_ticker, bid_order_book, datetime.now())
        
        liquidity_score = arbitrage_engine._calculate_liquidity_score(ask_data, bid_data)
        assert liquidity_score == Decimal("1.5")  # Min of ask and bid liquidity
    
    def test_risk_score_calculation(self, arbitrage_engine):
        """Test risk score calculation."""
        # Create market data with spreads
        ask_ticker = Ticker(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            open_price=Decimal("49000"),
            high_price=Decimal("51000"),
            low_price=Decimal("48000"),
            close_price=Decimal("50000"),
            volume=Decimal("100"),
            quote_volume=Decimal("5000000"),
            price_change=Decimal("1000"),
            price_change_percent=Decimal("2"),
            bid_price=Decimal("49950"),
            ask_price=Decimal("50050")
        )
        
        order_book = OrderBook(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            bids=[(Decimal("49950"), Decimal("1"))],
            asks=[(Decimal("50050"), Decimal("1"))]
        )
        
        ask_data = MarketData("binance", "BTCUSDT", ask_ticker, order_book, datetime.now())
        bid_data = MarketData("coinbase", "BTCUSDT", ask_ticker, order_book, datetime.now())
        
        risk_score = arbitrage_engine._calculate_risk_score(ask_data, bid_data)
        assert risk_score > 0
    
    def test_fee_estimation(self, arbitrage_engine):
        """Test fee estimation."""
        fees = arbitrage_engine._estimate_fees("binance", "coinbase")
        expected_fees = Decimal("0.1") + Decimal("0.5")  # Binance taker + Coinbase taker
        assert fees == expected_fees
    
    def test_opportunity_similarity_check(self, arbitrage_engine):
        """Test opportunity similarity detection."""
        opp1 = ArbitrageOpportunity(
            type=ArbitrageType.SIMPLE,
            symbol="BTCUSDT",
            profit_pct=Decimal("1.0"),
            profit_absolute=Decimal("500"),
            confidence_score=Decimal("8"),
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=30),
            min_capital=Decimal("100"),
            max_capital=Decimal("10000"),
            buy_exchange="binance",
            sell_exchange="coinbase"
        )
        
        opp2 = ArbitrageOpportunity(
            type=ArbitrageType.SIMPLE,
            symbol="BTCUSDT",
            profit_pct=Decimal("1.2"),
            profit_absolute=Decimal("600"),
            confidence_score=Decimal("8.5"),
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=30),
            min_capital=Decimal("100"),
            max_capital=Decimal("10000"),
            buy_exchange="binance",
            sell_exchange="coinbase"
        )
        
        opp3 = ArbitrageOpportunity(
            type=ArbitrageType.SIMPLE,
            symbol="ETHUSDT",
            profit_pct=Decimal("1.0"),
            profit_absolute=Decimal("30"),
            confidence_score=Decimal("7"),
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=30),
            min_capital=Decimal("100"),
            max_capital=Decimal("10000"),
            buy_exchange="binance",
            sell_exchange="coinbase"
        )
        
        # Same symbol and exchanges should be similar
        assert arbitrage_engine._is_similar_opportunity(opp1, opp2)
        
        # Different symbol should not be similar
        assert not arbitrage_engine._is_similar_opportunity(opp1, opp3)
    
    def test_opportunity_filtering(self, arbitrage_engine):
        """Test opportunity filtering."""
        # Add some test opportunities
        opportunities = [
            ArbitrageOpportunity(
                type=ArbitrageType.SIMPLE,
                symbol="BTCUSDT",
                profit_pct=Decimal("1.0"),
                profit_absolute=Decimal("500"),
                confidence_score=Decimal("8"),
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=30),
                min_capital=Decimal("100"),
                max_capital=Decimal("10000"),
                risk_score=Decimal("2")
            ),
            ArbitrageOpportunity(
                type=ArbitrageType.TRIANGULAR,
                symbol="ETHUSDT",
                profit_pct=Decimal("2.0"),
                profit_absolute=Decimal("60"),
                confidence_score=Decimal("9"),
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=30),
                min_capital=Decimal("500"),
                max_capital=Decimal("5000"),
                risk_score=Decimal("5")
            )
        ]
        
        arbitrage_engine.opportunities = opportunities
        
        # Filter by symbol
        btc_opportunities = arbitrage_engine.get_opportunities(symbol="BTCUSDT")
        assert len(btc_opportunities) == 1
        assert btc_opportunities[0].symbol == "BTCUSDT"
        
        # Filter by type
        triangular_opportunities = arbitrage_engine.get_opportunities(arbitrage_type=ArbitrageType.TRIANGULAR)
        assert len(triangular_opportunities) == 1
        assert triangular_opportunities[0].type == ArbitrageType.TRIANGULAR
        
        # Filter by risk score
        low_risk_opportunities = arbitrage_engine.get_opportunities(max_risk_score=Decimal("3"))
        assert len(low_risk_opportunities) == 1
        assert low_risk_opportunities[0].risk_score <= Decimal("3")
    
    def test_statistics_generation(self, arbitrage_engine):
        """Test statistics generation."""
        # Add some test opportunities
        arbitrage_engine.opportunities = [
            ArbitrageOpportunity(
                type=ArbitrageType.SIMPLE,
                symbol="BTCUSDT",
                profit_pct=Decimal("1.0"),
                profit_absolute=Decimal("500"),
                confidence_score=Decimal("8"),
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=30),
                min_capital=Decimal("100"),
                max_capital=Decimal("10000")
            ),
            ArbitrageOpportunity(
                type=ArbitrageType.TRIANGULAR,
                symbol="ETHUSDT",
                profit_pct=Decimal("2.0"),
                profit_absolute=Decimal("60"),
                confidence_score=Decimal("9"),
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=30),
                min_capital=Decimal("500"),
                max_capital=Decimal("5000")
            )
        ]
        
        stats = arbitrage_engine.get_statistics()
        
        assert stats['total_opportunities'] == 2
        assert stats['avg_profit_pct'] == 1.5  # (1.0 + 2.0) / 2
        assert stats['max_profit_pct'] == 2.0
        assert stats['type_distribution']['simple'] == 1
        assert stats['type_distribution']['triangular'] == 1


class TestArbitrageOpportunity:
    """Test arbitrage opportunity class."""
    
    def test_opportunity_properties(self):
        """Test opportunity property calculations."""
        opportunity = ArbitrageOpportunity(
            type=ArbitrageType.SIMPLE,
            symbol="BTCUSDT",
            profit_pct=Decimal("1.0"),
            profit_absolute=Decimal("500"),
            confidence_score=Decimal("8"),
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=30),
            min_capital=Decimal("100"),
            max_capital=Decimal("10000"),
            fees_estimate=Decimal("0.2"),
            slippage_estimate=Decimal("0.1")
        )
        
        # Test net profit calculation
        assert opportunity.net_profit_pct == Decimal("0.7")  # 1.0 - 0.2 - 0.1
        
        # Test profitability check
        assert opportunity.is_profitable  # 0.7% > 0.1% minimum
        
        # Test expiry check
        assert not opportunity.is_expired  # Should not be expired yet
    
    def test_expired_opportunity(self):
        """Test expired opportunity detection."""
        opportunity = ArbitrageOpportunity(
            type=ArbitrageType.SIMPLE,
            symbol="BTCUSDT",
            profit_pct=Decimal("1.0"),
            profit_absolute=Decimal("500"),
            confidence_score=Decimal("8"),
            timestamp=datetime.now() - timedelta(minutes=1),
            expires_at=datetime.now() - timedelta(seconds=1),  # Expired 1 second ago
            min_capital=Decimal("100"),
            max_capital=Decimal("10000")
        )
        
        assert opportunity.is_expired


async def test_engine_lifecycle():
    """Test engine start/stop lifecycle."""
    mock_manager = Mock(spec=ExchangeManager)
    mock_manager.get_active_exchanges.return_value = []
    
    engine = ArbitrageDetectionEngine(mock_manager)
    
    # Test start
    await engine.start()
    assert engine.running
    assert engine.update_task is not None
    
    # Test stop
    await engine.stop()
    assert not engine.running


if __name__ == "__main__":
    # Run a simple demo
    async def demo():
        """Demo the arbitrage detection engine."""
        print("Running Arbitrage Detection Engine Demo...")
        
        # Create mock exchange manager
        mock_manager = Mock(spec=ExchangeManager)
        mock_manager.get_active_exchanges.return_value = ["binance", "coinbase"]
        
        # Mock market data
        mock_tickers = {
            "binance": Ticker(
                symbol="BTCUSDT",
                timestamp=datetime.now(),
                open_price=Decimal("49000"),
                high_price=Decimal("51000"),
                low_price=Decimal("48000"),
                close_price=Decimal("50000"),
                volume=Decimal("100"),
                quote_volume=Decimal("5000000"),
                price_change=Decimal("1000"),
                price_change_percent=Decimal("2"),
                bid_price=Decimal("49950"),
                ask_price=Decimal("50050")
            ),
            "coinbase": Ticker(
                symbol="BTCUSDT",
                timestamp=datetime.now(),
                open_price=Decimal("49000"),
                high_price=Decimal("51000"),
                low_price=Decimal("48000"),
                close_price=Decimal("50300"),
                volume=Decimal("80"),
                quote_volume=Decimal("4024000"),
                price_change=Decimal("1300"),
                price_change_percent=Decimal("2.65"),
                bid_price=Decimal("50250"),
                ask_price=Decimal("50350")
            )
        }
        
        mock_order_books = {
            "binance": OrderBook(
                symbol="BTCUSDT",
                timestamp=datetime.now(),
                bids=[(Decimal("49950"), Decimal("2"))],
                asks=[(Decimal("50050"), Decimal("1.5"))]
            ),
            "coinbase": OrderBook(
                symbol="BTCUSDT",
                timestamp=datetime.now(),
                bids=[(Decimal("50250"), Decimal("1"))],
                asks=[(Decimal("50350"), Decimal("2"))]
            )
        }
        
        mock_manager.get_all_tickers = AsyncMock(return_value=mock_tickers)
        mock_manager.get_all_order_books = AsyncMock(return_value=mock_order_books)
        
        # Create and start engine
        engine = ArbitrageDetectionEngine(mock_manager)
        
        # Manually update market data and detect opportunities
        await engine._update_market_data()
        await engine._detect_opportunities()
        
        # Display results
        opportunities = engine.get_opportunities()
        print(f"Found {len(opportunities)} arbitrage opportunities:")
        
        for opp in opportunities:
            print(f"  {opp.type.value}: {opp.symbol} - {opp.profit_pct:.2f}% profit")
            print(f"    Buy on {opp.buy_exchange} at {opp.buy_price}")
            print(f"    Sell on {opp.sell_exchange} at {opp.sell_price}")
            print(f"    Net profit: {opp.net_profit_pct:.2f}% (after fees/slippage)")
            print(f"    Confidence: {opp.confidence_score}/10")
            print()
        
        # Display statistics
        stats = engine.get_statistics()
        print("Engine Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Run the demo
    asyncio.run(demo())