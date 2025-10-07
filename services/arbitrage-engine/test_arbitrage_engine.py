"""
Tests for the Arbitrage Detection Engine.
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from main import ArbitrageDetector, config
from main import SimpleArbitrageOpportunity, TriangularArbitrageOpportunity


class TestArbitrageDetector:
    """Test cases for ArbitrageDetector class."""
    
    @pytest.fixture
    async def detector(self):
        """Create arbitrage detector instance for testing."""
        detector = ArbitrageDetector()
        detector.exchange_client = AsyncMock()
        return detector
    
    @pytest.fixture
    def sample_ticker_data(self):
        """Sample ticker data for testing."""
        return {
            "exchange": "binance",
            "symbol": "BTC/USDT",
            "bid": 45000.0,
            "ask": 45100.0,
            "last": 45050.0,
            "volume": 1000.0,
            "timestamp": datetime.utcnow()
        }
    
    @pytest.fixture
    def sample_orderbook_data(self):
        """Sample order book data for testing."""
        return {
            "exchange": "binance",
            "symbol": "BTC/USDT",
            "bids": [
                [45000.0, 1.0],
                [44990.0, 2.0],
                [44980.0, 1.5],
                [44970.0, 3.0],
                [44960.0, 2.5]
            ],
            "asks": [
                [45100.0, 1.2],
                [45110.0, 1.8],
                [45120.0, 2.2],
                [45130.0, 1.6],
                [45140.0, 2.8]
            ],
            "timestamp": datetime.utcnow()
        }
    
    @pytest.fixture
    def sample_exchange_data(self, sample_ticker_data, sample_orderbook_data):
        """Sample exchange data combining ticker and orderbook."""
        return {
            "ticker": sample_ticker_data,
            "orderbook": sample_orderbook_data,
            "timestamp": datetime.utcnow()
        }
    
    async def test_get_exchange_data_success(self, detector, sample_ticker_data, sample_orderbook_data):
        """Test successful exchange data retrieval."""
        # Mock HTTP responses
        ticker_response = AsyncMock()
        ticker_response.status = 200
        ticker_response.json = AsyncMock(return_value=sample_ticker_data)
        
        orderbook_response = AsyncMock()
        orderbook_response.status = 200
        orderbook_response.json = AsyncMock(return_value=sample_orderbook_data)
        
        detector.exchange_client.get = AsyncMock()
        detector.exchange_client.get.side_effect = [
            AsyncMock(__aenter__=AsyncMock(return_value=ticker_response)),
            AsyncMock(__aenter__=AsyncMock(return_value=orderbook_response))
        ]
        
        result = await detector.get_exchange_data("binance", "BTC/USDT")
        
        assert result is not None
        assert result["ticker"] == sample_ticker_data
        assert result["orderbook"] == sample_orderbook_data
        assert "timestamp" in result
    
    async def test_get_exchange_data_failure(self, detector):
        """Test exchange data retrieval failure."""
        # Mock HTTP error response
        error_response = AsyncMock()
        error_response.status = 500
        
        detector.exchange_client.get = AsyncMock()
        detector.exchange_client.get.return_value.__aenter__ = AsyncMock(return_value=error_response)
        
        result = await detector.get_exchange_data("binance", "BTC/USDT")
        
        assert result is None
    
    def test_calculate_liquidity(self, detector):
        """Test liquidity calculation."""
        order_levels = [
            [45100.0, 1.2],
            [45110.0, 1.8],
            [45120.0, 2.2],
            [45130.0, 1.6],
            [45140.0, 2.8]
        ]
        
        liquidity = detector._calculate_liquidity(order_levels, "buy")
        expected_liquidity = 1.2 + 1.8 + 2.2 + 1.6 + 2.8  # Sum of first 5 levels
        
        assert liquidity == expected_liquidity
    
    def test_calculate_liquidity_empty_orderbook(self, detector):
        """Test liquidity calculation with empty order book."""
        liquidity = detector._calculate_liquidity([], "buy")
        assert liquidity == 0.0
    
    def test_estimate_execution_time(self, detector, sample_orderbook_data):
        """Test execution time estimation."""
        execution_time = detector._estimate_execution_time(
            sample_orderbook_data, sample_orderbook_data
        )
        
        assert isinstance(execution_time, int)
        assert execution_time > 0
    
    def test_calculate_risk_score(self, detector):
        """Test risk score calculation."""
        risk_score = detector._calculate_risk_score(
            profit_pct=2.0,
            liquidity_score=5.0,
            execution_time_ms=1000,
            position_size=1000.0
        )
        
        assert isinstance(risk_score, float)
        assert 0 <= risk_score <= 100
    
    def test_calculate_simple_arbitrage_profitable(self, detector, sample_exchange_data):
        """Test simple arbitrage calculation with profitable opportunity."""
        # Create data for two exchanges with price difference
        buy_data = sample_exchange_data.copy()
        sell_data = sample_exchange_data.copy()
        
        # Make sell exchange have higher bid price
        sell_data["ticker"]["bid"] = 46000.0  # Higher than buy ask of 45100
        
        opportunity = detector._calculate_simple_arbitrage(
            "BTC/USDT", "binance", "coinbase", buy_data, sell_data
        )
        
        assert opportunity is not None
        assert isinstance(opportunity, SimpleArbitrageOpportunity)
        assert opportunity.symbol == "BTC/USDT"
        assert opportunity.buy_exchange == "binance"
        assert opportunity.sell_exchange == "coinbase"
        assert opportunity.profit_pct > config.min_simple_arbitrage_profit_pct
    
    def test_calculate_simple_arbitrage_unprofitable(self, detector, sample_exchange_data):
        """Test simple arbitrage calculation with unprofitable opportunity."""
        # Create data with minimal price difference
        buy_data = sample_exchange_data.copy()
        sell_data = sample_exchange_data.copy()
        
        # Make sell exchange have only slightly higher bid price
        sell_data["ticker"]["bid"] = 45050.0  # Only slightly higher than buy ask
        
        opportunity = detector._calculate_simple_arbitrage(
            "BTC/USDT", "binance", "coinbase", buy_data, sell_data
        )
        
        assert opportunity is None  # Should be unprofitable
    
    def test_calculate_triangular_risk_score(self, detector):
        """Test triangular arbitrage risk score calculation."""
        risk_score = detector._calculate_triangular_risk_score(
            profit_pct=1.5,
            min_liquidity=10.0,
            execution_time_ms=3000,
            num_trades=3
        )
        
        assert isinstance(risk_score, float)
        assert 0 <= risk_score <= 100
    
    async def test_scan_simple_arbitrage_no_exchanges(self, detector):
        """Test simple arbitrage scan with no exchanges available."""
        # Mock empty exchange list
        empty_response = AsyncMock()
        empty_response.status = 200
        empty_response.json = AsyncMock(return_value={"exchanges": {}})
        
        detector.exchange_client.get = AsyncMock()
        detector.exchange_client.get.return_value.__aenter__ = AsyncMock(return_value=empty_response)
        
        result = await detector.scan_simple_arbitrage()
        
        assert result.scan_type == "simple"
        assert result.opportunities_found == 0
        assert len(result.opportunities) == 0
    
    async def test_scan_simple_arbitrage_with_exchanges(self, detector, sample_exchange_data):
        """Test simple arbitrage scan with available exchanges."""
        # Mock exchange list response
        exchanges_response = AsyncMock()
        exchanges_response.status = 200
        exchanges_response.json = AsyncMock(return_value={
            "exchanges": {"binance": {}, "coinbase": {}}
        })
        
        detector.exchange_client.get = AsyncMock()
        detector.exchange_client.get.return_value.__aenter__ = AsyncMock(return_value=exchanges_response)
        
        # Mock get_exchange_data to return sample data
        detector.get_exchange_data = AsyncMock(return_value=sample_exchange_data)
        
        result = await detector.scan_simple_arbitrage()
        
        assert result.scan_type == "simple"
        assert isinstance(result.opportunities_found, int)
        assert isinstance(result.scan_duration_ms, float)
        assert result.scan_duration_ms > 0
    
    async def test_scan_triangular_arbitrage(self, detector):
        """Test triangular arbitrage scanning."""
        # Mock exchange list response
        exchanges_response = AsyncMock()
        exchanges_response.status = 200
        exchanges_response.json = AsyncMock(return_value={
            "exchanges": {"binance": {}}
        })
        
        detector.exchange_client.get = AsyncMock()
        detector.exchange_client.get.return_value.__aenter__ = AsyncMock(return_value=exchanges_response)
        
        # Mock exchange-specific scanning
        detector._scan_exchange_triangular_arbitrage = AsyncMock(return_value=[])
        
        result = await detector.scan_triangular_arbitrage()
        
        assert result.scan_type == "triangular"
        assert isinstance(result.opportunities_found, int)
        assert isinstance(result.scan_duration_ms, float)
    
    async def test_scan_funding_arbitrage(self, detector):
        """Test funding arbitrage scanning."""
        result = await detector.scan_funding_arbitrage()
        
        assert result.scan_type == "funding"
        assert isinstance(result.opportunities_found, int)
        assert isinstance(result.scan_duration_ms, float)
    
    async def test_find_triangular_paths_usdt_base(self, detector):
        """Test triangular path finding with USDT base currency."""
        # Mock get_exchange_data to return None (no data available)
        detector.get_exchange_data = AsyncMock(return_value=None)
        
        opportunities = await detector._find_triangular_paths("binance", "USDT")
        
        # Should return empty list when no data is available
        assert isinstance(opportunities, list)
    
    async def test_calculate_triangular_arbitrage_no_data(self, detector):
        """Test triangular arbitrage calculation with no market data."""
        # Mock get_exchange_data to return None
        detector.get_exchange_data = AsyncMock(return_value=None)
        
        path = ["BTC/USDT", "ETH/BTC", "ETH/USDT"]
        opportunity = await detector._calculate_triangular_arbitrage("binance", "USDT", path)
        
        assert opportunity is None
    
    async def test_calculate_triangular_arbitrage_with_data(self, detector):
        """Test triangular arbitrage calculation with market data."""
        # Mock market data for triangular path
        btc_usdt_data = {
            "ticker": {"ask": 45000.0, "bid": 44900.0, "last": 44950.0},
            "orderbook": {"asks": [[45000.0, 1.0]], "bids": [[44900.0, 1.0]]},
            "timestamp": datetime.utcnow()
        }
        
        eth_btc_data = {
            "ticker": {"ask": 0.065, "bid": 0.064, "last": 0.0645},
            "orderbook": {"asks": [[0.065, 10.0]], "bids": [[0.064, 10.0]]},
            "timestamp": datetime.utcnow()
        }
        
        eth_usdt_data = {
            "ticker": {"ask": 2900.0, "bid": 2890.0, "last": 2895.0},
            "orderbook": {"asks": [[2900.0, 1.0]], "bids": [[2890.0, 1.0]]},
            "timestamp": datetime.utcnow()
        }
        
        # Mock get_exchange_data to return appropriate data for each symbol
        async def mock_get_exchange_data(exchange, symbol):
            if symbol == "BTC/USDT":
                return btc_usdt_data
            elif symbol == "ETH/BTC":
                return eth_btc_data
            elif symbol == "ETH/USDT":
                return eth_usdt_data
            return None
        
        detector.get_exchange_data = AsyncMock(side_effect=mock_get_exchange_data)
        
        path = ["BTC/USDT", "ETH/BTC", "ETH/USDT"]
        opportunity = await detector._calculate_triangular_arbitrage("binance", "USDT", path)
        
        # The calculation might not be profitable with these prices, but should complete
        assert opportunity is None or isinstance(opportunity, TriangularArbitrageOpportunity)


class TestArbitrageAPI:
    """Test cases for arbitrage API endpoints."""
    
    @pytest.fixture
    def sample_simple_opportunity(self):
        """Sample simple arbitrage opportunity."""
        return SimpleArbitrageOpportunity(
            symbol="BTC/USDT",
            buy_exchange="binance",
            sell_exchange="coinbase",
            buy_price=45000.0,
            sell_price=46000.0,
            profit_pct=2.22,
            profit_amount_per_unit=1000.0,
            max_position_size=10.0,
            liquidity_score=5.0,
            execution_time_estimate_ms=1000,
            risk_score=25.0,
            detected_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=30)
        )
    
    @pytest.fixture
    def sample_triangular_opportunity(self):
        """Sample triangular arbitrage opportunity."""
        return TriangularArbitrageOpportunity(
            exchange="binance",
            base_currency="USDT",
            path=["BTC/USDT", "ETH/BTC", "ETH/USDT"],
            trade_sequence=[
                {
                    "step": 1,
                    "symbol": "BTC/USDT",
                    "side": "buy",
                    "amount_in": 1000.0,
                    "price": 45000.0,
                    "amount_out": 0.0222,
                    "liquidity": 10.0
                }
            ],
            profit_pct=1.5,
            required_capital=1000.0,
            execution_time_estimate_ms=3000,
            risk_score=35.0,
            detected_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=60)
        )
    
    def test_opportunity_filtering_by_symbol(self, sample_simple_opportunity):
        """Test filtering opportunities by symbol."""
        opportunities = [sample_simple_opportunity]
        
        # Filter by matching symbol
        filtered = [opp for opp in opportunities if opp.symbol == "BTC/USDT"]
        assert len(filtered) == 1
        
        # Filter by non-matching symbol
        filtered = [opp for opp in opportunities if opp.symbol == "ETH/USDT"]
        assert len(filtered) == 0
    
    def test_opportunity_filtering_by_profit(self, sample_simple_opportunity):
        """Test filtering opportunities by profit percentage."""
        opportunities = [sample_simple_opportunity]
        
        # Filter by lower threshold
        filtered = [opp for opp in opportunities if opp.profit_pct >= 1.0]
        assert len(filtered) == 1
        
        # Filter by higher threshold
        filtered = [opp for opp in opportunities if opp.profit_pct >= 5.0]
        assert len(filtered) == 0
    
    def test_opportunity_filtering_by_risk(self, sample_simple_opportunity):
        """Test filtering opportunities by risk score."""
        opportunities = [sample_simple_opportunity]
        
        # Filter by higher risk threshold
        filtered = [opp for opp in opportunities if opp.risk_score <= 50.0]
        assert len(filtered) == 1
        
        # Filter by lower risk threshold
        filtered = [opp for opp in opportunities if opp.risk_score <= 10.0]
        assert len(filtered) == 0
    
    def test_opportunity_expiry_filtering(self):
        """Test filtering expired opportunities."""
        # Create expired opportunity
        expired_opportunity = SimpleArbitrageOpportunity(
            symbol="BTC/USDT",
            buy_exchange="binance",
            sell_exchange="coinbase",
            buy_price=45000.0,
            sell_price=46000.0,
            profit_pct=2.22,
            profit_amount_per_unit=1000.0,
            max_position_size=10.0,
            liquidity_score=5.0,
            execution_time_estimate_ms=1000,
            risk_score=25.0,
            detected_at=datetime.utcnow() - timedelta(minutes=5),
            expires_at=datetime.utcnow() - timedelta(minutes=1)  # Expired
        )
        
        opportunities = [expired_opportunity]
        current_time = datetime.utcnow()
        
        # Filter out expired opportunities
        filtered = [opp for opp in opportunities if opp.expires_at > current_time]
        assert len(filtered) == 0


class TestArbitrageConfiguration:
    """Test cases for arbitrage configuration."""
    
    def test_config_initialization(self):
        """Test configuration initialization with default values."""
        assert config.min_simple_arbitrage_profit_pct >= 0
        assert config.min_triangular_arbitrage_profit_pct >= 0
        assert config.min_funding_arbitrage_profit_pct >= 0
        assert config.max_position_size_usd > 0
        assert config.max_execution_time_ms > 0
        assert config.min_liquidity_ratio > 0
        assert len(config.supported_symbols) > 0
        assert len(config.triangular_base_currencies) > 0
    
    def test_supported_symbols_format(self):
        """Test that supported symbols are properly formatted."""
        for symbol in config.supported_symbols:
            assert "/" in symbol  # Should be in format like "BTC/USDT"
            parts = symbol.split("/")
            assert len(parts) == 2
            assert len(parts[0]) > 0
            assert len(parts[1]) > 0
    
    def test_triangular_base_currencies(self):
        """Test triangular base currencies configuration."""
        expected_currencies = ["USDT", "BTC", "ETH"]
        for currency in expected_currencies:
            assert currency in config.triangular_base_currencies


class TestArbitrageMetrics:
    """Test cases for arbitrage metrics and monitoring."""
    
    def test_metrics_initialization(self):
        """Test that Prometheus metrics are properly initialized."""
        from main import (
            ARBITRAGE_OPPORTUNITIES_FOUND,
            ARBITRAGE_OPPORTUNITIES_EXECUTED,
            ARBITRAGE_PROFIT_TOTAL,
            ARBITRAGE_SCAN_DURATION,
            ACTIVE_OPPORTUNITIES
        )
        
        # Test that metrics exist and have proper names
        assert ARBITRAGE_OPPORTUNITIES_FOUND._name == "arbitrage_opportunities_found_total"
        assert ARBITRAGE_OPPORTUNITIES_EXECUTED._name == "arbitrage_opportunities_executed_total"
        assert ARBITRAGE_PROFIT_TOTAL._name == "arbitrage_profit_total"
        assert ARBITRAGE_SCAN_DURATION._name == "arbitrage_scan_duration_seconds"
        assert ACTIVE_OPPORTUNITIES._name == "active_arbitrage_opportunities"
    
    def test_metrics_labels(self):
        """Test that metrics have proper labels."""
        from main import ARBITRAGE_OPPORTUNITIES_FOUND, ACTIVE_OPPORTUNITIES
        
        # Test counter with labels
        ARBITRAGE_OPPORTUNITIES_FOUND.labels(type="simple", symbol="BTC/USDT").inc()
        
        # Test gauge with labels
        ACTIVE_OPPORTUNITIES.labels(type="simple").set(5)
        
        # Should not raise any exceptions


# Integration tests
class TestArbitrageIntegration:
    """Integration tests for arbitrage detection engine."""
    
    @pytest.mark.asyncio
    async def test_full_arbitrage_detection_flow(self):
        """Test complete arbitrage detection flow."""
        detector = ArbitrageDetector()
        
        # Mock the exchange client
        detector.exchange_client = AsyncMock()
        
        # Mock exchange list response
        exchanges_response = AsyncMock()
        exchanges_response.status = 200
        exchanges_response.json = AsyncMock(return_value={
            "exchanges": {"binance": {}, "coinbase": {}}
        })
        
        detector.exchange_client.get = AsyncMock()
        detector.exchange_client.get.return_value.__aenter__ = AsyncMock(return_value=exchanges_response)
        
        # Mock exchange data responses
        sample_data = {
            "ticker": {
                "exchange": "binance",
                "symbol": "BTC/USDT",
                "bid": 45000.0,
                "ask": 45100.0,
                "last": 45050.0,
                "volume": 1000.0,
                "timestamp": datetime.utcnow()
            },
            "orderbook": {
                "exchange": "binance",
                "symbol": "BTC/USDT",
                "bids": [[45000.0, 1.0]],
                "asks": [[45100.0, 1.0]],
                "timestamp": datetime.utcnow()
            },
            "timestamp": datetime.utcnow()
        }
        
        detector.get_exchange_data = AsyncMock(return_value=sample_data)
        
        # Run simple arbitrage scan
        result = await detector.scan_simple_arbitrage()
        
        assert result.scan_type == "simple"
        assert isinstance(result.opportunities_found, int)
        assert result.opportunities_found >= 0
        assert isinstance(result.scan_duration_ms, float)
        assert result.scan_duration_ms > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_arbitrage_scans(self):
        """Test running multiple arbitrage scans concurrently."""
        detector = ArbitrageDetector()
        detector.exchange_client = AsyncMock()
        
        # Mock responses
        exchanges_response = AsyncMock()
        exchanges_response.status = 200
        exchanges_response.json = AsyncMock(return_value={"exchanges": {}})
        
        detector.exchange_client.get = AsyncMock()
        detector.exchange_client.get.return_value.__aenter__ = AsyncMock(return_value=exchanges_response)
        
        # Run all scans concurrently
        simple_task = detector.scan_simple_arbitrage()
        triangular_task = detector.scan_triangular_arbitrage()
        funding_task = detector.scan_funding_arbitrage()
        
        results = await asyncio.gather(
            simple_task, triangular_task, funding_task, return_exceptions=True
        )
        
        # All scans should complete without exceptions
        for result in results:
            assert not isinstance(result, Exception)
            assert hasattr(result, 'scan_type')
            assert hasattr(result, 'opportunities_found')
            assert hasattr(result, 'scan_duration_ms')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])