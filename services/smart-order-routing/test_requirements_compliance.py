"""
Test compliance with specific requirements for Smart Order Routing System.
Tests for Requirements 2.5, 6.4, and 6.5.
"""
import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from main import (
    SmartOrderRouter, 
    OrderRequest, 
    RoutingDecision, 
    RoutingReason,
    ExchangeMetrics
)


class MockExchangeForRequirements:
    """Mock exchange specifically for requirements testing."""
    
    def __init__(self, name: str, latency_ms: float = 100, success_rate: float = 1.0, 
                 price_offset: float = 0, liquidity_multiplier: float = 1.0):
        self.name = name
        self.latency_ms = latency_ms
        self.success_rate = success_rate
        self.price_offset = price_offset
        self.liquidity_multiplier = liquidity_multiplier
        self.call_count = 0
        self.is_failing = False
    
    def set_failing(self, failing: bool):
        """Set exchange to failing state for testing failover."""
        self.is_failing = failing
    
    async def get_status(self):
        """Mock get status."""
        await asyncio.sleep(self.latency_ms / 1000)
        self.call_count += 1
        
        if self.is_failing:
            raise Exception(f"Exchange {self.name} is failing")
        
        return {"status": "online", "timestamp": datetime.utcnow().isoformat()}
    
    async def get_ticker(self, symbol: str):
        """Mock get ticker with different prices for best execution testing."""
        await asyncio.sleep(self.latency_ms / 1000)
        
        if self.is_failing:
            raise Exception(f"Exchange {self.name} ticker unavailable")
        
        base_price = 50000 if symbol == "BTC/USDT" else 3000
        
        return {
            "bid": base_price + self.price_offset - 5,
            "ask": base_price + self.price_offset + 5,
            "last": base_price + self.price_offset,
            "volume": 1000.0 * self.liquidity_multiplier,
            "timestamp": datetime.utcnow()
        }
    
    async def get_order_book(self, symbol: str, limit: int = 100):
        """Mock order book with different liquidity levels."""
        await asyncio.sleep(self.latency_ms / 1000)
        
        if self.is_failing:
            raise Exception(f"Exchange {self.name} order book unavailable")
        
        base_price = 50000 if symbol == "BTC/USDT" else 3000
        
        bids = []
        asks = []
        
        for i in range(10):
            bid_price = base_price + self.price_offset - 5 - i
            ask_price = base_price + self.price_offset + 5 + i
            
            bid_size = (10 - i) * self.liquidity_multiplier
            ask_size = (10 - i) * self.liquidity_multiplier
            
            bids.append([bid_price, bid_size])
            asks.append([ask_price, ask_size])
        
        return {
            "bids": bids,
            "asks": asks,
            "timestamp": datetime.utcnow()
        }
    
    async def place_order(self, symbol: str, side: str, type: str, amount: float, 
                         price: float = None, time_in_force: str = "GTC"):
        """Mock place order."""
        await asyncio.sleep(self.latency_ms / 1000)
        
        if self.is_failing:
            raise Exception(f"Exchange {self.name} order placement failed")
        
        return {
            "order_id": f"{self.name}_{symbol}_{int(datetime.utcnow().timestamp())}",
            "symbol": symbol,
            "side": side,
            "type": type,
            "amount": amount,
            "price": price,
            "status": "filled",
            "timestamp": datetime.utcnow(),
            "fees": {"currency": "USDT", "amount": amount * 0.001}
        }


@pytest.fixture
async def requirement_test_router():
    """Create router with exchanges designed for requirements testing."""
    exchanges = {
        # Best price exchange (Requirement 2.5 & 6.5)
        "best_price_exchange": MockExchangeForRequirements(
            "best_price_exchange", 
            latency_ms=80, 
            price_offset=-10,  # Better prices
            liquidity_multiplier=2.0
        ),
        
        # High liquidity exchange (Requirement 6.5)
        "high_liquidity_exchange": MockExchangeForRequirements(
            "high_liquidity_exchange", 
            latency_ms=120, 
            price_offset=5,   # Slightly worse prices
            liquidity_multiplier=5.0  # Much higher liquidity
        ),
        
        # Fast but unreliable exchange (for failover testing)
        "fast_unreliable_exchange": MockExchangeForRequirements(
            "fast_unreliable_exchange", 
            latency_ms=30, 
            success_rate=0.7,
            price_offset=0
        ),
        
        # Backup exchange (Requirement 6.4)
        "backup_exchange": MockExchangeForRequirements(
            "backup_exchange", 
            latency_ms=200, 
            price_offset=15,  # Worse prices but reliable
            liquidity_multiplier=1.5
        )
    }
    
    router = SmartOrderRouter()
    await router.initialize(exchanges)
    
    # Wait for initial health checks
    await asyncio.sleep(0.2)
    
    return router, exchanges


class TestRequirement25:
    """Test Requirement 2.5: Route orders to exchange with best execution."""
    
    @pytest.mark.asyncio
    async def test_routes_to_best_execution_exchange(self, requirement_test_router):
        """Test that system routes to exchange with best execution."""
        router, exchanges = requirement_test_router
        
        order = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=1.0,
            execution_strategy="best_execution"
        )
        
        decision = await router.route_order(order)
        
        # Should route to best_price_exchange due to better prices and good liquidity
        assert decision.selected_exchange == "best_price_exchange"
        assert decision.reason == RoutingReason.BEST_PRICE
        assert decision.confidence > 0.8
        
        # Verify execution quality metrics
        assert decision.expected_price > 0
        assert decision.expected_slippage_bps >= 0
        assert decision.liquidity_score > 0
    
    @pytest.mark.asyncio
    async def test_considers_multiple_execution_factors(self, requirement_test_router):
        """Test that best execution considers price, liquidity, and reliability."""
        router, exchanges = requirement_test_router
        
        # Test with large order that needs high liquidity
        large_order = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=10.0,  # Large order
            execution_strategy="best_execution"
        )
        
        decision = await router.route_order(large_order)
        
        # For large orders, might prefer high liquidity exchange despite worse prices
        assert decision.selected_exchange in ["best_price_exchange", "high_liquidity_exchange"]
        assert decision.liquidity_score > 2.0  # Should have good liquidity
        
        # Check that alternatives are provided
        assert len(decision.alternatives) > 0


class TestRequirement64:
    """Test Requirement 6.4: Automatic failover when exchange API fails."""
    
    @pytest.mark.asyncio
    async def test_automatic_failover_on_api_failure(self, requirement_test_router):
        """Test automatic failover when primary exchange API fails."""
        router, exchanges = requirement_test_router
        
        order = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=1.0,
            execution_strategy="best_execution"
        )
        
        # Get initial routing decision
        decision = await router.route_order(order)
        primary_exchange = decision.selected_exchange
        
        # Make primary exchange fail
        exchanges[primary_exchange].set_failing(True)
        
        # Execute with failover
        result = await router.execute_with_failover(order, decision)
        
        # Should succeed with failover
        assert result["success"] is True
        assert result["failover_used"] is True
        assert result["exchange"] != primary_exchange
        assert "primary_error" in result
        
        # Verify backup exchange was used
        backup_exchange = result["exchange"]
        assert backup_exchange in exchanges
        assert not exchanges[backup_exchange].is_failing
    
    @pytest.mark.asyncio
    async def test_failover_updates_exchange_health(self, requirement_test_router):
        """Test that failover updates exchange health metrics."""
        router, exchanges = requirement_test_router
        
        # Make an exchange fail
        exchanges["fast_unreliable_exchange"].set_failing(True)
        
        # Wait for health monitoring to detect failure
        await asyncio.sleep(0.5)
        
        # Check that exchange is marked as unhealthy
        status = router.get_exchange_status()
        assert not status["fast_unreliable_exchange"]["is_healthy"]
        assert status["fast_unreliable_exchange"]["consecutive_failures"] > 0
    
    @pytest.mark.asyncio
    async def test_emergency_failover_when_all_alternatives_fail(self, requirement_test_router):
        """Test emergency failover when all backup exchanges fail."""
        router, exchanges = requirement_test_router
        
        order = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=1.0,
            execution_strategy="best_execution"
        )
        
        # Get routing decision
        decision = await router.route_order(order)
        
        # Make primary and most backup exchanges fail
        for name in ["best_price_exchange", "high_liquidity_exchange", "fast_unreliable_exchange"]:
            exchanges[name].set_failing(True)
        
        # Should still succeed with emergency failover to backup_exchange
        result = await router.execute_with_failover(order, decision)
        
        assert result["success"] is True
        assert result["failover_used"] is True
        assert result.get("emergency_failover") is True
        assert result["exchange"] == "backup_exchange"


class TestRequirement65:
    """Test Requirement 6.5: Select exchange with best price and liquidity."""
    
    @pytest.mark.asyncio
    async def test_selects_best_price_when_liquidity_adequate(self, requirement_test_router):
        """Test selection of best price when liquidity is adequate."""
        router, exchanges = requirement_test_router
        
        # Small order where price is most important
        small_order = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=0.5,  # Small order
            execution_strategy="best_execution"
        )
        
        decision = await router.route_order(small_order)
        
        # Should prefer best_price_exchange for small orders
        assert decision.selected_exchange == "best_price_exchange"
        assert decision.expected_price > 0
    
    @pytest.mark.asyncio
    async def test_selects_best_liquidity_for_large_orders(self, requirement_test_router):
        """Test selection based on liquidity for large orders."""
        router, exchanges = requirement_test_router
        
        # Use best liquidity strategy explicitly
        liquidity_order = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=5.0,  # Large order
            execution_strategy="best_liquidity"
        )
        
        decision = await router.route_order(liquidity_order)
        
        # Should prefer high_liquidity_exchange
        assert decision.selected_exchange == "high_liquidity_exchange"
        assert decision.reason == RoutingReason.BEST_LIQUIDITY
        assert decision.liquidity_score > 3.0  # High liquidity score
    
    @pytest.mark.asyncio
    async def test_price_liquidity_tradeoff_analysis(self, requirement_test_router):
        """Test that system properly analyzes price vs liquidity tradeoffs."""
        router, exchanges = requirement_test_router
        
        order = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=2.0,  # Medium order
            execution_strategy="best_execution"
        )
        
        decision = await router.route_order(order)
        
        # Should have analyzed multiple exchanges
        assert len(decision.alternatives) >= 2
        
        # Check that decision metadata includes execution cost analysis
        assert "execution_cost" in decision.metadata
        
        # Verify that the selected exchange has reasonable execution quality
        assert decision.confidence > 0.7
        assert decision.expected_slippage_bps < 100  # Reasonable slippage


class TestIntegratedRequirements:
    """Test integrated scenarios covering all three requirements."""
    
    @pytest.mark.asyncio
    async def test_best_execution_with_failover_scenario(self, requirement_test_router):
        """Test best execution routing with failover capability."""
        router, exchanges = requirement_test_router
        
        order = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=1.0,
            execution_strategy="best_execution",
            max_slippage_bps=50
        )
        
        # Initial routing should select best execution exchange
        decision = await router.route_order(order)
        initial_exchange = decision.selected_exchange
        
        # Verify best execution criteria
        assert decision.reason == RoutingReason.BEST_PRICE
        assert decision.expected_slippage_bps <= order.max_slippage_bps
        
        # Simulate exchange failure and test failover
        exchanges[initial_exchange].set_failing(True)
        
        result = await router.execute_with_failover(order, decision)
        
        # Should succeed with failover
        assert result["success"] is True
        assert result["failover_used"] is True
        assert result["exchange"] != initial_exchange
    
    @pytest.mark.asyncio
    async def test_high_frequency_trading_optimization(self, requirement_test_router):
        """Test HFT optimizations for latency-sensitive orders."""
        router, exchanges = requirement_test_router
        
        # Small, frequent order typical of HFT
        hft_order = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=0.1,  # Very small order
            execution_strategy="lowest_latency"
        )
        
        decision = await router.route_order(hft_order)
        
        # Should prioritize latency
        assert decision.reason == RoutingReason.LOWEST_LATENCY
        assert decision.latency_ms < 200  # Should be low latency
        
        # Should have high confidence for latency-optimized routing
        assert decision.confidence >= 0.8
    
    @pytest.mark.asyncio
    async def test_comprehensive_routing_statistics(self, requirement_test_router):
        """Test that routing system provides comprehensive performance statistics."""
        router, exchanges = requirement_test_router
        
        # Execute several orders to generate statistics
        orders = [
            OrderRequest(symbol="BTC/USDT", side="buy", type="market", amount=1.0, execution_strategy="best_execution"),
            OrderRequest(symbol="BTC/USDT", side="sell", type="market", amount=0.5, execution_strategy="lowest_latency"),
            OrderRequest(symbol="BTC/USDT", side="buy", type="market", amount=2.0, execution_strategy="best_liquidity"),
        ]
        
        for order in orders:
            await router.route_order(order)
        
        # Check exchange status provides comprehensive metrics
        status = router.get_exchange_status()
        
        for exchange_name, metrics in status.items():
            assert "is_healthy" in metrics
            assert "latency_ms" in metrics
            assert "latency_category" in metrics
            assert "success_rate" in metrics
            assert "health_score" in metrics
            assert "liquidity_score" in metrics
            
            # Verify latency categorization
            assert metrics["latency_category"] in ["excellent", "good", "acceptable", "poor", "unacceptable"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])