"""
Tests for Smart Order Routing Service.
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from main import (
    SmartOrderRouter, 
    OrderRequest, 
    RoutingDecision, 
    RoutingReason,
    ExchangeMetrics
)


class MockExchange:
    """Mock exchange for testing."""
    
    def __init__(self, name: str, latency_ms: float = 100, success_rate: float = 1.0):
        self.name = name
        self.latency_ms = latency_ms
        self.success_rate = success_rate
        self.call_count = 0
    
    async def get_status(self):
        """Mock get status."""
        await asyncio.sleep(self.latency_ms / 1000)  # Simulate latency
        self.call_count += 1
        
        if self.call_count % (1 / self.success_rate) == 0 and self.success_rate < 1.0:
            raise Exception("Simulated exchange error")
        
        return {"status": "online", "timestamp": datetime.utcnow().isoformat()}
    
    async def get_ticker(self, symbol: str):
        """Mock get ticker."""
        await asyncio.sleep(self.latency_ms / 1000)
        self.call_count += 1
        
        if self.call_count % (1 / self.success_rate) == 0 and self.success_rate < 1.0:
            raise Exception("Simulated ticker error")
        
        # Return different prices for different exchanges
        base_price = 50000 if symbol == "BTC/USDT" else 3000
        price_offset = hash(self.name) % 100  # Deterministic price difference
        
        return {
            "bid": base_price + price_offset - 5,
            "ask": base_price + price_offset + 5,
            "last": base_price + price_offset,
            "volume": 1000.0,
            "timestamp": datetime.utcnow()
        }
    
    async def get_order_book(self, symbol: str, limit: int = 100):
        """Mock get order book."""
        await asyncio.sleep(self.latency_ms / 1000)
        self.call_count += 1
        
        if self.call_count % (1 / self.success_rate) == 0 and self.success_rate < 1.0:
            raise Exception("Simulated order book error")
        
        base_price = 50000 if symbol == "BTC/USDT" else 3000
        price_offset = hash(self.name) % 100
        
        # Generate mock order book with different liquidity
        liquidity_multiplier = 1.0 + (hash(self.name) % 5) * 0.2  # 1.0 to 2.0
        
        bids = []
        asks = []
        
        for i in range(10):
            bid_price = base_price + price_offset - 5 - i
            ask_price = base_price + price_offset + 5 + i
            
            bid_size = (10 - i) * liquidity_multiplier
            ask_size = (10 - i) * liquidity_multiplier
            
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
        self.call_count += 1
        
        if self.call_count % (1 / self.success_rate) == 0 and self.success_rate < 1.0:
            raise Exception("Simulated order placement error")
        
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
def mock_exchanges():
    """Create mock exchanges with different characteristics."""
    return {
        "binance": MockExchange("binance", latency_ms=50, success_rate=0.99),
        "coinbase": MockExchange("coinbase", latency_ms=100, success_rate=0.98),
        "kraken": MockExchange("kraken", latency_ms=150, success_rate=0.97),
        "slow_exchange": MockExchange("slow_exchange", latency_ms=500, success_rate=0.95),
        "unreliable_exchange": MockExchange("unreliable_exchange", latency_ms=80, success_rate=0.8)
    }


@pytest.fixture
async def smart_router(mock_exchanges):
    """Create and initialize smart router with mock exchanges."""
    router = SmartOrderRouter()
    await router.initialize(mock_exchanges)
    
    # Wait a bit for initial metrics to stabilize
    await asyncio.sleep(0.1)
    
    return router


@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    return OrderRequest(
        symbol="BTC/USDT",
        side="buy",
        type="market",
        amount=1.0,
        max_slippage_bps=50,
        execution_strategy="best_execution"
    )


class TestSmartOrderRouter:
    """Test cases for Smart Order Router."""
    
    @pytest.mark.asyncio
    async def test_router_initialization(self, mock_exchanges):
        """Test router initialization."""
        router = SmartOrderRouter()
        await router.initialize(mock_exchanges)
        
        assert len(router.exchange_metrics) == len(mock_exchanges)
        assert len(router.exchange_clients) == len(mock_exchanges)
        
        for exchange_name in mock_exchanges.keys():
            assert exchange_name in router.exchange_metrics
            assert isinstance(router.exchange_metrics[exchange_name], ExchangeMetrics)
    
    @pytest.mark.asyncio
    async def test_best_execution_strategy(self, smart_router, sample_order):
        """Test best execution strategy."""
        decision = await smart_router.route_order(sample_order)
        
        assert isinstance(decision, RoutingDecision)
        assert decision.selected_exchange in smart_router.exchange_clients
        assert decision.reason == RoutingReason.BEST_PRICE
        assert decision.expected_price > 0
        assert decision.confidence > 0
        assert len(decision.alternatives) > 0
    
    @pytest.mark.asyncio
    async def test_lowest_latency_strategy(self, smart_router, sample_order):
        """Test lowest latency strategy."""
        sample_order.execution_strategy = "lowest_latency"
        decision = await smart_router.route_order(sample_order)
        
        assert decision.reason == RoutingReason.LOWEST_LATENCY
        assert decision.selected_exchange in ["binance", "coinbase", "unreliable_exchange"]  # Fastest exchanges
        assert decision.latency_ms <= 200  # Should select low-latency exchange
    
    @pytest.mark.asyncio
    async def test_best_liquidity_strategy(self, smart_router, sample_order):
        """Test best liquidity strategy."""
        sample_order.execution_strategy = "best_liquidity"
        decision = await smart_router.route_order(sample_order)
        
        assert decision.reason == RoutingReason.BEST_LIQUIDITY
        assert decision.liquidity_score > 0
    
    @pytest.mark.asyncio
    async def test_cost_optimization_strategy(self, smart_router, sample_order):
        """Test cost optimization strategy."""
        sample_order.execution_strategy = "cost_optimization"
        decision = await smart_router.route_order(sample_order)
        
        assert decision.reason == RoutingReason.COST_OPTIMIZATION
        assert "total_cost" in decision.metadata
    
    @pytest.mark.asyncio
    async def test_failover_execution(self, smart_router, sample_order):
        """Test failover execution when primary exchange fails."""
        # Get initial routing decision
        decision = await smart_router.route_order(sample_order)
        primary_exchange = decision.selected_exchange
        
        # Make primary exchange fail
        smart_router.exchange_clients[primary_exchange].success_rate = 0.0
        
        # Execute with failover
        result = await smart_router.execute_with_failover(sample_order, decision)
        
        assert result["success"] is True
        assert result["failover_used"] is True
        assert result["exchange"] != primary_exchange
        assert "primary_error" in result
    
    @pytest.mark.asyncio
    async def test_exchange_health_monitoring(self, smart_router):
        """Test exchange health monitoring."""
        # Wait for health monitoring to run
        await asyncio.sleep(0.2)
        
        status = smart_router.get_exchange_status()
        
        assert len(status) == len(smart_router.exchange_clients)
        
        for exchange_name, metrics in status.items():
            assert "is_healthy" in metrics
            assert "latency_ms" in metrics
            assert "success_rate" in metrics
            assert "health_score" in metrics
            assert metrics["health_score"] >= 0
            assert metrics["health_score"] <= 1
    
    @pytest.mark.asyncio
    async def test_market_data_gathering(self, smart_router):
        """Test concurrent market data gathering."""
        market_data = await smart_router._gather_market_data("BTC/USDT")
        
        assert len(market_data) > 0
        
        for exchange_name, data in market_data.items():
            assert "ticker" in data
            assert "order_book" in data
            assert "latency_ms" in data
            
            ticker = data["ticker"]
            assert "bid" in ticker
            assert "ask" in ticker
            assert ticker["bid"] > 0
            assert ticker["ask"] > ticker["bid"]
            
            order_book = data["order_book"]
            assert "bids" in order_book
            assert "asks" in order_book
            assert len(order_book["bids"]) > 0
            assert len(order_book["asks"]) > 0
    
    @pytest.mark.asyncio
    async def test_execution_cost_calculation(self, smart_router):
        """Test execution cost calculation."""
        # Create mock order book
        order_book = {
            "bids": [[49995, 1.0], [49990, 2.0], [49985, 3.0]],
            "asks": [[50005, 1.0], [50010, 2.0], [50015, 3.0]]
        }
        
        # Test buy order
        buy_cost = smart_router._calculate_execution_cost(order_book, "buy", 2.5)
        
        assert buy_cost["fillable_amount"] == 2.5
        assert buy_cost["average_price"] > 50005  # Should be higher due to slippage
        assert buy_cost["slippage_bps"] > 0
        assert buy_cost["liquidity_score"] > 0
        
        # Test sell order
        sell_cost = smart_router._calculate_execution_cost(order_book, "sell", 2.5)
        
        assert sell_cost["fillable_amount"] == 2.5
        assert sell_cost["average_price"] < 49995  # Should be lower due to slippage
        assert sell_cost["slippage_bps"] > 0
    
    @pytest.mark.asyncio
    async def test_routing_cache(self, smart_router, sample_order):
        """Test routing decision caching."""
        # First request
        start_time = datetime.utcnow()
        decision1 = await smart_router.route_order(sample_order)
        first_duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Second request (should use cache)
        start_time = datetime.utcnow()
        decision2 = await smart_router.route_order(sample_order)
        second_duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Cache should make second request faster
        assert second_duration < first_duration
        assert decision1.selected_exchange == decision2.selected_exchange
    
    @pytest.mark.asyncio
    async def test_slippage_limit_enforcement(self, smart_router):
        """Test slippage limit enforcement."""
        # Create order with very low slippage tolerance
        strict_order = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=100.0,  # Large order to cause slippage
            max_slippage_bps=1,  # Very strict limit
            execution_strategy="best_execution"
        )
        
        decision = await smart_router.route_order(strict_order)
        
        # Should still route but may have higher slippage
        assert isinstance(decision, RoutingDecision)
        assert decision.selected_exchange in smart_router.exchange_clients
    
    @pytest.mark.asyncio
    async def test_exchange_failure_handling(self, smart_router, sample_order):
        """Test handling of exchange failures."""
        # Make all exchanges except one fail
        for name, exchange in smart_router.exchange_clients.items():
            if name != "binance":
                exchange.success_rate = 0.0
        
        # Should still be able to route to healthy exchange
        decision = await smart_router.route_order(sample_order)
        assert decision.selected_exchange == "binance"
    
    @pytest.mark.asyncio
    async def test_latency_optimization(self, smart_router):
        """Test latency optimization features."""
        # Test that latency is being tracked
        await smart_router._gather_market_data("BTC/USDT")
        
        for exchange_name, metrics in smart_router.exchange_metrics.items():
            assert metrics.latency_ms > 0
            assert metrics.last_update is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_routing_requests(self, smart_router, sample_order):
        """Test handling of concurrent routing requests."""
        # Create multiple concurrent requests
        tasks = []
        for i in range(10):
            order = OrderRequest(
                symbol="BTC/USDT",
                side="buy" if i % 2 == 0 else "sell",
                type="market",
                amount=1.0 + i * 0.1,
                execution_strategy="best_execution"
            )
            tasks.append(smart_router.route_order(order))
        
        # Execute all requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All requests should succeed
        for result in results:
            assert isinstance(result, RoutingDecision)
            assert result.selected_exchange in smart_router.exchange_clients
    
    def test_exchange_metrics_update(self):
        """Test exchange metrics updating."""
        metrics = ExchangeMetrics("test_exchange")
        
        # Test latency updates
        initial_latency = metrics.latency_ms
        metrics.update_latency(200.0)
        assert metrics.latency_ms != initial_latency
        
        # Test success rate updates
        initial_success_rate = metrics.success_rate
        metrics.update_success_rate(False)
        assert metrics.success_rate < initial_success_rate
        assert metrics.consecutive_failures == 1
        
        # Test health status
        for _ in range(5):
            metrics.update_success_rate(False)
        
        assert not metrics.is_healthy
    
    @pytest.mark.asyncio
    async def test_preferred_exchanges(self, smart_router):
        """Test preferred exchanges functionality."""
        order_with_preference = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=1.0,
            preferred_exchanges=["coinbase", "kraken"],
            execution_strategy="best_execution"
        )
        
        decision = await smart_router.route_order(order_with_preference)
        
        # Should prefer one of the specified exchanges if they're healthy
        assert decision.selected_exchange in smart_router.exchange_clients


class TestIntegration:
    """Integration tests for the smart routing system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_routing_and_execution(self, smart_router, sample_order):
        """Test complete end-to-end routing and execution."""
        # Route the order
        decision = await smart_router.route_order(sample_order)
        
        # Execute with failover
        result = await smart_router.execute_with_failover(sample_order, decision)
        
        assert result["success"] is True
        assert "exchange" in result
        assert "result" in result
        
        execution_result = result["result"]
        assert "order_id" in execution_result
        assert execution_result["symbol"] == sample_order.symbol
        assert execution_result["side"] == sample_order.side
        assert execution_result["amount"] == sample_order.amount
    
    @pytest.mark.asyncio
    async def test_multiple_symbol_routing(self, smart_router):
        """Test routing for multiple trading symbols."""
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        
        for symbol in symbols:
            order = OrderRequest(
                symbol=symbol,
                side="buy",
                type="market",
                amount=1.0,
                execution_strategy="best_execution"
            )
            
            decision = await smart_router.route_order(order)
            assert isinstance(decision, RoutingDecision)
            assert decision.selected_exchange in smart_router.exchange_clients
    
    @pytest.mark.asyncio
    async def test_stress_testing(self, smart_router):
        """Stress test the routing system."""
        # Create many concurrent orders
        orders = []
        for i in range(50):
            order = OrderRequest(
                symbol="BTC/USDT",
                side="buy" if i % 2 == 0 else "sell",
                type="market",
                amount=0.1 + (i % 10) * 0.1,
                execution_strategy=["best_execution", "lowest_latency", "best_liquidity"][i % 3]
            )
            orders.append(order)
        
        # Route all orders concurrently
        start_time = datetime.utcnow()
        tasks = [smart_router.route_order(order) for order in orders]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Check results
        successful_routes = [r for r in results if isinstance(r, RoutingDecision)]
        assert len(successful_routes) >= len(orders) * 0.9  # At least 90% success rate
        
        # Should complete within reasonable time
        assert duration < 10.0  # Less than 10 seconds for 50 orders
        
        logger.info(f"Stress test completed: {len(successful_routes)}/{len(orders)} successful routes in {duration:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])