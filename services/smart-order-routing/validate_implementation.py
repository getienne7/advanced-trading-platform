#!/usr/bin/env python3
"""
Validation script for Smart Order Routing System implementation.
Verifies compliance with Requirements 2.5, 6.4, and 6.5.
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add shared directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "shared"))

from main import SmartOrderRouter, OrderRequest, RoutingReason
from utils import setup_logging

logger = setup_logging("validation")


class MockValidationExchange:
    """Simple mock exchange for validation."""
    
    def __init__(self, name: str, latency_ms: float = 100, price_offset: float = 0):
        self.name = name
        self.latency_ms = latency_ms
        self.price_offset = price_offset
        self.is_healthy = True
    
    async def get_status(self):
        await asyncio.sleep(self.latency_ms / 1000)
        if not self.is_healthy:
            raise Exception(f"{self.name} is down")
        return {"status": "online"}
    
    async def get_ticker(self, symbol: str):
        await asyncio.sleep(self.latency_ms / 1000)
        if not self.is_healthy:
            raise Exception(f"{self.name} ticker unavailable")
        
        base_price = 50000
        return {
            "bid": base_price + self.price_offset - 5,
            "ask": base_price + self.price_offset + 5,
            "last": base_price + self.price_offset,
            "volume": 1000.0
        }
    
    async def get_order_book(self, symbol: str, limit: int = 100):
        await asyncio.sleep(self.latency_ms / 1000)
        if not self.is_healthy:
            raise Exception(f"{self.name} order book unavailable")
        
        base_price = 50000 + self.price_offset
        bids = [[base_price - 5 - i, 10 - i] for i in range(10)]
        asks = [[base_price + 5 + i, 10 - i] for i in range(10)]
        
        return {"bids": bids, "asks": asks}
    
    async def place_order(self, **kwargs):
        await asyncio.sleep(self.latency_ms / 1000)
        if not self.is_healthy:
            raise Exception(f"{self.name} order placement failed")
        
        return {
            "order_id": f"{self.name}_order_{int(datetime.utcnow().timestamp())}",
            "status": "filled",
            **kwargs
        }


async def validate_requirement_25():
    """Validate Requirement 2.5: Route orders to exchange with best execution."""
    logger.info("Validating Requirement 2.5: Best execution routing")
    
    # Create exchanges with different characteristics
    exchanges = {
        "best_price": MockValidationExchange("best_price", latency_ms=80, price_offset=-10),
        "high_liquidity": MockValidationExchange("high_liquidity", latency_ms=120, price_offset=5),
        "fast_exchange": MockValidationExchange("fast_exchange", latency_ms=30, price_offset=0)
    }
    
    router = SmartOrderRouter()
    await router.initialize(exchanges)
    await asyncio.sleep(0.2)  # Wait for health checks
    
    # Test best execution routing
    order = OrderRequest(
        symbol="BTC/USDT",
        side="buy",
        type="market",
        amount=1.0,
        execution_strategy="best_execution"
    )
    
    decision = await router.route_order(order)
    
    # Validate routing decision
    assert decision.selected_exchange in exchanges, "Selected exchange must be valid"
    assert decision.reason == RoutingReason.BEST_PRICE, "Should use best price reasoning"
    assert decision.confidence > 0, "Must have positive confidence"
    assert decision.expected_price > 0, "Must have valid expected price"
    assert len(decision.alternatives) > 0, "Must provide alternatives"
    
    logger.info(f"✓ Requirement 2.5 validated: Routed to {decision.selected_exchange} with {decision.confidence:.2f} confidence")
    return True


async def validate_requirement_64():
    """Validate Requirement 6.4: Automatic failover when exchange API fails."""
    logger.info("Validating Requirement 6.4: Automatic failover")
    
    exchanges = {
        "primary": MockValidationExchange("primary", latency_ms=50),
        "backup1": MockValidationExchange("backup1", latency_ms=100),
        "backup2": MockValidationExchange("backup2", latency_ms=150)
    }
    
    router = SmartOrderRouter()
    await router.initialize(exchanges)
    await asyncio.sleep(0.2)
    
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
    
    # Simulate primary exchange failure
    exchanges[primary_exchange].is_healthy = False
    
    # Test failover execution
    result = await router.execute_with_failover(order, decision)
    
    # Validate failover behavior
    assert result["success"] is True, "Failover execution must succeed"
    assert result["failover_used"] is True, "Must indicate failover was used"
    assert result["exchange"] != primary_exchange, "Must use different exchange"
    assert "primary_error" in result, "Must report primary error"
    
    logger.info(f"✓ Requirement 6.4 validated: Failed over from {primary_exchange} to {result['exchange']}")
    return True


async def validate_requirement_65():
    """Validate Requirement 6.5: Select exchange with best price and liquidity."""
    logger.info("Validating Requirement 6.5: Best price and liquidity selection")
    
    exchanges = {
        "best_price": MockValidationExchange("best_price", latency_ms=100, price_offset=-20),  # Best price
        "best_liquidity": MockValidationExchange("best_liquidity", latency_ms=150, price_offset=10),  # Worse price, better liquidity
        "balanced": MockValidationExchange("balanced", latency_ms=80, price_offset=0)  # Balanced
    }
    
    router = SmartOrderRouter()
    await router.initialize(exchanges)
    await asyncio.sleep(0.2)
    
    # Test small order (should prefer price)
    small_order = OrderRequest(
        symbol="BTC/USDT",
        side="buy",
        type="market",
        amount=0.5,
        execution_strategy="best_execution"
    )
    
    small_decision = await router.route_order(small_order)
    
    # Test large order (should consider liquidity more)
    large_order = OrderRequest(
        symbol="BTC/USDT",
        side="buy",
        type="market",
        amount=5.0,
        execution_strategy="best_liquidity"
    )
    
    large_decision = await router.route_order(large_order)
    
    # Validate price/liquidity selection
    assert small_decision.selected_exchange in exchanges, "Small order routing must be valid"
    assert large_decision.selected_exchange in exchanges, "Large order routing must be valid"
    assert large_decision.reason == RoutingReason.BEST_LIQUIDITY, "Large order should prioritize liquidity"
    
    # Both should have positive liquidity scores
    assert small_decision.liquidity_score >= 0, "Must have valid liquidity score"
    assert large_decision.liquidity_score >= 0, "Must have valid liquidity score"
    
    logger.info(f"✓ Requirement 6.5 validated: Small order → {small_decision.selected_exchange}, Large order → {large_decision.selected_exchange}")
    return True


async def validate_latency_optimization():
    """Validate latency optimization for high-frequency trading."""
    logger.info("Validating latency optimization features")
    
    exchanges = {
        "ultra_fast": MockValidationExchange("ultra_fast", latency_ms=25),
        "fast": MockValidationExchange("fast", latency_ms=75),
        "slow": MockValidationExchange("slow", latency_ms=200)
    }
    
    router = SmartOrderRouter()
    await router.initialize(exchanges)
    await asyncio.sleep(0.2)
    
    # Test latency-optimized routing
    hft_order = OrderRequest(
        symbol="BTC/USDT",
        side="buy",
        type="market",
        amount=0.1,  # Small HFT-style order
        execution_strategy="lowest_latency"
    )
    
    decision = await router.route_order(hft_order)
    
    # Validate latency optimization
    assert decision.reason == RoutingReason.LOWEST_LATENCY, "Should use latency reasoning"
    assert decision.latency_ms < 100, "Should select low-latency exchange"
    assert decision.selected_exchange in ["ultra_fast", "fast"], "Should select fast exchange"
    
    logger.info(f"✓ Latency optimization validated: Selected {decision.selected_exchange} with {decision.latency_ms}ms latency")
    return True


async def validate_exchange_health_monitoring():
    """Validate exchange health monitoring and status reporting."""
    logger.info("Validating exchange health monitoring")
    
    exchanges = {
        "healthy": MockValidationExchange("healthy", latency_ms=50),
        "degraded": MockValidationExchange("degraded", latency_ms=300),
    }
    
    router = SmartOrderRouter()
    await router.initialize(exchanges)
    await asyncio.sleep(0.5)  # Wait for health monitoring
    
    # Get exchange status
    status = router.get_exchange_status()
    
    # Validate status reporting
    assert len(status) == len(exchanges), "Must report status for all exchanges"
    
    for exchange_name, metrics in status.items():
        assert "is_healthy" in metrics, "Must report health status"
        assert "latency_ms" in metrics, "Must report latency"
        assert "latency_category" in metrics, "Must categorize latency"
        assert "success_rate" in metrics, "Must report success rate"
        assert "health_score" in metrics, "Must calculate health score"
        
        # Validate latency categorization
        assert metrics["latency_category"] in ["excellent", "good", "acceptable", "poor", "unacceptable"]
        
        # Health score should be between 0 and 1
        assert 0 <= metrics["health_score"] <= 1, "Health score must be normalized"
    
    logger.info("✓ Exchange health monitoring validated")
    return True


async def run_validation():
    """Run all validation tests."""
    logger.info("Starting Smart Order Routing System validation...")
    
    validations = [
        ("Requirement 2.5 - Best Execution Routing", validate_requirement_25),
        ("Requirement 6.4 - Automatic Failover", validate_requirement_64),
        ("Requirement 6.5 - Price and Liquidity Selection", validate_requirement_65),
        ("Latency Optimization", validate_latency_optimization),
        ("Exchange Health Monitoring", validate_exchange_health_monitoring),
    ]
    
    results = {}
    
    for name, validation_func in validations:
        try:
            logger.info(f"Running validation: {name}")
            result = await validation_func()
            results[name] = "PASSED" if result else "FAILED"
            
        except Exception as e:
            logger.error(f"Validation failed: {name}", error=str(e))
            results[name] = f"FAILED: {str(e)}"
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SMART ORDER ROUTING VALIDATION SUMMARY")
    logger.info("="*60)
    
    passed = 0
    total = len(results)
    
    for name, result in results.items():
        status_symbol = "✓" if result == "PASSED" else "✗"
        logger.info(f"{status_symbol} {name}: {result}")
        if result == "PASSED":
            passed += 1
    
    logger.info("="*60)
    logger.info(f"OVERALL RESULT: {passed}/{total} validations passed")
    
    if passed == total:
        logger.info("🎉 ALL REQUIREMENTS VALIDATED SUCCESSFULLY!")
        logger.info("\nThe Smart Order Routing System fully implements:")
        logger.info("  • Requirement 2.5: Best execution algorithm")
        logger.info("  • Requirement 6.4: Automatic failover system")
        logger.info("  • Requirement 6.5: Price and liquidity optimization")
        logger.info("  • Latency optimization for high-frequency trading")
        logger.info("  • Comprehensive health monitoring and metrics")
    else:
        logger.warning(f"⚠️  {total - passed} validation(s) failed - review implementation")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_validation())
    sys.exit(0 if success else 1)