"""Demo of Smart Order Routing System."""
import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock

# Import our smart router components
import sys
import os
sys.path.append(os.path.dirname(__file__))

from smart_order_router import (
    SmartOrderRouter, SmartOrderRequest, RoutingStrategy, 
    ExecutionAlgorithm, OrderSide, OrderType
)


class MockExchangeManager:
    """Mock exchange manager for demo."""
    
    def get_active_exchanges(self):
        return ["binance", "coinbase", "kraken"]


async def demo_smart_order_routing():
    """Demonstrate smart order routing capabilities."""
    print("=== Smart Order Routing System Demo ===\\n")
    
    # Create mock exchange manager
    exchange_manager = MockExchangeManager()
    
    # Create smart order router
    router = SmartOrderRouter(exchange_manager)
    
    print("Initialized Smart Order Router with venues:")
    for venue_name, venue in router.venue_selector.venues.items():
        print(f"  {venue_name}: Health={venue.health_score:.1f}, "
              f"Latency={venue.latency_ms}ms, "
              f"Fees={venue.maker_fee}/{venue.taker_fee}%")
    
    print("\\n" + "="*60)
    
    # Demo 1: Best Price Strategy
    print("\\nDemo 1: Best Price Strategy")
    print("-" * 30)
    
    request1 = SmartOrderRequest(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=Decimal("1.0"),
        order_type=OrderType.MARKET,
        strategy=RoutingStrategy.BEST_PRICE,
        algorithm=ExecutionAlgorithm.IMMEDIATE
    )
    
    report1 = await router.route_order(request1)
    print(f"Order: Buy {request1.quantity} {request1.symbol}")
    print(f"Strategy: {request1.strategy}")
    print(f"Result: {report1.status}")
    print(f"Filled: {report1.filled_quantity} at avg price ${report1.avg_fill_price}")
    print(f"Venues used: {', '.join(report1.venues_used)}")
    print(f"Execution time: {report1.execution_time:.3f}s")
    print(f"Total fees: ${report1.total_fees}")
    
    # Demo 2: Lowest Cost Strategy
    print("\\nDemo 2: Lowest Cost Strategy")
    print("-" * 30)
    
    request2 = SmartOrderRequest(
        symbol="ETHUSDT",
        side=OrderSide.SELL,
        quantity=Decimal("10.0"),
        order_type=OrderType.MARKET,
        strategy=RoutingStrategy.LOWEST_COST,
        algorithm=ExecutionAlgorithm.IMMEDIATE
    )
    
    report2 = await router.route_order(request2)
    print(f"Order: Sell {request2.quantity} {request2.symbol}")
    print(f"Strategy: {request2.strategy}")
    print(f"Result: {report2.status}")
    print(f"Filled: {report2.filled_quantity} at avg price ${report2.avg_fill_price}")
    print(f"Venues used: {', '.join(report2.venues_used)}")
    print(f"Total cost: ${report2.total_cost}")
    print(f"Total fees: ${report2.total_fees}")
    
    # Demo 3: Smart Split Strategy
    print("\\nDemo 3: Smart Split Strategy")
    print("-" * 30)
    
    request3 = SmartOrderRequest(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=Decimal("5.0"),
        order_type=OrderType.MARKET,
        strategy=RoutingStrategy.SMART_SPLIT,
        algorithm=ExecutionAlgorithm.IMMEDIATE,
        max_participation_rate=0.3
    )
    
    report3 = await router.route_order(request3)
    print(f"Order: Buy {request3.quantity} {request3.symbol}")
    print(f"Strategy: {request3.strategy}")
    print(f"Result: {report3.status}")
    print(f"Filled: {report3.filled_quantity} at avg price ${report3.avg_fill_price}")
    print(f"Venues used: {', '.join(report3.venues_used)}")
    print(f"Routing decisions: {len(report3.routing_decisions)}")
    
    for i, decision in enumerate(report3.routing_decisions):
        print(f"  Decision {i+1}: {decision.quantity} on {decision.venue} "
              f"(confidence: {decision.confidence_score:.2f})")
        print(f"    Reasoning: {decision.reasoning}")
    
    # Demo 4: Fastest Execution Strategy
    print("\\nDemo 4: Fastest Execution Strategy")
    print("-" * 30)
    
    request4 = SmartOrderRequest(
        symbol="ADAUSDT",
        side=OrderSide.BUY,
        quantity=Decimal("1000.0"),
        order_type=OrderType.MARKET,
        strategy=RoutingStrategy.FASTEST_EXECUTION,
        algorithm=ExecutionAlgorithm.IMMEDIATE,
        urgency=1.0  # Maximum urgency
    )
    
    report4 = await router.route_order(request4)
    print(f"Order: Buy {request4.quantity} {request4.symbol}")
    print(f"Strategy: {request4.strategy}")
    print(f"Result: {report4.status}")
    print(f"Filled: {report4.filled_quantity} at avg price ${report4.avg_fill_price}")
    print(f"Venues used: {', '.join(report4.venues_used)}")
    print(f"Execution time: {report4.execution_time:.3f}s")
    
    # Demo 5: Liquidity Seeking Strategy
    print("\\nDemo 5: Liquidity Seeking Strategy")
    print("-" * 30)
    
    request5 = SmartOrderRequest(
        symbol="ETHUSDT",
        side=OrderSide.SELL,
        quantity=Decimal("50.0"),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("3000"),
        strategy=RoutingStrategy.LIQUIDITY_SEEKING,
        algorithm=ExecutionAlgorithm.IMMEDIATE
    )
    
    report5 = await router.route_order(request5)
    print(f"Order: Sell {request5.quantity} {request5.symbol} @ ${request5.limit_price}")
    print(f"Strategy: {request5.strategy}")
    print(f"Result: {report5.status}")
    print(f"Filled: {report5.filled_quantity} at avg price ${report5.avg_fill_price}")
    print(f"Venues used: {', '.join(report5.venues_used)}")
    print(f"Slippage: {report5.slippage_pct:.3f}%")
    
    # Show overall statistics
    print("\\n" + "="*60)
    print("\\nOverall Execution Statistics")
    print("-" * 30)
    
    stats = router.get_execution_statistics()
    print(f"Total orders executed: {stats['total_orders']}")
    print(f"Successful orders: {stats['successful_orders']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")
    print(f"Average execution time: {stats['avg_execution_time']:.3f}s")
    print(f"Average slippage: {stats['avg_slippage_pct']:.3f}%")
    print(f"Total volume: {stats['total_volume']}")
    print(f"Total fees: ${stats['total_fees']}")
    
    print("\\nVenue usage:")
    for venue, count in stats['venue_usage'].items():
        print(f"  {venue}: {count} orders")
    
    # Show venue health scores
    print("\\nVenue Health Scores:")
    for venue_name, venue in router.venue_selector.venues.items():
        print(f"  {venue_name}: {venue.health_score:.1f}/100 "
              f"(Success rate: {venue.success_rate:.1%}, "
              f"Recent failures: {venue.recent_failures})")
    
    print("\\n" + "="*60)
    print("\\nDemo completed! The Smart Order Router demonstrated:")
    print("1. Best Price routing - finds the best available price")
    print("2. Lowest Cost routing - considers fees and slippage")
    print("3. Smart Split routing - intelligently splits across venues")
    print("4. Fastest Execution routing - prioritizes speed")
    print("5. Liquidity Seeking routing - finds the most liquid venue")
    print("\\nThe system adapts to venue performance and market conditions")
    print("to optimize execution quality.")


def demo_venue_performance_tracking():
    """Demonstrate venue performance tracking."""
    print("\\n=== Venue Performance Tracking Demo ===\\n")
    
    exchange_manager = MockExchangeManager()
    router = SmartOrderRouter(exchange_manager)
    
    # Simulate some venue performance updates
    print("Simulating venue performance updates...")
    
    # Binance performs well
    for i in range(10):
        router.venue_selector.update_venue_performance("binance", 0.5, True)
    
    # Coinbase has some failures
    for i in range(7):
        router.venue_selector.update_venue_performance("coinbase", 1.2, True)
    for i in range(3):
        router.venue_selector.update_venue_performance("coinbase", 5.0, False)
    
    # Kraken is slow but reliable
    for i in range(8):
        router.venue_selector.update_venue_performance("kraken", 2.0, True)
    
    print("\\nUpdated venue health scores:")
    for venue_name, venue in router.venue_selector.venues.items():
        print(f"{venue_name}:")
        print(f"  Health Score: {venue.health_score:.1f}/100")
        print(f"  Success Rate: {venue.success_rate:.1%}")
        print(f"  Avg Execution Time: {venue.avg_execution_time:.2f}s")
        print(f"  Recent Failures: {venue.recent_failures}")
        print(f"  Is Healthy: {venue.is_healthy}")
        print()
    
    # Show healthy venues
    healthy_venues = router.venue_selector.get_healthy_venues()
    print(f"Healthy venues: {[v.name for v in healthy_venues]}")


if __name__ == "__main__":
    # Run the demos
    asyncio.run(demo_smart_order_routing())
    demo_venue_performance_tracking()