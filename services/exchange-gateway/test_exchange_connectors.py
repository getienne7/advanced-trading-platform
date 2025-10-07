#!/usr/bin/env python3
"""
Comprehensive test suite for exchange connectors.
Tests all exchange implementations for compliance with the ExchangeInterface.
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import pytest
import structlog

# Add project paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "shared"))

from exchanges.base_exchange import ExchangeInterface
from exchanges.binance_connector import BinanceConnector
from exchanges.coinbase_connector import CoinbaseConnector
from exchanges.kraken_connector import KrakenConnector
from exchanges.demo_exchange import DemoExchange

logger = structlog.get_logger(__name__)


class ExchangeConnectorTester:
    """Test suite for exchange connectors."""
    
    def __init__(self):
        self.test_symbol = "BTC/USDT"
        self.test_amount = 0.001
        self.test_price = 50000.0
        
    async def test_exchange_interface_compliance(self, exchange: ExchangeInterface) -> Dict[str, Any]:
        """Test that an exchange implements all required interface methods."""
        results = {
            'exchange': exchange.name,
            'tests': {},
            'overall_status': 'pass',
            'errors': []
        }
        
        # Test initialization
        try:
            if not exchange.is_initialized:
                await exchange.initialize()
            results['tests']['initialization'] = 'pass'
        except Exception as e:
            results['tests']['initialization'] = f'fail: {str(e)}'
            results['errors'].append(f'Initialization failed: {str(e)}')
            results['overall_status'] = 'fail'
        
        # Test status check
        try:
            status = await exchange.get_status()
            assert isinstance(status, dict)
            results['tests']['get_status'] = 'pass'
        except Exception as e:
            results['tests']['get_status'] = f'fail: {str(e)}'
            results['errors'].append(f'Status check failed: {str(e)}')
            results['overall_status'] = 'fail'
        
        # Test exchange info
        try:
            info = await exchange.get_exchange_info()
            assert isinstance(info, dict)
            results['tests']['get_exchange_info'] = 'pass'
        except Exception as e:
            results['tests']['get_exchange_info'] = f'fail: {str(e)}'
            results['errors'].append(f'Exchange info failed: {str(e)}')
            results['overall_status'] = 'fail'
        
        # Test market data methods
        market_data_tests = [
            ('get_ticker', lambda: exchange.get_ticker(self.test_symbol)),
            ('get_order_book', lambda: exchange.get_order_book(self.test_symbol, 10)),
            ('get_trades', lambda: exchange.get_trades(self.test_symbol, 10)),
            ('get_klines', lambda: exchange.get_klines(self.test_symbol, '1h', limit=10))
        ]
        
        for test_name, test_func in market_data_tests:
            try:
                result = await test_func()
                assert result is not None
                results['tests'][test_name] = 'pass'
            except Exception as e:
                results['tests'][test_name] = f'fail: {str(e)}'
                results['errors'].append(f'{test_name} failed: {str(e)}')
                if results['overall_status'] != 'fail':
                    results['overall_status'] = 'warning'
        
        # Test account methods (may fail without credentials)
        account_tests = [
            ('get_balances', lambda: exchange.get_balances()),
            ('get_account_info', lambda: exchange.get_account_info())
        ]
        
        for test_name, test_func in account_tests:
            try:
                result = await test_func()
                assert result is not None
                results['tests'][test_name] = 'pass'
            except Exception as e:
                results['tests'][test_name] = f'skip: {str(e)}'
                # Don't mark as error since credentials might not be available
        
        # Test utility methods
        try:
            normalized = exchange.normalize_symbol(self.test_symbol)
            denormalized = exchange.denormalize_symbol(normalized)
            results['tests']['symbol_normalization'] = 'pass'
        except Exception as e:
            results['tests']['symbol_normalization'] = f'fail: {str(e)}'
            results['errors'].append(f'Symbol normalization failed: {str(e)}')
        
        # Test validation
        try:
            exchange.validate_order_params(self.test_symbol, 'buy', 'limit', self.test_amount, self.test_price)
            results['tests']['order_validation'] = 'pass'
        except Exception as e:
            results['tests']['order_validation'] = f'fail: {str(e)}'
            results['errors'].append(f'Order validation failed: {str(e)}')
        
        # Test trading rules
        try:
            rules = exchange.get_trading_rules(self.test_symbol)
            assert isinstance(rules, dict)
            results['tests']['trading_rules'] = 'pass'
        except Exception as e:
            results['tests']['trading_rules'] = f'fail: {str(e)}'
            results['errors'].append(f'Trading rules failed: {str(e)}')
        
        # Test health check
        try:
            health = await exchange.get_exchange_health()
            assert isinstance(health, dict)
            results['tests']['health_check'] = 'pass'
        except Exception as e:
            results['tests']['health_check'] = f'fail: {str(e)}'
            results['errors'].append(f'Health check failed: {str(e)}')
        
        return results
    
    async def test_advanced_features(self, exchange: ExchangeInterface) -> Dict[str, Any]:
        """Test advanced features if supported by the exchange."""
        results = {
            'exchange': exchange.name,
            'advanced_features': {},
            'supported_features': []
        }
        
        # Test futures trading (if supported)
        try:
            await exchange.place_futures_order(
                self.test_symbol, 'buy', 'limit', self.test_amount, self.test_price
            )
            results['advanced_features']['futures_trading'] = 'supported'
            results['supported_features'].append('futures_trading')
        except NotImplementedError:
            results['advanced_features']['futures_trading'] = 'not_supported'
        except Exception as e:
            results['advanced_features']['futures_trading'] = f'error: {str(e)}'
        
        # Test margin trading (if supported)
        try:
            await exchange.place_margin_order(
                self.test_symbol, 'buy', 'limit', self.test_amount, self.test_price
            )
            results['advanced_features']['margin_trading'] = 'supported'
            results['supported_features'].append('margin_trading')
        except NotImplementedError:
            results['advanced_features']['margin_trading'] = 'not_supported'
        except Exception as e:
            results['advanced_features']['margin_trading'] = f'error: {str(e)}'
        
        # Test advanced orders (if supported)
        try:
            await exchange.place_advanced_order(
                self.test_symbol, 'buy', 'limit', self.test_amount, self.test_price
            )
            results['advanced_features']['advanced_orders'] = 'supported'
            results['supported_features'].append('advanced_orders')
        except NotImplementedError:
            results['advanced_features']['advanced_orders'] = 'not_supported'
        except Exception as e:
            results['advanced_features']['advanced_orders'] = f'error: {str(e)}'
        
        # Test conditional orders (if supported)
        try:
            await exchange.place_conditional_order(
                self.test_symbol, 'buy', 'stop-loss', self.test_amount, self.test_price
            )
            results['advanced_features']['conditional_orders'] = 'supported'
            results['supported_features'].append('conditional_orders')
        except NotImplementedError:
            results['advanced_features']['conditional_orders'] = 'not_supported'
        except Exception as e:
            results['advanced_features']['conditional_orders'] = f'error: {str(e)}'
        
        return results
    
    async def test_performance_metrics(self, exchange: ExchangeInterface) -> Dict[str, Any]:
        """Test performance and latency of exchange operations."""
        import time
        
        results = {
            'exchange': exchange.name,
            'performance': {}
        }
        
        # Test ticker latency
        try:
            start_time = time.time()
            await exchange.get_ticker(self.test_symbol)
            latency = (time.time() - start_time) * 1000
            results['performance']['ticker_latency_ms'] = round(latency, 2)
        except Exception as e:
            results['performance']['ticker_latency_ms'] = f'error: {str(e)}'
        
        # Test order book latency
        try:
            start_time = time.time()
            await exchange.get_order_book(self.test_symbol, 20)
            latency = (time.time() - start_time) * 1000
            results['performance']['orderbook_latency_ms'] = round(latency, 2)
        except Exception as e:
            results['performance']['orderbook_latency_ms'] = f'error: {str(e)}'
        
        # Test multiple concurrent requests
        try:
            start_time = time.time()
            tasks = [
                exchange.get_ticker(self.test_symbol),
                exchange.get_order_book(self.test_symbol, 10),
                exchange.get_trades(self.test_symbol, 10)
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
            latency = (time.time() - start_time) * 1000
            results['performance']['concurrent_requests_latency_ms'] = round(latency, 2)
        except Exception as e:
            results['performance']['concurrent_requests_latency_ms'] = f'error: {str(e)}'
        
        return results
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test suite on all available exchanges."""
        exchanges_to_test = []
        
        # Always test demo exchange
        exchanges_to_test.append(DemoExchange())
        
        # Test real exchanges if credentials are available
        if os.getenv("BINANCE_API_KEY") and os.getenv("BINANCE_SECRET_KEY"):
            exchanges_to_test.append(BinanceConnector(
                api_key=os.getenv("BINANCE_API_KEY"),
                secret_key=os.getenv("BINANCE_SECRET_KEY"),
                testnet=True
            ))
        
        if (os.getenv("COINBASE_API_KEY") and 
            os.getenv("COINBASE_SECRET_KEY") and 
            os.getenv("COINBASE_PASSPHRASE")):
            exchanges_to_test.append(CoinbaseConnector(
                api_key=os.getenv("COINBASE_API_KEY"),
                secret_key=os.getenv("COINBASE_SECRET_KEY"),
                passphrase=os.getenv("COINBASE_PASSPHRASE"),
                sandbox=True
            ))
        
        if os.getenv("KRAKEN_API_KEY") and os.getenv("KRAKEN_SECRET_KEY"):
            exchanges_to_test.append(KrakenConnector(
                api_key=os.getenv("KRAKEN_API_KEY"),
                secret_key=os.getenv("KRAKEN_SECRET_KEY")
            ))
        
        test_results = {
            'test_summary': {
                'total_exchanges': len(exchanges_to_test),
                'passed': 0,
                'failed': 0,
                'warnings': 0
            },
            'exchange_results': {}
        }
        
        for exchange in exchanges_to_test:
            logger.info(f"Testing exchange: {exchange.name}")
            
            try:
                # Run interface compliance test
                compliance_result = await self.test_exchange_interface_compliance(exchange)
                
                # Run advanced features test
                advanced_result = await self.test_advanced_features(exchange)
                
                # Run performance test
                performance_result = await self.test_performance_metrics(exchange)
                
                # Combine results
                exchange_result = {
                    'compliance': compliance_result,
                    'advanced_features': advanced_result,
                    'performance': performance_result
                }
                
                test_results['exchange_results'][exchange.name] = exchange_result
                
                # Update summary
                if compliance_result['overall_status'] == 'pass':
                    test_results['test_summary']['passed'] += 1
                elif compliance_result['overall_status'] == 'warning':
                    test_results['test_summary']['warnings'] += 1
                else:
                    test_results['test_summary']['failed'] += 1
                
                # Cleanup
                await exchange.close()
                
            except Exception as e:
                logger.error(f"Failed to test exchange {exchange.name}: {str(e)}")
                test_results['exchange_results'][exchange.name] = {
                    'error': str(e),
                    'status': 'failed'
                }
                test_results['test_summary']['failed'] += 1
        
        return test_results
    
    def print_test_results(self, results: Dict[str, Any]):
        """Print formatted test results."""
        print("\n" + "="*80)
        print("EXCHANGE CONNECTOR TEST RESULTS")
        print("="*80)
        
        summary = results['test_summary']
        print(f"\nSUMMARY:")
        print(f"  Total Exchanges: {summary['total_exchanges']}")
        print(f"  Passed: {summary['passed']}")
        print(f"  Warnings: {summary['warnings']}")
        print(f"  Failed: {summary['failed']}")
        
        for exchange_name, exchange_result in results['exchange_results'].items():
            print(f"\n{'-'*60}")
            print(f"EXCHANGE: {exchange_name.upper()}")
            print(f"{'-'*60}")
            
            if 'error' in exchange_result:
                print(f"  ERROR: {exchange_result['error']}")
                continue
            
            # Compliance results
            compliance = exchange_result.get('compliance', {})
            print(f"  Interface Compliance: {compliance.get('overall_status', 'unknown').upper()}")
            
            if compliance.get('errors'):
                print(f"  Errors:")
                for error in compliance['errors']:
                    print(f"    - {error}")
            
            # Advanced features
            advanced = exchange_result.get('advanced_features', {})
            supported_features = advanced.get('supported_features', [])
            if supported_features:
                print(f"  Advanced Features: {', '.join(supported_features)}")
            else:
                print(f"  Advanced Features: None")
            
            # Performance
            performance = exchange_result.get('performance', {})
            if 'ticker_latency_ms' in performance:
                print(f"  Ticker Latency: {performance['ticker_latency_ms']} ms")
            if 'orderbook_latency_ms' in performance:
                print(f"  Order Book Latency: {performance['orderbook_latency_ms']} ms")
        
        print("\n" + "="*80)


async def main():
    """Run the comprehensive exchange connector test suite."""
    tester = ExchangeConnectorTester()
    
    print("Starting comprehensive exchange connector tests...")
    print("This will test all available exchange connectors for:")
    print("- Interface compliance")
    print("- Advanced features support")
    print("- Performance metrics")
    print("- Error handling")
    
    results = await tester.run_comprehensive_test()
    tester.print_test_results(results)
    
    # Return exit code based on results
    if results['test_summary']['failed'] > 0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)