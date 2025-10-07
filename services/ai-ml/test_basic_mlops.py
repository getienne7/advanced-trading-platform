#!/usr/bin/env python3
"""
Basic test for enhanced MLOps pipeline functionality
"""
import asyncio
import tempfile
import shutil
from unittest.mock import Mock
from mlops_pipeline import MLOpsManager, ModelMetrics


async def test_basic_functionality():
    """Test basic MLOps functionality."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create temporary config
        config = Mock()
        config.price_model_path = temp_dir
        
        # Create MLOps manager
        manager = MLOpsManager(config)
        
        # Test model registration
        metrics = ModelMetrics(
            mse=0.08, mae=0.04, r2_score=0.88, accuracy=0.82,
            precision=0.78, recall=0.85, f1_score=0.81
        )
        
        model_version = await manager.register_model(
            model={'type': 'test'},
            model_type='lstm',
            symbol='BTC/USDT',
            metrics=metrics
        )
        
        print(f'✓ Model registered: {model_version.model_id}')
        print(f'✓ Model status: {model_version.status.value}')
        print(f'✓ Model accuracy: {model_version.metrics.accuracy}')
        
        # Test system health
        health = await manager.get_system_health()
        print(f'✓ System health: {health["status"]}')
        print(f'✓ Total models: {health["models"]["total"]}')
        
        # Test performance history (should be empty initially)
        history = await manager.get_model_performance_history(model_version.model_id)
        print(f'✓ Performance history entries: {len(history)}')
        
        # Test drift history (should be empty initially)
        drift_history = await manager.get_drift_history(model_version.model_id)
        print(f'✓ Drift history entries: {len(drift_history)}')
        
        # Test A/B test history (should be empty initially)
        ab_history = await manager.get_ab_test_history()
        print(f'✓ A/B test history entries: {len(ab_history)}')
        
        print('\n🎉 Enhanced MLOps pipeline test completed successfully!')
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    asyncio.run(test_basic_functionality())