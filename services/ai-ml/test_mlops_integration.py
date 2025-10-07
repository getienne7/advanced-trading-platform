"""
Integration tests for enhanced MLOps pipeline
Tests the complete MLOps workflow including automated training, deployment, monitoring, and A/B testing.
"""
import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock, patch
import json

from mlops_pipeline import (
    MLOpsManager, ModelMetrics, ModelVersion, ModelStatus,
    DriftDetectionResult, ABTestResult
)
from model_scheduler import ModelScheduler, ScheduleType, TriggerType
from mlops_service import MLOpsService
from mlops_config import MLOpsConfig


class TestEnhancedMLOpsPipeline:
    """Test enhanced MLOps pipeline functionality."""
    
    @pytest.fixture
    async def mlops_system(self):
        """Create complete MLOps system for testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Mock config
        config = Mock()
        config.price_model_path = temp_dir
        
        # Create components
        mlops_manager = MLOpsManager(config)
        scheduler = ModelScheduler(config, mlops_manager)
        service = MLOpsService(config)
        
        yield {
            'manager': mlops_manager,
            'scheduler': scheduler,
            'service': service,
            'config': config,
            'temp_dir': temp_dir
        }
        
        # Cleanup
        await scheduler.stop()
        shutil.rmtree(temp_dir)
    
    async def test_automated_model_training_pipeline(self, mlops_system):
        """Test complete automated model training pipeline."""
        manager = mlops_system['manager']
        scheduler = mlops_system['scheduler']
        
        # 1. Set up automated training job
        job_id = scheduler.add_training_job(
            model_type='lstm',
            symbol='BTC/USDT',
            schedule_type=ScheduleType.DAILY,
            schedule_config={'hour': 2, 'minute': 0},
            training_config={
                'data_window_days': 30,
                'validation_split': 0.2,
                'early_stopping_patience': 10
            },
            deployment_config={
                'auto_deploy': True,
                'deployment_strategy': 'blue_green',
                'require_approval': False
            }
        )
        
        # 2. Mock training components
        with patch.object(scheduler, '_get_training_data') as mock_data, \
             patch.object(scheduler, '_train_model') as mock_train, \
             patch.object(scheduler, '_validate_model') as mock_validate, \
             patch.object(manager, '_validate_model_for_deployment') as mock_deploy_validate:
            
            # Setup realistic training data
            training_data = pd.DataFrame({
                'open': np.random.uniform(45000, 55000, 1000),
                'high': np.random.uniform(45000, 55000, 1000),
                'low': np.random.uniform(45000, 55000, 1000),
                'close': np.random.uniform(45000, 55000, 1000),
                'volume': np.random.uniform(1000, 10000, 1000),
                'rsi': np.random.uniform(20, 80, 1000),
                'macd': np.random.normal(0, 0.1, 1000)
            })
            
            mock_data.return_value = training_data
            
            # Mock successful training
            mock_train.return_value = (
                {'model_type': 'lstm', 'trained_at': datetime.utcnow().isoformat()},
                ModelMetrics(
                    mse=0.05, mae=0.03, r2_score=0.92, accuracy=0.85,
                    precision=0.82, recall=0.88, f1_score=0.85
                )
            )
            
            mock_validate.return_value = True
            mock_deploy_validate.return_value = {'valid': True, 'reason': 'All validations passed'}
            
            # 3. Trigger training
            success = await scheduler.trigger_job(job_id, TriggerType.SCHEDULED)
            assert success is True
            
            # Wait for job completion
            await asyncio.sleep(0.5)
            
            # 4. Verify model was registered and deployed
            assert len(manager.model_registry) > 0
            
            # Find the registered model
            model_versions = []
            for versions in manager.model_registry.values():
                model_versions.extend(versions)
            
            assert len(model_versions) > 0
            model_version = model_versions[0]
            
            assert model_version.model_type == 'lstm'
            assert model_version.symbol == 'BTC/USDT'
            assert model_version.status == ModelStatus.DEPLOYED
            assert model_version.metrics.accuracy > 0.8
    
    async def test_model_performance_monitoring_and_drift_detection(self, mlops_system):
        """Test model performance monitoring and drift detection."""
        manager = mlops_system['manager']
        
        # 1. Register a model
        sample_model = {'type': 'lstm', 'symbol': 'BTC/USDT'}
        sample_metrics = ModelMetrics(
            mse=0.08, mae=0.04, r2_score=0.88, accuracy=0.82,
            precision=0.78, recall=0.85, f1_score=0.81
        )
        
        model_version = await manager.register_model(
            model=sample_model,
            model_type='lstm',
            symbol='BTC/USDT',
            metrics=sample_metrics,
            metadata={
                'training_stats': {
                    'feature1': {'mean': 0.5, 'std': 0.1},
                    'feature2': {'mean': 1.0, 'std': 0.2},
                    'rsi': {'mean': 50.0, 'std': 15.0},
                    'macd': {'mean': 0.0, 'std': 0.1}
                }
            }
        )
        
        # 2. Test normal data (no drift)
        normal_data = pd.DataFrame({
            'feature1': np.random.normal(0.5, 0.1, 100),
            'feature2': np.random.normal(1.0, 0.2, 100),
            'rsi': np.random.normal(50.0, 15.0, 100),
            'macd': np.random.normal(0.0, 0.1, 100)
        })
        
        with patch.object(manager, '_load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = np.random.normal(0, 0.05, 100)
            mock_load.return_value = mock_model
            
            drift_result = await manager.monitor_model_drift(
                model_version.model_id, normal_data
            )
            
            assert drift_result.drift_detected is False
            assert drift_result.recommendation == "NO_ACTION_NEEDED"
        
        # 3. Test drifted data
        drifted_data = pd.DataFrame({
            'feature1': np.random.normal(0.8, 0.15, 100),  # Drifted mean and std
            'feature2': np.random.normal(1.5, 0.3, 100),   # Drifted mean and std
            'rsi': np.random.normal(70.0, 20.0, 100),      # Drifted RSI
            'macd': np.random.normal(0.2, 0.2, 100)        # Drifted MACD
        })
        
        with patch.object(manager, '_load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = np.random.normal(0, 0.15, 100)  # Higher error
            mock_load.return_value = mock_model
            
            drift_result = await manager.monitor_model_drift(
                model_version.model_id, drifted_data
            )
            
            assert drift_result.drift_detected is True
            assert drift_result.recommendation in ["RETRAIN_IMMEDIATELY", "RETRAIN_SOON"]
            assert drift_result.drift_score > 0.1
    
    async def test_ab_testing_framework(self, mlops_system):
        """Test A/B testing framework for model comparison."""
        manager = mlops_system['manager']
        
        # 1. Register two models with different performance characteristics
        model_a = await manager.register_model(
            model={'type': 'lstm', 'units': 128, 'dropout': 0.2},
            model_type='lstm',
            symbol='BTC/USDT',
            metrics=ModelMetrics(
                mse=0.10, mae=0.06, r2_score=0.85, accuracy=0.78,
                precision=0.75, recall=0.82, f1_score=0.78
            )
        )
        
        model_b = await manager.register_model(
            model={'type': 'lstm', 'units': 256, 'dropout': 0.3},
            model_type='lstm',
            symbol='BTC/USDT',
            metrics=ModelMetrics(
                mse=0.08, mae=0.05, r2_score=0.88, accuracy=0.82,
                precision=0.80, recall=0.85, f1_score=0.82
            )
        )
        
        # 2. Start A/B test
        experiment_id = await manager.start_ab_test(
            model_a_id=model_a.model_id,
            model_b_id=model_b.model_id,
            test_name='LSTM Architecture Comparison',
            traffic_split=0.5,
            duration_hours=48
        )
        
        assert experiment_id in manager.active_experiments
        experiment = manager.active_experiments[experiment_id]
        assert experiment['status'] == 'running'
        
        # 3. Simulate test data collection (Model B performs better)
        np.random.seed(42)  # For reproducible results
        
        # Generate actual values
        actual_values = np.random.normal(100, 5, 2000)
        
        # Model A predictions (higher error)
        model_a_predictions = actual_values + np.random.normal(0, 3, 2000)
        
        # Model B predictions (lower error)
        model_b_predictions = actual_values + np.random.normal(0, 2, 2000)
        
        experiment['results']['actual_values'] = actual_values.tolist()
        experiment['results']['model_a_predictions'] = model_a_predictions.tolist()
        experiment['results']['model_b_predictions'] = model_b_predictions.tolist()
        experiment['results']['timestamps'] = [
            (datetime.utcnow() - timedelta(minutes=i)).isoformat()
            for i in range(2000)
        ]
        
        # 4. Evaluate A/B test
        with patch('mlops_pipeline.stats') as mock_stats:
            # Mock statistical test to show significant difference
            mock_stats.ttest_rel.return_value = (5.2, 0.001)  # Highly significant
            
            result = await manager.evaluate_ab_test(experiment_id)
            
            assert isinstance(result, ABTestResult)
            assert result.winner == model_b.model_id  # Model B should win
            assert result.statistical_significance < 0.05
            assert result.model_b_metrics.mse < result.model_a_metrics.mse
    
    async def test_automated_retraining_triggers(self, mlops_system):
        """Test automated retraining triggers."""
        manager = mlops_system['manager']
        scheduler = mlops_system['scheduler']
        
        # 1. Register and deploy a model
        model_version = await manager.register_model(
            model={'type': 'lstm'},
            model_type='lstm',
            symbol='BTC/USDT',
            metrics=ModelMetrics(
                mse=0.08, mae=0.04, r2_score=0.88, accuracy=0.82,
                precision=0.78, recall=0.85, f1_score=0.81
            ),
            metadata={
                'training_stats': {
                    'feature1': {'mean': 0.5, 'std': 0.1},
                    'feature2': {'mean': 1.0, 'std': 0.2}
                }
            }
        )
        
        # Deploy the model
        with patch.object(manager, '_validate_model_for_deployment') as mock_validate:
            mock_validate.return_value = {'valid': True, 'reason': 'All validations passed'}
            await manager.deploy_model(model_version.model_id)
        
        # 2. Set up training job for the model
        job_id = scheduler.add_training_job(
            model_type='lstm',
            symbol='BTC/USDT',
            schedule_type=ScheduleType.DAILY,
            schedule_config={'hour': 2}
        )
        
        # 3. Test performance degradation trigger
        with patch.object(manager, 'get_model_performance_history') as mock_history:
            # Mock degraded performance history
            mock_history.return_value = [
                {'timestamp': datetime.utcnow().isoformat(), 'mse': 0.15},
                {'timestamp': datetime.utcnow().isoformat(), 'mse': 0.16},
                {'timestamp': datetime.utcnow().isoformat(), 'mse': 0.17},
                {'timestamp': datetime.utcnow().isoformat(), 'mse': 0.18},
                {'timestamp': datetime.utcnow().isoformat(), 'mse': 0.19}
            ]
            
            # Check performance degradation
            await scheduler._check_model_performance_degradation(model_version)
            
            # Should trigger retraining (mocked)
            mock_history.assert_called_once()
        
        # 4. Test drift-based trigger
        drifted_data = pd.DataFrame({
            'feature1': np.random.normal(0.9, 0.2, 100),  # Significant drift
            'feature2': np.random.normal(1.8, 0.4, 100)   # Significant drift
        })
        
        with patch.object(manager, '_load_model') as mock_load, \
             patch.object(scheduler, '_get_recent_market_data') as mock_recent_data:
            
            mock_model = Mock()
            mock_model.predict.return_value = np.random.normal(0, 0.2, 100)
            mock_load.return_value = mock_model
            mock_recent_data.return_value = drifted_data
            
            # Check drift
            await scheduler._check_model_drift(model_version)
            
            mock_recent_data.assert_called_once()
    
    async def test_model_lifecycle_management(self, mlops_system):
        """Test complete model lifecycle management."""
        manager = mlops_system['manager']
        
        # 1. Register multiple model versions
        models = []
        for i in range(3):
            model = await manager.register_model(
                model={'type': 'lstm', 'version': i},
                model_type='lstm',
                symbol='BTC/USDT',
                metrics=ModelMetrics(
                    mse=0.1 - i * 0.01,  # Each version gets better
                    mae=0.06 - i * 0.005,
                    r2_score=0.85 + i * 0.02,
                    accuracy=0.78 + i * 0.02,
                    precision=0.75 + i * 0.02,
                    recall=0.82 + i * 0.01,
                    f1_score=0.78 + i * 0.02
                )
            )
            models.append(model)
            
            # Small delay between registrations
            await asyncio.sleep(0.1)
        
        # 2. Deploy the best model (latest)
        with patch.object(manager, '_validate_model_for_deployment') as mock_validate:
            mock_validate.return_value = {'valid': True, 'reason': 'All validations passed'}
            
            success = await manager.deploy_model(models[-1].model_id)
            assert success is True
            assert models[-1].status == ModelStatus.DEPLOYED
        
        # 3. Test model promotion workflow
        # Register a new model in validation
        new_model = await manager.register_model(
            model={'type': 'lstm', 'version': 'new'},
            model_type='lstm',
            symbol='BTC/USDT',
            metrics=ModelMetrics(
                mse=0.06, mae=0.03, r2_score=0.92, accuracy=0.88,
                precision=0.85, recall=0.90, f1_score=0.87
            )
        )
        
        assert new_model.status == ModelStatus.VALIDATION
        
        # 4. Test system health monitoring
        health = await manager.get_system_health()
        
        assert health['status'] in ['healthy', 'degraded', 'unhealthy']
        assert 'models' in health
        assert 'experiments' in health
        assert 'storage' in health
        assert health['models']['total'] == 4  # 3 + 1 new model
        assert health['models']['by_status']['deployed'] == 1
        assert health['models']['by_status']['validation'] == 1
        assert health['models']['by_status']['deprecated'] >= 0
    
    async def test_mlops_service_api_endpoints(self, mlops_system):
        """Test MLOps service API endpoints."""
        service = mlops_system['service']
        manager = mlops_system['manager']
        
        # Register a test model first
        model_version = await manager.register_model(
            model={'type': 'test'},
            model_type='lstm',
            symbol='BTC/USDT',
            metrics=ModelMetrics(
                mse=0.08, mae=0.04, r2_score=0.88, accuracy=0.82,
                precision=0.78, recall=0.85, f1_score=0.81
            )
        )
        
        # Test model info endpoint
        model_info = service.mlops_manager._find_model_version(model_version.model_id)
        assert model_info is not None
        assert model_info.model_id == model_version.model_id
        
        # Test system health endpoint
        health = await manager.get_system_health()
        assert 'status' in health
        assert 'models' in health
        
        # Test performance history (empty initially)
        history = await manager.get_model_performance_history(model_version.model_id)
        assert isinstance(history, list)
        
        # Test drift history (empty initially)
        drift_history = await manager.get_drift_history(model_version.model_id)
        assert isinstance(drift_history, list)
    
    async def test_concurrent_operations(self, mlops_system):
        """Test concurrent MLOps operations."""
        manager = mlops_system['manager']
        
        # Test concurrent model registrations
        tasks = []
        for i in range(5):
            task = manager.register_model(
                model={'type': 'test', 'id': i},
                model_type='lstm',
                symbol=f'TEST{i}/USDT',
                metrics=ModelMetrics(
                    mse=0.1 + i * 0.01, mae=0.05 + i * 0.005,
                    r2_score=0.85 - i * 0.01, accuracy=0.8 - i * 0.01,
                    precision=0.75, recall=0.8, f1_score=0.77
                )
            )
            tasks.append(task)
        
        # Wait for all registrations
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert len(manager.model_registry) == 5
        
        # Verify all models were registered correctly
        for i, model_version in enumerate(results):
            assert model_version.symbol == f'TEST{i}/USDT'
            assert model_version.status == ModelStatus.VALIDATION
    
    async def test_error_handling_and_recovery(self, mlops_system):
        """Test error handling and recovery mechanisms."""
        manager = mlops_system['manager']
        scheduler = mlops_system['scheduler']
        
        # Test invalid model registration
        with pytest.raises(Exception):
            await manager.register_model(
                model=None,  # Invalid model
                model_type='lstm',
                symbol='BTC/USDT',
                metrics=ModelMetrics(
                    mse=0.1, mae=0.05, r2_score=0.85, accuracy=0.8,
                    precision=0.75, recall=0.8, f1_score=0.77
                )
            )
        
        # Test deployment of non-existent model
        success = await manager.deploy_model('non_existent_model_id')
        assert success is False
        
        # Test drift monitoring with invalid model
        with pytest.raises(ValueError):
            await manager.monitor_model_drift(
                'non_existent_model_id',
                pd.DataFrame({'feature1': [1, 2, 3]})
            )
        
        # Test scheduler with invalid job
        success = await scheduler.trigger_job('non_existent_job_id')
        assert success is False
        
        # Test A/B test with invalid models
        with pytest.raises(ValueError):
            await manager.start_ab_test(
                'non_existent_model_a',
                'non_existent_model_b',
                'Invalid Test'
            )


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])