"""
Test suite for MLOps Pipeline
Comprehensive tests for model management, deployment, monitoring, and A/B testing.
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

from mlops_pipeline import (
    MLOpsManager, ModelMetrics, ModelVersion, ModelStatus,
    DriftDetectionResult, ABTestResult, ExperimentType
)
from model_scheduler import ModelScheduler, TrainingJob, ScheduleType, TriggerType
from mlops_config import MLOpsConfig


class TestMLOpsManager:
    """Test MLOps Manager functionality."""
    
    @pytest.fixture
    async def mlops_manager(self):
        """Create MLOps manager for testing."""
        # Create temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        
        # Mock config
        config = Mock()
        config.price_model_path = temp_dir
        
        manager = MLOpsManager(config)
        yield manager
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_metrics(self):
        """Sample model metrics for testing."""
        return ModelMetrics(
            mse=0.1,
            mae=0.05,
            r2_score=0.85,
            accuracy=0.75,
            precision=0.7,
            recall=0.8,
            f1_score=0.75
        )
    
    @pytest.fixture
    def sample_model(self):
        """Sample model for testing."""
        return {
            'type': 'lstm',
            'symbol': 'BTC/USDT',
            'parameters': {'units': 128, 'dropout': 0.2}
        }
    
    async def test_register_model(self, mlops_manager, sample_model, sample_metrics):
        """Test model registration."""
        model_version = await mlops_manager.register_model(
            model=sample_model,
            model_type='lstm',
            symbol='BTC/USDT',
            metrics=sample_metrics,
            metadata={'test': True}
        )
        
        assert model_version.model_type == 'lstm'
        assert model_version.symbol == 'BTC/USDT'
        assert model_version.status == ModelStatus.VALIDATION
        assert model_version.metrics.mse == 0.1
        assert model_version.metadata['test'] is True
        
        # Check if model is in registry
        registry_key = 'lstm_BTC/USDT'
        assert registry_key in mlops_manager.model_registry
        assert len(mlops_manager.model_registry[registry_key]) == 1
    
    async def test_deploy_model(self, mlops_manager, sample_model, sample_metrics):
        """Test model deployment."""
        # Register model first
        model_version = await mlops_manager.register_model(
            model=sample_model,
            model_type='lstm',
            symbol='BTC/USDT',
            metrics=sample_metrics
        )
        
        # Mock validation to pass
        with patch.object(mlops_manager, '_validate_model_for_deployment') as mock_validate:
            mock_validate.return_value = {'valid': True, 'reason': 'All validations passed'}
            
            # Deploy model
            success = await mlops_manager.deploy_model(model_version.model_id)
            
            assert success is True
            assert model_version.status == ModelStatus.DEPLOYED
            assert model_version.deployed_at is not None
    
    async def test_monitor_model_drift(self, mlops_manager, sample_model, sample_metrics):
        """Test model drift monitoring."""
        # Register model first
        model_version = await mlops_manager.register_model(
            model=sample_model,
            model_type='lstm',
            symbol='BTC/USDT',
            metrics=sample_metrics,
            metadata={
                'training_stats': {
                    'feature1': {'mean': 0.5, 'std': 0.1},
                    'feature2': {'mean': 1.0, 'std': 0.2}
                }
            }
        )
        
        # Create recent data with drift
        recent_data = pd.DataFrame({
            'feature1': [0.8, 0.9, 1.0, 1.1, 1.2],  # Drifted from 0.5
            'feature2': [1.5, 1.6, 1.7, 1.8, 1.9]   # Drifted from 1.0
        })
        
        # Mock model loading
        with patch.object(mlops_manager, '_load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            mock_load.return_value = mock_model
            
            drift_result = await mlops_manager.monitor_model_drift(
                model_id=model_version.model_id,
                recent_data=recent_data
            )
            
            assert isinstance(drift_result, DriftDetectionResult)
            assert drift_result.model_id == model_version.model_id
            assert drift_result.drift_score > 0
            assert 'feature1' in drift_result.feature_drifts
            assert 'feature2' in drift_result.feature_drifts
    
    async def test_start_ab_test(self, mlops_manager, sample_model, sample_metrics):
        """Test A/B test initialization."""
        # Register two models
        model_a = await mlops_manager.register_model(
            model=sample_model,
            model_type='lstm',
            symbol='BTC/USDT',
            metrics=sample_metrics
        )
        
        model_b = await mlops_manager.register_model(
            model={**sample_model, 'parameters': {'units': 256, 'dropout': 0.3}},
            model_type='lstm',
            symbol='BTC/USDT',
            metrics=ModelMetrics(
                mse=0.08, mae=0.04, r2_score=0.88, accuracy=0.78,
                precision=0.72, recall=0.82, f1_score=0.77
            )
        )
        
        # Start A/B test
        experiment_id = await mlops_manager.start_ab_test(
            model_a_id=model_a.model_id,
            model_b_id=model_b.model_id,
            test_name='LSTM Units Comparison',
            traffic_split=0.5,
            duration_hours=24
        )
        
        assert experiment_id in mlops_manager.active_experiments
        experiment = mlops_manager.active_experiments[experiment_id]
        assert experiment['model_a_id'] == model_a.model_id
        assert experiment['model_b_id'] == model_b.model_id
        assert experiment['status'] == 'running'
    
    async def test_evaluate_ab_test(self, mlops_manager, sample_model, sample_metrics):
        """Test A/B test evaluation."""
        # Register two models
        model_a = await mlops_manager.register_model(
            model=sample_model,
            model_type='lstm',
            symbol='BTC/USDT',
            metrics=sample_metrics
        )
        
        model_b = await mlops_manager.register_model(
            model={**sample_model, 'parameters': {'units': 256}},
            model_type='lstm',
            symbol='BTC/USDT',
            metrics=ModelMetrics(
                mse=0.08, mae=0.04, r2_score=0.88, accuracy=0.78,
                precision=0.72, recall=0.82, f1_score=0.77
            )
        )
        
        # Start A/B test
        experiment_id = await mlops_manager.start_ab_test(
            model_a_id=model_a.model_id,
            model_b_id=model_b.model_id,
            test_name='Test Evaluation'
        )
        
        # Add test data
        experiment = mlops_manager.active_experiments[experiment_id]
        experiment['results']['model_a_predictions'] = [1.0, 1.1, 0.9, 1.2, 0.8] * 200  # 1000 samples
        experiment['results']['model_b_predictions'] = [1.05, 1.15, 0.95, 1.25, 0.85] * 200
        experiment['results']['actual_values'] = [1.02, 1.12, 0.92, 1.22, 0.82] * 200
        
        # Evaluate test
        with patch('mlops_pipeline.stats') as mock_stats:
            mock_stats.ttest_rel.return_value = (2.5, 0.01)  # Significant result
            
            result = await mlops_manager.evaluate_ab_test(experiment_id)
            
            assert isinstance(result, ABTestResult)
            assert result.experiment_id == experiment_id
            assert result.winner is not None
            assert result.statistical_significance < 0.05


class TestModelScheduler:
    """Test Model Scheduler functionality."""
    
    @pytest.fixture
    async def model_scheduler(self):
        """Create model scheduler for testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Mock config and MLOps manager
        config = Mock()
        config.price_model_path = temp_dir
        
        mlops_manager = Mock()
        mlops_manager.register_model = AsyncMock()
        mlops_manager.deploy_model = AsyncMock(return_value=True)
        
        scheduler = ModelScheduler(config, mlops_manager)
        yield scheduler
        
        # Cleanup
        await scheduler.stop()
        shutil.rmtree(temp_dir)
    
    def test_add_training_job(self, model_scheduler):
        """Test adding a training job."""
        job_id = model_scheduler.add_training_job(
            model_type='lstm',
            symbol='BTC/USDT',
            schedule_type=ScheduleType.DAILY,
            schedule_config={'hour': 2, 'minute': 0}
        )
        
        assert job_id in model_scheduler.training_jobs
        job = model_scheduler.training_jobs[job_id]
        assert job.model_type == 'lstm'
        assert job.symbol == 'BTC/USDT'
        assert job.schedule_type == ScheduleType.DAILY
        assert job.enabled is True
        assert job.next_run is not None
    
    def test_remove_training_job(self, model_scheduler):
        """Test removing a training job."""
        job_id = model_scheduler.add_training_job(
            model_type='lstm',
            symbol='BTC/USDT',
            schedule_type=ScheduleType.DAILY,
            schedule_config={'hour': 2}
        )
        
        success = model_scheduler.remove_training_job(job_id)
        assert success is True
        assert job_id not in model_scheduler.training_jobs
    
    def test_enable_disable_job(self, model_scheduler):
        """Test enabling and disabling jobs."""
        job_id = model_scheduler.add_training_job(
            model_type='lstm',
            symbol='BTC/USDT',
            schedule_type=ScheduleType.DAILY,
            schedule_config={'hour': 2}
        )
        
        # Disable job
        success = model_scheduler.disable_job(job_id)
        assert success is True
        assert model_scheduler.training_jobs[job_id].enabled is False
        
        # Enable job
        success = model_scheduler.enable_job(job_id)
        assert success is True
        assert model_scheduler.training_jobs[job_id].enabled is True
    
    async def test_trigger_job(self, model_scheduler):
        """Test manually triggering a job."""
        job_id = model_scheduler.add_training_job(
            model_type='lstm',
            symbol='BTC/USDT',
            schedule_type=ScheduleType.DAILY,
            schedule_config={'hour': 2}
        )
        
        # Mock training methods
        with patch.object(model_scheduler, '_get_training_data') as mock_data, \
             patch.object(model_scheduler, '_train_model') as mock_train, \
             patch.object(model_scheduler, '_validate_model') as mock_validate:
            
            mock_data.return_value = pd.DataFrame({
                'open': [100, 101, 102],
                'high': [101, 102, 103],
                'low': [99, 100, 101],
                'close': [100.5, 101.5, 102.5],
                'volume': [1000, 1100, 1200]
            })
            
            mock_train.return_value = ({'model': 'test'}, ModelMetrics(
                mse=0.1, mae=0.05, r2_score=0.85, accuracy=0.75,
                precision=0.7, recall=0.8, f1_score=0.75
            ))
            
            mock_validate.return_value = True
            
            success = await model_scheduler.trigger_job(job_id, TriggerType.MANUAL)
            assert success is True
            assert job_id in model_scheduler.active_jobs
    
    def test_calculate_next_run_daily(self, model_scheduler):
        """Test daily schedule calculation."""
        next_run = model_scheduler._calculate_next_run(
            ScheduleType.DAILY,
            {'hour': 2, 'minute': 30}
        )
        
        assert next_run.hour == 2
        assert next_run.minute == 30
        assert next_run > datetime.utcnow()
    
    def test_calculate_next_run_weekly(self, model_scheduler):
        """Test weekly schedule calculation."""
        next_run = model_scheduler._calculate_next_run(
            ScheduleType.WEEKLY,
            {'weekday': 0, 'hour': 3, 'minute': 0}  # Monday 3 AM
        )
        
        assert next_run.weekday() == 0  # Monday
        assert next_run.hour == 3
        assert next_run.minute == 0
        assert next_run > datetime.utcnow()
    
    def test_calculate_next_run_monthly(self, model_scheduler):
        """Test monthly schedule calculation."""
        next_run = model_scheduler._calculate_next_run(
            ScheduleType.MONTHLY,
            {'day': 1, 'hour': 4, 'minute': 0}  # 1st of month, 4 AM
        )
        
        assert next_run.day == 1
        assert next_run.hour == 4
        assert next_run.minute == 0
        assert next_run > datetime.utcnow()


class TestMLOpsIntegration:
    """Integration tests for MLOps components."""
    
    @pytest.fixture
    async def mlops_system(self):
        """Create complete MLOps system for integration testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Mock config
        config = Mock()
        config.price_model_path = temp_dir
        
        # Create components
        mlops_manager = MLOpsManager(config)
        scheduler = ModelScheduler(config, mlops_manager)
        
        yield {
            'manager': mlops_manager,
            'scheduler': scheduler,
            'config': config
        }
        
        # Cleanup
        await scheduler.stop()
        shutil.rmtree(temp_dir)
    
    async def test_end_to_end_model_lifecycle(self, mlops_system):
        """Test complete model lifecycle from training to deployment."""
        manager = mlops_system['manager']
        scheduler = mlops_system['scheduler']
        
        # 1. Add training job
        job_id = scheduler.add_training_job(
            model_type='lstm',
            symbol='BTC/USDT',
            schedule_type=ScheduleType.DAILY,
            schedule_config={'hour': 2},
            deployment_config={'auto_deploy': True, 'require_approval': False}
        )
        
        # 2. Mock training components
        with patch.object(scheduler, '_get_training_data') as mock_data, \
             patch.object(scheduler, '_train_model') as mock_train, \
             patch.object(scheduler, '_validate_model') as mock_validate, \
             patch.object(manager, '_validate_model_for_deployment') as mock_deploy_validate:
            
            # Setup mocks
            mock_data.return_value = pd.DataFrame({
                'open': np.random.rand(1000),
                'high': np.random.rand(1000),
                'low': np.random.rand(1000),
                'close': np.random.rand(1000),
                'volume': np.random.rand(1000)
            })
            
            mock_train.return_value = ({'model': 'test'}, ModelMetrics(
                mse=0.05, mae=0.03, r2_score=0.9, accuracy=0.8,
                precision=0.75, recall=0.85, f1_score=0.8
            ))
            
            mock_validate.return_value = True
            mock_deploy_validate.return_value = {'valid': True, 'reason': 'All validations passed'}
            
            # 3. Trigger training job
            success = await scheduler.trigger_job(job_id, TriggerType.MANUAL)
            assert success is True
            
            # Wait for job completion
            await asyncio.sleep(0.1)
            
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
    
    async def test_drift_detection_and_retraining(self, mlops_system):
        """Test drift detection triggering retraining."""
        manager = mlops_system['manager']
        
        # Register a model
        sample_model = {'type': 'lstm', 'symbol': 'BTC/USDT'}
        sample_metrics = ModelMetrics(
            mse=0.1, mae=0.05, r2_score=0.85, accuracy=0.75,
            precision=0.7, recall=0.8, f1_score=0.75
        )
        
        model_version = await manager.register_model(
            model=sample_model,
            model_type='lstm',
            symbol='BTC/USDT',
            metrics=sample_metrics,
            metadata={
                'training_stats': {
                    'feature1': {'mean': 0.5, 'std': 0.1},
                    'feature2': {'mean': 1.0, 'std': 0.2}
                }
            }
        )
        
        # Create drifted data
        drifted_data = pd.DataFrame({
            'feature1': [1.5, 1.6, 1.7, 1.8, 1.9] * 20,  # Significant drift
            'feature2': [2.0, 2.1, 2.2, 2.3, 2.4] * 20
        })
        
        # Mock model loading and prediction
        with patch.object(manager, '_load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = np.random.rand(100)
            mock_load.return_value = mock_model
            
            # Monitor drift
            drift_result = await manager.monitor_model_drift(
                model_id=model_version.model_id,
                recent_data=drifted_data
            )
            
            # Verify drift was detected
            assert drift_result.drift_detected is True
            assert drift_result.recommendation in ['RETRAIN_IMMEDIATELY', 'RETRAIN_SOON']
    
    async def test_ab_test_workflow(self, mlops_system):
        """Test complete A/B testing workflow."""
        manager = mlops_system['manager']
        
        # Register two models with different performance
        model_a = await manager.register_model(
            model={'type': 'lstm', 'units': 128},
            model_type='lstm',
            symbol='BTC/USDT',
            metrics=ModelMetrics(
                mse=0.1, mae=0.05, r2_score=0.85, accuracy=0.75,
                precision=0.7, recall=0.8, f1_score=0.75
            )
        )
        
        model_b = await manager.register_model(
            model={'type': 'lstm', 'units': 256},
            model_type='lstm',
            symbol='BTC/USDT',
            metrics=ModelMetrics(
                mse=0.08, mae=0.04, r2_score=0.88, accuracy=0.78,
                precision=0.72, recall=0.82, f1_score=0.77
            )
        )
        
        # Start A/B test
        experiment_id = await manager.start_ab_test(
            model_a_id=model_a.model_id,
            model_b_id=model_b.model_id,
            test_name='LSTM Units Comparison'
        )
        
        # Simulate test data collection
        experiment = manager.active_experiments[experiment_id]
        
        # Model B performs better
        experiment['results']['model_a_predictions'] = [1.0 + np.random.normal(0, 0.1) for _ in range(1000)]
        experiment['results']['model_b_predictions'] = [1.0 + np.random.normal(0, 0.05) for _ in range(1000)]  # Lower variance
        experiment['results']['actual_values'] = [1.0 + np.random.normal(0, 0.02) for _ in range(1000)]
        
        # Evaluate test
        with patch('mlops_pipeline.stats') as mock_stats:
            mock_stats.ttest_rel.return_value = (3.0, 0.001)  # Highly significant
            
            result = await manager.evaluate_ab_test(experiment_id)
            
            assert result.winner == model_b.model_id
            assert result.statistical_significance < 0.05


# Performance and stress tests
class TestMLOpsPerformance:
    """Performance and stress tests for MLOps components."""
    
    @pytest.mark.asyncio
    async def test_concurrent_model_registration(self):
        """Test concurrent model registration performance."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            config = Mock()
            config.price_model_path = temp_dir
            
            manager = MLOpsManager(config)
            
            # Register multiple models concurrently
            tasks = []
            for i in range(10):
                task = manager.register_model(
                    model={'type': 'test', 'id': i},
                    model_type='lstm',
                    symbol=f'TEST{i}/USDT',
                    metrics=ModelMetrics(
                        mse=0.1, mae=0.05, r2_score=0.85, accuracy=0.75,
                        precision=0.7, recall=0.8, f1_score=0.75
                    )
                )
                tasks.append(task)
            
            # Wait for all registrations to complete
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 10
            assert len(manager.model_registry) == 10
            
        finally:
            shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_large_drift_detection_dataset(self):
        """Test drift detection with large dataset."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            config = Mock()
            config.price_model_path = temp_dir
            
            manager = MLOpsManager(config)
            
            # Register model
            model_version = await manager.register_model(
                model={'type': 'test'},
                model_type='lstm',
                symbol='BTC/USDT',
                metrics=ModelMetrics(
                    mse=0.1, mae=0.05, r2_score=0.85, accuracy=0.75,
                    precision=0.7, recall=0.8, f1_score=0.75
                ),
                metadata={
                    'training_stats': {
                        f'feature_{i}': {'mean': 0.5, 'std': 0.1}
                        for i in range(100)
                    }
                }
            )
            
            # Create large dataset
            large_data = pd.DataFrame({
                f'feature_{i}': np.random.normal(0.6, 0.12, 10000)  # Slight drift
                for i in range(100)
            })
            
            # Mock model
            with patch.object(manager, '_load_model') as mock_load:
                mock_model = Mock()
                mock_model.predict.return_value = np.random.rand(10000)
                mock_load.return_value = mock_model
                
                start_time = datetime.utcnow()
                
                drift_result = await manager.monitor_model_drift(
                    model_id=model_version.model_id,
                    recent_data=large_data
                )
                
                end_time = datetime.utcnow()
                processing_time = (end_time - start_time).total_seconds()
                
                # Should complete within reasonable time (< 10 seconds)
                assert processing_time < 10
                assert isinstance(drift_result, DriftDetectionResult)
                
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])