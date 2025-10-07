"""
Automated Model Training and Deployment Scheduler
Handles scheduled model training, validation, and deployment workflows.
"""
import asyncio
import schedule
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
import structlog
from dataclasses import dataclass
from enum import Enum
import json

import pandas as pd
from mlops_pipeline import MLOpsManager, ModelMetrics, ModelStatus

logger = structlog.get_logger(__name__)


class ScheduleType(Enum):
    """Schedule type enumeration."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class TriggerType(Enum):
    """Training trigger type enumeration."""
    SCHEDULED = "scheduled"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    MANUAL = "manual"


@dataclass
class TrainingJob:
    """Training job configuration."""
    job_id: str
    model_type: str
    symbol: str
    schedule_type: ScheduleType
    schedule_config: Dict[str, Any]
    training_config: Dict[str, Any]
    validation_config: Dict[str, Any]
    deployment_config: Dict[str, Any]
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'job_id': self.job_id,
            'model_type': self.model_type,
            'symbol': self.symbol,
            'schedule_type': self.schedule_type.value,
            'schedule_config': self.schedule_config,
            'training_config': self.training_config,
            'validation_config': self.validation_config,
            'deployment_config': self.deployment_config,
            'enabled': self.enabled,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None
        }


@dataclass
class TrainingResult:
    """Training job result."""
    job_id: str
    model_id: str
    trigger_type: TriggerType
    start_time: datetime
    end_time: datetime
    success: bool
    metrics: Optional[ModelMetrics] = None
    error_message: Optional[str] = None
    deployed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'job_id': self.job_id,
            'model_id': self.model_id,
            'trigger_type': self.trigger_type.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'success': self.success,
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'error_message': self.error_message,
            'deployed': self.deployed
        }


class ModelScheduler:
    """Automated model training and deployment scheduler."""
    
    def __init__(self, config, mlops_manager: MLOpsManager):
        self.config = config
        self.mlops_manager = mlops_manager
        
        # Scheduler configuration
        self.scheduler_config = {
            'max_concurrent_jobs': 3,
            'job_timeout_hours': 6,
            'retry_attempts': 2,
            'retry_delay_minutes': 30
        }
        
        # Job management
        self.training_jobs: Dict[str, TrainingJob] = {}
        self.active_jobs: Dict[str, asyncio.Task] = {}
        self.job_history: List[TrainingResult] = []
        
        # Paths
        self.jobs_path = Path(config.price_model_path) / "scheduler"
        self.jobs_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing jobs
        self._load_jobs()
        
        # Scheduler task
        self.scheduler_task: Optional[asyncio.Task] = None
        self.running = False
    
    async def start(self):
        """Start the scheduler."""
        if self.running:
            logger.warning("Scheduler already running")
            return
        
        self.running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Model scheduler started")
    
    async def stop(self):
        """Stop the scheduler."""
        if not self.running:
            return
        
        self.running = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Cancel active jobs
        for job_id, task in self.active_jobs.items():
            logger.info(f"Cancelling active job: {job_id}")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.active_jobs.clear()
        logger.info("Model scheduler stopped")
    
    def add_training_job(
        self,
        model_type: str,
        symbol: str,
        schedule_type: ScheduleType,
        schedule_config: Dict[str, Any],
        training_config: Dict[str, Any] = None,
        validation_config: Dict[str, Any] = None,
        deployment_config: Dict[str, Any] = None
    ) -> str:
        """Add a new training job."""
        
        job_id = f"{model_type}_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Default configurations
        if training_config is None:
            training_config = {
                'data_window_days': 30,
                'validation_split': 0.2,
                'early_stopping_patience': 10
            }
        
        if validation_config is None:
            validation_config = {
                'min_accuracy': 0.6,
                'max_mse': 1000,
                'min_r2_score': 0.5
            }
        
        if deployment_config is None:
            deployment_config = {
                'auto_deploy': True,
                'deployment_strategy': 'blue_green',
                'require_approval': False
            }
        
        # Calculate next run time
        next_run = self._calculate_next_run(schedule_type, schedule_config)
        
        job = TrainingJob(
            job_id=job_id,
            model_type=model_type,
            symbol=symbol,
            schedule_type=schedule_type,
            schedule_config=schedule_config,
            training_config=training_config,
            validation_config=validation_config,
            deployment_config=deployment_config,
            next_run=next_run
        )
        
        self.training_jobs[job_id] = job
        self._save_jobs()
        
        logger.info(f"Training job added: {job_id}", next_run=next_run)
        return job_id
    
    def remove_training_job(self, job_id: str) -> bool:
        """Remove a training job."""
        if job_id not in self.training_jobs:
            return False
        
        # Cancel if currently running
        if job_id in self.active_jobs:
            self.active_jobs[job_id].cancel()
            del self.active_jobs[job_id]
        
        del self.training_jobs[job_id]
        self._save_jobs()
        
        logger.info(f"Training job removed: {job_id}")
        return True
    
    def enable_job(self, job_id: str) -> bool:
        """Enable a training job."""
        if job_id not in self.training_jobs:
            return False
        
        self.training_jobs[job_id].enabled = True
        self._save_jobs()
        return True
    
    def disable_job(self, job_id: str) -> bool:
        """Disable a training job."""
        if job_id not in self.training_jobs:
            return False
        
        self.training_jobs[job_id].enabled = False
        self._save_jobs()
        return True
    
    async def trigger_job(self, job_id: str, trigger_type: TriggerType = TriggerType.MANUAL) -> bool:
        """Manually trigger a training job."""
        if job_id not in self.training_jobs:
            logger.error(f"Job not found: {job_id}")
            return False
        
        if job_id in self.active_jobs:
            logger.warning(f"Job already running: {job_id}")
            return False
        
        if len(self.active_jobs) >= self.scheduler_config['max_concurrent_jobs']:
            logger.warning(f"Maximum concurrent jobs reached, cannot start: {job_id}")
            return False
        
        job = self.training_jobs[job_id]
        logger.info(f"Triggering job: {job_id}", trigger_type=trigger_type.value)
        
        # Start job
        task = asyncio.create_task(self._execute_training_job(job, trigger_type))
        self.active_jobs[job_id] = task
        
        return True
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status."""
        if job_id not in self.training_jobs:
            return None
        
        job = self.training_jobs[job_id]
        status = {
            **job.to_dict(),
            'running': job_id in self.active_jobs,
            'last_results': [
                result.to_dict() for result in self.job_history
                if result.job_id == job_id
            ][-5:]  # Last 5 results
        }
        
        return status
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all training jobs."""
        return [
            {
                **job.to_dict(),
                'running': job_id in self.active_jobs
            }
            for job_id, job in self.training_jobs.items()
        ]
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        try:
            while self.running:
                current_time = datetime.utcnow()
                
                # Check for jobs that need to run
                for job_id, job in self.training_jobs.items():
                    if (job.enabled and 
                        job.next_run and 
                        current_time >= job.next_run and
                        job_id not in self.active_jobs):
                        
                        # Check concurrent job limit
                        if len(self.active_jobs) >= self.scheduler_config['max_concurrent_jobs']:
                            logger.warning("Maximum concurrent jobs reached, skipping scheduled job", job_id=job_id)
                            continue
                        
                        logger.info(f"Starting scheduled job: {job_id}")
                        
                        # Start job
                        task = asyncio.create_task(self._execute_training_job(job, TriggerType.SCHEDULED))
                        self.active_jobs[job_id] = task
                
                # Clean up completed jobs
                completed_jobs = []
                for job_id, task in self.active_jobs.items():
                    if task.done():
                        completed_jobs.append(job_id)
                
                for job_id in completed_jobs:
                    del self.active_jobs[job_id]
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
        except asyncio.CancelledError:
            logger.info("Scheduler loop cancelled")
        except Exception as e:
            logger.error(f"Scheduler loop error: {e}")
    
    async def _execute_training_job(self, job: TrainingJob, trigger_type: TriggerType) -> TrainingResult:
        """Execute a training job."""
        start_time = datetime.utcnow()
        model_id = ""
        success = False
        metrics = None
        error_message = None
        deployed = False
        
        try:
            logger.info(f"Executing training job: {job.job_id}")
            
            # Update job timing
            job.last_run = start_time
            job.next_run = self._calculate_next_run(job.schedule_type, job.schedule_config)
            
            # Get training data
            training_data = await self._get_training_data(job.symbol, job.training_config)
            
            if len(training_data) < 100:  # Minimum data requirement
                raise ValueError("Insufficient training data")
            
            # Train model
            model, metrics = await self._train_model(job, training_data)
            
            # Validate model
            validation_passed = await self._validate_model(model, metrics, job.validation_config)
            
            if not validation_passed:
                raise ValueError("Model validation failed")
            
            # Register model
            model_version = await self.mlops_manager.register_model(
                model=model,
                model_type=job.model_type,
                symbol=job.symbol,
                metrics=metrics,
                metadata={
                    'training_job_id': job.job_id,
                    'trigger_type': trigger_type.value,
                    'training_config': job.training_config
                }
            )
            
            model_id = model_version.model_id
            success = True
            
            # Deploy if configured
            if job.deployment_config.get('auto_deploy', False):
                if not job.deployment_config.get('require_approval', False):
                    deployment_success = await self.mlops_manager.deploy_model(
                        model_id=model_id,
                        deployment_strategy=job.deployment_config.get('deployment_strategy', 'blue_green')
                    )
                    deployed = deployment_success
                    
                    if deployment_success:
                        logger.info(f"Model deployed automatically: {model_id}")
                    else:
                        logger.warning(f"Automatic deployment failed: {model_id}")
            
            logger.info(f"Training job completed successfully: {job.job_id}", model_id=model_id)
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Training job failed: {job.job_id}", error=error_message)
        
        finally:
            end_time = datetime.utcnow()
            
            # Create result
            result = TrainingResult(
                job_id=job.job_id,
                model_id=model_id,
                trigger_type=trigger_type,
                start_time=start_time,
                end_time=end_time,
                success=success,
                metrics=metrics,
                error_message=error_message,
                deployed=deployed
            )
            
            # Store result
            self.job_history.append(result)
            
            # Keep only last 100 results
            if len(self.job_history) > 100:
                self.job_history = self.job_history[-100:]
            
            # Save jobs (to update timing)
            self._save_jobs()
            
            return result
    
    async def _get_training_data(self, symbol: str, training_config: Dict[str, Any]) -> pd.DataFrame:
        """Get training data for a symbol."""
        # Placeholder implementation - would fetch real market data
        logger.info(f"Fetching training data for {symbol}")
        
        # Generate synthetic data for testing
        import numpy as np
        
        days = training_config.get('data_window_days', 30)
        periods = days * 24  # Hourly data
        
        dates = pd.date_range(end=datetime.utcnow(), periods=periods, freq='1H')
        
        # Generate realistic price data
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
        returns = np.random.normal(0.0001, 0.02, periods)
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 0.01))
        
        # Create OHLCV data with technical indicators
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            volatility = close * 0.01
            high = close + np.random.uniform(0, volatility)
            low = close - np.random.uniform(0, volatility)
            open_price = low + np.random.uniform(0, high - low)
            
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            volume = np.random.uniform(1000, 10000)
            
            # Add technical indicators (simplified)
            rsi = 50 + np.random.normal(0, 15)  # RSI around 50
            rsi = max(0, min(100, rsi))
            
            macd = np.random.normal(0, 0.1)
            bb_upper = close * 1.02
            bb_lower = close * 0.98
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'rsi': rsi,
                'macd': macd,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # Add data quality metrics
        df_quality = self._assess_data_quality(df)
        logger.info(f"Data quality score: {df_quality['quality_score']:.2f}")
        
        return df
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality of training data."""
        quality_metrics = {
            'completeness': 1.0 - (df.isnull().sum().sum() / (len(df) * len(df.columns))),
            'consistency': 1.0,  # Placeholder - would check for data consistency
            'timeliness': 1.0,   # Placeholder - would check data freshness
            'accuracy': 1.0      # Placeholder - would validate against known sources
        }
        
        # Calculate overall quality score
        quality_score = sum(quality_metrics.values()) / len(quality_metrics)
        
        return {
            'quality_score': quality_score,
            'metrics': quality_metrics,
            'recommendations': self._get_data_quality_recommendations(quality_metrics)
        }
    
    def _get_data_quality_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Get recommendations for improving data quality."""
        recommendations = []
        
        if metrics['completeness'] < 0.95:
            recommendations.append("Improve data completeness - missing values detected")
        
        if metrics['consistency'] < 0.9:
            recommendations.append("Check data consistency - inconsistencies detected")
        
        if metrics['timeliness'] < 0.9:
            recommendations.append("Update data sources - stale data detected")
        
        if metrics['accuracy'] < 0.9:
            recommendations.append("Validate data accuracy - potential errors detected")
        
        return recommendations
    
    async def _train_model(self, job: TrainingJob, training_data: pd.DataFrame) -> tuple:
        """Train a model."""
        logger.info(f"Training {job.model_type} model for {job.symbol}")
        
        # Placeholder implementation - would use actual ML training
        # For now, create a mock model and metrics
        
        model = {
            'type': job.model_type,
            'symbol': job.symbol,
            'trained_at': datetime.utcnow().isoformat(),
            'data_size': len(training_data)
        }
        
        # Simulate training metrics
        metrics = ModelMetrics(
            mse=np.random.uniform(0.05, 0.2),
            mae=np.random.uniform(0.03, 0.15),
            r2_score=np.random.uniform(0.7, 0.95),
            accuracy=np.random.uniform(0.65, 0.85),
            precision=np.random.uniform(0.6, 0.8),
            recall=np.random.uniform(0.6, 0.8),
            f1_score=np.random.uniform(0.6, 0.8)
        )
        
        # Simulate training time
        await asyncio.sleep(2)
        
        return model, metrics
    
    async def _validate_model(self, model, metrics: ModelMetrics, validation_config: Dict[str, Any]) -> bool:
        """Validate trained model."""
        logger.info("Validating trained model")
        
        # Check validation criteria
        if metrics.accuracy < validation_config.get('min_accuracy', 0.6):
            logger.warning(f"Model accuracy too low: {metrics.accuracy}")
            return False
        
        if metrics.mse > validation_config.get('max_mse', 1000):
            logger.warning(f"Model MSE too high: {metrics.mse}")
            return False
        
        if metrics.r2_score < validation_config.get('min_r2_score', 0.5):
            logger.warning(f"Model R2 score too low: {metrics.r2_score}")
            return False
        
        logger.info("Model validation passed")
        return True
    
    def _calculate_next_run(self, schedule_type: ScheduleType, schedule_config: Dict[str, Any]) -> datetime:
        """Calculate next run time for a job."""
        now = datetime.utcnow()
        
        if schedule_type == ScheduleType.DAILY:
            hour = schedule_config.get('hour', 2)  # Default 2 AM
            minute = schedule_config.get('minute', 0)
            
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            
        elif schedule_type == ScheduleType.WEEKLY:
            weekday = schedule_config.get('weekday', 0)  # Monday
            hour = schedule_config.get('hour', 2)
            minute = schedule_config.get('minute', 0)
            
            days_ahead = weekday - now.weekday()
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
            
            next_run = now + timedelta(days=days_ahead)
            next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
        elif schedule_type == ScheduleType.MONTHLY:
            day = schedule_config.get('day', 1)  # First day of month
            hour = schedule_config.get('hour', 2)
            minute = schedule_config.get('minute', 0)
            
            if now.day <= day:
                next_run = now.replace(day=day, hour=hour, minute=minute, second=0, microsecond=0)
            else:
                # Next month
                if now.month == 12:
                    next_run = now.replace(year=now.year + 1, month=1, day=day, hour=hour, minute=minute, second=0, microsecond=0)
                else:
                    next_run = now.replace(month=now.month + 1, day=day, hour=hour, minute=minute, second=0, microsecond=0)
            
        elif schedule_type == ScheduleType.CUSTOM:
            interval_hours = schedule_config.get('interval_hours', 24)
            next_run = now + timedelta(hours=interval_hours)
            
        else:
            # Default to daily
            next_run = now + timedelta(days=1)
        
        return next_run
    
    def _load_jobs(self):
        """Load training jobs from disk."""
        jobs_file = self.jobs_path / "training_jobs.json"
        
        if jobs_file.exists():
            try:
                with open(jobs_file, 'r') as f:
                    jobs_data = json.load(f)
                
                for job_data in jobs_data:
                    job = TrainingJob(
                        job_id=job_data['job_id'],
                        model_type=job_data['model_type'],
                        symbol=job_data['symbol'],
                        schedule_type=ScheduleType(job_data['schedule_type']),
                        schedule_config=job_data['schedule_config'],
                        training_config=job_data['training_config'],
                        validation_config=job_data['validation_config'],
                        deployment_config=job_data['deployment_config'],
                        enabled=job_data['enabled'],
                        last_run=datetime.fromisoformat(job_data['last_run']) if job_data['last_run'] else None,
                        next_run=datetime.fromisoformat(job_data['next_run']) if job_data['next_run'] else None
                    )
                    
                    self.training_jobs[job.job_id] = job
                
                logger.info(f"Loaded {len(self.training_jobs)} training jobs")
                
            except Exception as e:
                logger.error(f"Failed to load training jobs: {e}")
    
    def _save_jobs(self):
        """Save training jobs to disk."""
        jobs_file = self.jobs_path / "training_jobs.json"
        
        try:
            jobs_data = [job.to_dict() for job in self.training_jobs.values()]
            
            with open(jobs_file, 'w') as f:
                json.dump(jobs_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save training jobs: {e}")
    
    async def setup_automated_triggers(self):
        """Set up automated triggers for model retraining."""
        try:
            logger.info("Setting up automated triggers")
            
            # Performance degradation trigger
            asyncio.create_task(self._performance_monitoring_task())
            
            # Data drift trigger
            asyncio.create_task(self._drift_monitoring_task())
            
            # Market regime change trigger
            asyncio.create_task(self._market_regime_monitoring_task())
            
            logger.info("Automated triggers set up successfully")
            
        except Exception as e:
            logger.error(f"Failed to set up automated triggers: {e}")
    
    async def _performance_monitoring_task(self):
        """Monitor model performance and trigger retraining if needed."""
        try:
            while self.running:
                logger.debug("Checking model performance for degradation")
                
                # Check all deployed models
                for versions in self.mlops_manager.model_registry.values():
                    for version in versions:
                        if version.status == ModelStatus.DEPLOYED:
                            await self._check_model_performance_degradation(version)
                
                # Wait before next check (every 4 hours)
                await asyncio.sleep(4 * 3600)
                
        except asyncio.CancelledError:
            logger.info("Performance monitoring task cancelled")
        except Exception as e:
            logger.error(f"Performance monitoring task error: {e}")
    
    async def _drift_monitoring_task(self):
        """Monitor for data drift and trigger retraining if needed."""
        try:
            while self.running:
                logger.debug("Checking for data drift")
                
                # Check all deployed models
                for versions in self.mlops_manager.model_registry.values():
                    for version in versions:
                        if version.status == ModelStatus.DEPLOYED:
                            await self._check_model_drift(version)
                
                # Wait before next check (every 2 hours)
                await asyncio.sleep(2 * 3600)
                
        except asyncio.CancelledError:
            logger.info("Drift monitoring task cancelled")
        except Exception as e:
            logger.error(f"Drift monitoring task error: {e}")
    
    async def _market_regime_monitoring_task(self):
        """Monitor for market regime changes and trigger retraining if needed."""
        try:
            while self.running:
                logger.debug("Checking for market regime changes")
                
                # This would integrate with the regime detection service
                # For now, simulate regime change detection
                regime_change_detected = await self._detect_market_regime_change()
                
                if regime_change_detected:
                    logger.info("Market regime change detected, triggering model retraining")
                    await self._trigger_regime_based_retraining()
                
                # Wait before next check (every 6 hours)
                await asyncio.sleep(6 * 3600)
                
        except asyncio.CancelledError:
            logger.info("Market regime monitoring task cancelled")
        except Exception as e:
            logger.error(f"Market regime monitoring task error: {e}")
    
    async def _check_model_performance_degradation(self, model_version: ModelVersion):
        """Check if a model's performance has degraded."""
        try:
            # Get recent performance history
            history = await self.mlops_manager.get_model_performance_history(
                model_version.model_id, days=7
            )
            
            if len(history) < 5:  # Need at least 5 data points
                return
            
            # Calculate performance trend
            recent_mse = [entry['mse'] for entry in history[-5:]]
            baseline_mse = model_version.metrics.mse
            
            avg_recent_mse = sum(recent_mse) / len(recent_mse)
            performance_degradation = (avg_recent_mse - baseline_mse) / baseline_mse
            
            # Trigger retraining if performance degraded significantly
            if performance_degradation > 0.2:  # 20% degradation threshold
                logger.warning(
                    f"Performance degradation detected for {model_version.model_id}: "
                    f"{performance_degradation:.2%}"
                )
                
                await self._trigger_performance_based_retraining(model_version)
                
        except Exception as e:
            logger.error(f"Failed to check performance degradation: {e}")
    
    async def _check_model_drift(self, model_version: ModelVersion):
        """Check if a model has data drift."""
        try:
            # Get recent data for drift detection
            recent_data = await self._get_recent_market_data(model_version.symbol)
            
            if len(recent_data) < 100:  # Need sufficient data
                return
            
            # Monitor drift
            drift_result = await self.mlops_manager.monitor_model_drift(
                model_version.model_id, recent_data
            )
            
            # Trigger retraining if significant drift detected
            if drift_result.recommendation in ["RETRAIN_IMMEDIATELY", "RETRAIN_SOON"]:
                logger.warning(
                    f"Data drift detected for {model_version.model_id}: "
                    f"score={drift_result.drift_score:.3f}, recommendation={drift_result.recommendation}"
                )
                
                await self._trigger_drift_based_retraining(model_version, drift_result)
                
        except Exception as e:
            logger.error(f"Failed to check model drift: {e}")
    
    async def _detect_market_regime_change(self) -> bool:
        """Detect if market regime has changed significantly."""
        try:
            # Placeholder implementation
            # In a real system, this would:
            # 1. Analyze recent market volatility
            # 2. Check correlation changes
            # 3. Monitor sentiment shifts
            # 4. Detect structural breaks
            
            # Simulate regime change detection (5% chance)
            import random
            return random.random() < 0.05
            
        except Exception as e:
            logger.error(f"Failed to detect market regime change: {e}")
            return False
    
    async def _get_recent_market_data(self, symbol: str, hours: int = 24) -> pd.DataFrame:
        """Get recent market data for drift detection."""
        try:
            # Placeholder - would fetch real recent data
            periods = hours
            dates = pd.date_range(end=datetime.utcnow(), periods=periods, freq='1H')
            
            # Generate recent data with potential drift
            data = []
            for date in dates:
                data.append({
                    'timestamp': date,
                    'feature1': np.random.normal(0.5, 0.1),
                    'feature2': np.random.normal(1.0, 0.2),
                    'feature3': np.random.normal(0.0, 0.5)
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get recent market data: {e}")
            return pd.DataFrame()
    
    async def _trigger_performance_based_retraining(self, model_version: ModelVersion):
        """Trigger retraining due to performance degradation."""
        try:
            # Find corresponding training job
            job_id = self._find_job_for_model(model_version)
            
            if job_id:
                logger.info(f"Triggering performance-based retraining for job {job_id}")
                await self.trigger_job(job_id, TriggerType.PERFORMANCE_DEGRADATION)
            else:
                # Create ad-hoc retraining job
                await self._create_adhoc_retraining_job(
                    model_version, TriggerType.PERFORMANCE_DEGRADATION
                )
                
        except Exception as e:
            logger.error(f"Failed to trigger performance-based retraining: {e}")
    
    async def _trigger_drift_based_retraining(self, model_version: ModelVersion, drift_result: DriftDetectionResult):
        """Trigger retraining due to data drift."""
        try:
            # Find corresponding training job
            job_id = self._find_job_for_model(model_version)
            
            if job_id:
                logger.info(f"Triggering drift-based retraining for job {job_id}")
                await self.trigger_job(job_id, TriggerType.DATA_DRIFT)
            else:
                # Create ad-hoc retraining job
                await self._create_adhoc_retraining_job(
                    model_version, TriggerType.DATA_DRIFT
                )
                
        except Exception as e:
            logger.error(f"Failed to trigger drift-based retraining: {e}")
    
    async def _trigger_regime_based_retraining(self):
        """Trigger retraining for all models due to market regime change."""
        try:
            logger.info("Triggering regime-based retraining for all active models")
            
            # Trigger retraining for all deployed models
            for versions in self.mlops_manager.model_registry.values():
                for version in versions:
                    if version.status == ModelStatus.DEPLOYED:
                        job_id = self._find_job_for_model(version)
                        if job_id and job_id not in self.active_jobs:
                            await self.trigger_job(job_id, TriggerType.MANUAL)  # Use manual for regime change
                            
                            # Add delay between jobs to avoid overload
                            await asyncio.sleep(60)
                            
        except Exception as e:
            logger.error(f"Failed to trigger regime-based retraining: {e}")
    
    def _find_job_for_model(self, model_version: ModelVersion) -> Optional[str]:
        """Find training job that corresponds to a model."""
        for job_id, job in self.training_jobs.items():
            if (job.model_type == model_version.model_type and 
                job.symbol == model_version.symbol):
                return job_id
        return None
    
    async def _create_adhoc_retraining_job(self, model_version: ModelVersion, trigger_type: TriggerType):
        """Create an ad-hoc retraining job for a model."""
        try:
            logger.info(f"Creating ad-hoc retraining job for {model_version.model_id}")
            
            # Create temporary job
            job_id = self.add_training_job(
                model_type=model_version.model_type,
                symbol=model_version.symbol,
                schedule_type=ScheduleType.CUSTOM,
                schedule_config={'interval_hours': 24 * 365},  # Very long interval (won't repeat)
                training_config={
                    'data_window_days': 30,
                    'validation_split': 0.2,
                    'trigger_reason': trigger_type.value
                },
                deployment_config={
                    'auto_deploy': True,
                    'require_approval': False,
                    'deployment_strategy': 'canary'  # Use canary for triggered retraining
                }
            )
            
            # Trigger immediately
            await self.trigger_job(job_id, trigger_type)
            
            # Remove job after completion (it will be cleaned up automatically)
            
        except Exception as e:
            logger.error(f"Failed to create ad-hoc retraining job: {e}")
    
    async def get_automation_status(self) -> Dict[str, Any]:
        """Get status of automation features."""
        try:
            return {
                'scheduler_running': self.running,
                'active_jobs': len(self.active_jobs),
                'total_jobs': len(self.training_jobs),
                'enabled_jobs': len([job for job in self.training_jobs.values() if job.enabled]),
                'recent_completions': len([
                    result for result in self.job_history[-10:]
                    if result.end_time > datetime.utcnow() - timedelta(hours=24)
                ]),
                'automation_triggers': {
                    'performance_monitoring': True,
                    'drift_monitoring': True,
                    'regime_monitoring': True
                },
                'last_trigger_check': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get automation status: {e}")
            return {'error': str(e)}


# Create scheduler instance
def create_model_scheduler(config, mlops_manager: MLOpsManager) -> ModelScheduler:
    """Create model scheduler instance."""
    return ModelScheduler(config, mlops_manager)