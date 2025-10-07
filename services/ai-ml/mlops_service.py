"""
MLOps Service API
Provides REST API endpoints for MLOps pipeline management.
"""
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import structlog

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
import pandas as pd

from mlops_pipeline import MLOpsManager, ModelMetrics, ModelStatus, ExperimentType

logger = structlog.get_logger(__name__)


# Pydantic models for API
class ModelRegistrationRequest(BaseModel):
    """Model registration request."""
    model_type: str = Field(..., description="Type of model (lstm, transformer, etc.)")
    symbol: str = Field(..., description="Trading symbol")
    model_data: str = Field(..., description="Base64 encoded model data")
    metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class ModelDeploymentRequest(BaseModel):
    """Model deployment request."""
    model_id: str = Field(..., description="Model ID to deploy")
    deployment_strategy: str = Field(default="blue_green", description="Deployment strategy")


class DriftMonitoringRequest(BaseModel):
    """Drift monitoring request."""
    model_id: str = Field(..., description="Model ID to monitor")
    data: List[Dict[str, float]] = Field(..., description="Recent data for drift detection")


class ABTestRequest(BaseModel):
    """A/B test request."""
    model_a_id: str = Field(..., description="First model ID")
    model_b_id: str = Field(..., description="Second model ID")
    test_name: str = Field(..., description="Test name")
    traffic_split: float = Field(default=0.5, description="Traffic split ratio")
    duration_hours: int = Field(default=24, description="Test duration in hours")


class RetrainRequest(BaseModel):
    """Model retraining request."""
    model_id: str = Field(..., description="Model ID to retrain")
    training_data: List[Dict[str, float]] = Field(..., description="New training data")
    retrain_strategy: str = Field(default="incremental", description="Retraining strategy")


class MLOpsService:
    """MLOps service for model lifecycle management."""
    
    def __init__(self, config):
        self.config = config
        self.mlops_manager = MLOpsManager(config)
        self.app = FastAPI(
            title="MLOps Service",
            description="Model lifecycle management service",
            version="1.0.0"
        )
        
        # Setup routes
        self._setup_routes()
        
        # Background tasks
        self.monitoring_tasks = {}
        
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "mlops_manager": "ready"
            }
        
        @self.app.post("/api/models/register")
        async def register_model(request: ModelRegistrationRequest):
            """Register a new model."""
            try:
                # Convert metrics to ModelMetrics object
                metrics = ModelMetrics(
                    mse=request.metrics.get('mse', 0.0),
                    mae=request.metrics.get('mae', 0.0),
                    r2_score=request.metrics.get('r2_score', 0.0),
                    accuracy=request.metrics.get('accuracy', 0.0),
                    precision=request.metrics.get('precision', 0.0),
                    recall=request.metrics.get('recall', 0.0),
                    f1_score=request.metrics.get('f1_score', 0.0)
                )
                
                # Decode model data (in real implementation)
                # For now, create a placeholder model
                model = {"type": request.model_type, "symbol": request.symbol}
                
                # Register model
                model_version = await self.mlops_manager.register_model(
                    model=model,
                    model_type=request.model_type,
                    symbol=request.symbol,
                    metrics=metrics,
                    metadata=request.metadata
                )
                
                return {
                    "status": "success",
                    "model_id": model_version.model_id,
                    "version": model_version.version,
                    "message": "Model registered successfully"
                }
                
            except Exception as e:
                logger.error(f"Model registration failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/models/deploy")
        async def deploy_model(request: ModelDeploymentRequest):
            """Deploy a model to production."""
            try:
                success = await self.mlops_manager.deploy_model(
                    model_id=request.model_id,
                    deployment_strategy=request.deployment_strategy
                )
                
                if success:
                    return {
                        "status": "success",
                        "model_id": request.model_id,
                        "message": "Model deployed successfully"
                    }
                else:
                    raise HTTPException(status_code=500, detail="Deployment failed")
                    
            except Exception as e:
                logger.error(f"Model deployment failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/models/{model_id}")
        async def get_model_info(model_id: str):
            """Get model information."""
            try:
                model_version = self.mlops_manager._find_model_version(model_id)
                if not model_version:
                    raise HTTPException(status_code=404, detail="Model not found")
                
                return model_version.to_dict()
                
            except Exception as e:
                logger.error(f"Failed to get model info: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/models")
        async def list_models(
            model_type: Optional[str] = None,
            symbol: Optional[str] = None,
            status: Optional[str] = None
        ):
            """List models with optional filters."""
            try:
                models = []
                
                for versions in self.mlops_manager.model_registry.values():
                    for version in versions:
                        # Apply filters
                        if model_type and version.model_type != model_type:
                            continue
                        if symbol and version.symbol != symbol:
                            continue
                        if status and version.status.value != status:
                            continue
                        
                        models.append(version.to_dict())
                
                return {
                    "models": models,
                    "total": len(models)
                }
                
            except Exception as e:
                logger.error(f"Failed to list models: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/monitoring/drift")
        async def monitor_drift(request: DriftMonitoringRequest):
            """Monitor model for drift."""
            try:
                # Convert data to DataFrame
                df = pd.DataFrame(request.data)
                
                # Monitor drift
                drift_result = await self.mlops_manager.monitor_model_drift(
                    model_id=request.model_id,
                    recent_data=df
                )
                
                return drift_result.to_dict()
                
            except Exception as e:
                logger.error(f"Drift monitoring failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/experiments/ab-test")
        async def start_ab_test(request: ABTestRequest):
            """Start an A/B test."""
            try:
                experiment_id = await self.mlops_manager.start_ab_test(
                    model_a_id=request.model_a_id,
                    model_b_id=request.model_b_id,
                    test_name=request.test_name,
                    traffic_split=request.traffic_split,
                    duration_hours=request.duration_hours
                )
                
                return {
                    "status": "success",
                    "experiment_id": experiment_id,
                    "message": "A/B test started successfully"
                }
                
            except Exception as e:
                logger.error(f"A/B test start failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/experiments/{experiment_id}")
        async def get_experiment_status(experiment_id: str):
            """Get experiment status."""
            try:
                if experiment_id not in self.mlops_manager.active_experiments:
                    raise HTTPException(status_code=404, detail="Experiment not found")
                
                experiment = self.mlops_manager.active_experiments[experiment_id]
                return experiment
                
            except Exception as e:
                logger.error(f"Failed to get experiment status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/experiments/{experiment_id}/evaluate")
        async def evaluate_ab_test(experiment_id: str):
            """Evaluate A/B test results."""
            try:
                result = await self.mlops_manager.evaluate_ab_test(experiment_id)
                return result.to_dict()
                
            except Exception as e:
                logger.error(f"A/B test evaluation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/models/retrain")
        async def retrain_model(request: RetrainRequest, background_tasks: BackgroundTasks):
            """Retrain a model with new data."""
            try:
                # Convert data to DataFrame
                df = pd.DataFrame(request.training_data)
                
                # Start retraining in background
                background_tasks.add_task(
                    self._retrain_model_background,
                    request.model_id,
                    df,
                    request.retrain_strategy
                )
                
                return {
                    "status": "accepted",
                    "model_id": request.model_id,
                    "message": "Retraining started in background"
                }
                
            except Exception as e:
                logger.error(f"Model retraining failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/monitoring/dashboard")
        async def get_monitoring_dashboard():
            """Get MLOps monitoring dashboard data."""
            try:
                # Collect dashboard metrics
                total_models = sum(len(versions) for versions in self.mlops_manager.model_registry.values())
                
                deployed_models = 0
                training_models = 0
                deprecated_models = 0
                
                for versions in self.mlops_manager.model_registry.values():
                    for version in versions:
                        if version.status == ModelStatus.DEPLOYED:
                            deployed_models += 1
                        elif version.status == ModelStatus.TRAINING:
                            training_models += 1
                        elif version.status == ModelStatus.DEPRECATED:
                            deprecated_models += 1
                
                active_experiments = len(self.mlops_manager.active_experiments)
                
                return {
                    "summary": {
                        "total_models": total_models,
                        "deployed_models": deployed_models,
                        "training_models": training_models,
                        "deprecated_models": deprecated_models,
                        "active_experiments": active_experiments
                    },
                    "recent_activities": await self._get_recent_activities(),
                    "performance_metrics": await self._get_performance_metrics(),
                    "drift_alerts": await self._get_drift_alerts()
                }
                
            except Exception as e:
                logger.error(f"Failed to get monitoring dashboard: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/monitoring/start-continuous")
        async def start_continuous_monitoring(model_id: str, interval_hours: int = 1):
            """Start continuous monitoring for a model."""
            try:
                if model_id in self.monitoring_tasks:
                    return {
                        "status": "already_running",
                        "model_id": model_id,
                        "message": "Continuous monitoring already active"
                    }
                
                # Start monitoring task
                task = asyncio.create_task(
                    self._continuous_monitoring_task(model_id, interval_hours)
                )
                self.monitoring_tasks[model_id] = task
                
                return {
                    "status": "started",
                    "model_id": model_id,
                    "message": "Continuous monitoring started"
                }
                
            except Exception as e:
                logger.error(f"Failed to start continuous monitoring: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/monitoring/stop-continuous")
        async def stop_continuous_monitoring(model_id: str):
            """Stop continuous monitoring for a model."""
            try:
                if model_id not in self.monitoring_tasks:
                    raise HTTPException(status_code=404, detail="Monitoring not active")
                
                # Cancel monitoring task
                self.monitoring_tasks[model_id].cancel()
                del self.monitoring_tasks[model_id]
                
                return {
                    "status": "stopped",
                    "model_id": model_id,
                    "message": "Continuous monitoring stopped"
                }
                
            except Exception as e:
                logger.error(f"Failed to stop continuous monitoring: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/models/{model_id}/performance-history")
        async def get_model_performance_history(model_id: str, days: int = 30):
            """Get model performance history."""
            try:
                history = await self.mlops_manager.get_model_performance_history(model_id, days)
                return {
                    "model_id": model_id,
                    "days": days,
                    "history": history
                }
                
            except Exception as e:
                logger.error(f"Failed to get performance history: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/models/{model_id}/drift-history")
        async def get_model_drift_history(model_id: str, days: int = 30):
            """Get model drift detection history."""
            try:
                drift_history = await self.mlops_manager.get_drift_history(model_id, days)
                return {
                    "model_id": model_id,
                    "days": days,
                    "drift_history": [drift.to_dict() for drift in drift_history]
                }
                
            except Exception as e:
                logger.error(f"Failed to get drift history: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/experiments/history")
        async def get_ab_test_history(days: int = 30):
            """Get A/B test history."""
            try:
                test_history = await self.mlops_manager.get_ab_test_history(days)
                return {
                    "days": days,
                    "test_history": [test.to_dict() for test in test_history]
                }
                
            except Exception as e:
                logger.error(f"Failed to get A/B test history: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/system/cleanup")
        async def cleanup_old_models(days: int = 30):
            """Clean up old deprecated models."""
            try:
                await self.mlops_manager.cleanup_old_models(days)
                return {
                    "status": "success",
                    "message": f"Cleaned up models older than {days} days"
                }
                
            except Exception as e:
                logger.error(f"Model cleanup failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/system/health")
        async def get_system_health():
            """Get MLOps system health status."""
            try:
                health = await self.mlops_manager.get_system_health()
                return health
                
            except Exception as e:
                logger.error(f"Failed to get system health: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/models/{model_id}/promote")
        async def promote_model(model_id: str):
            """Promote a model from validation to production."""
            try:
                model_version = self.mlops_manager._find_model_version(model_id)
                if not model_version:
                    raise HTTPException(status_code=404, detail="Model not found")
                
                if model_version.status != ModelStatus.VALIDATION:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Model must be in validation status, currently: {model_version.status.value}"
                    )
                
                # Deploy the model
                success = await self.mlops_manager.deploy_model(model_id, "blue_green")
                
                if success:
                    return {
                        "status": "success",
                        "model_id": model_id,
                        "message": "Model promoted to production"
                    }
                else:
                    raise HTTPException(status_code=500, detail="Model promotion failed")
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Model promotion failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/models/{model_id}/rollback")
        async def rollback_model(model_id: str):
            """Rollback a deployed model to previous version."""
            try:
                model_version = self.mlops_manager._find_model_version(model_id)
                if not model_version:
                    raise HTTPException(status_code=404, detail="Model not found")
                
                # Find previous deployed version
                registry_key = f"{model_version.model_type}_{model_version.symbol}"
                if registry_key not in self.mlops_manager.model_registry:
                    raise HTTPException(status_code=404, detail="No previous version found")
                
                versions = self.mlops_manager.model_registry[registry_key]
                previous_versions = [
                    v for v in versions 
                    if v.model_id != model_id and v.status == ModelStatus.DEPRECATED
                ]
                
                if not previous_versions:
                    raise HTTPException(status_code=404, detail="No previous version to rollback to")
                
                # Get most recent previous version
                previous_version = max(previous_versions, key=lambda x: x.created_at)
                
                # Rollback: deprecate current, activate previous
                model_version.status = ModelStatus.DEPRECATED
                model_version.deprecated_at = datetime.utcnow()
                
                previous_version.status = ModelStatus.DEPLOYED
                previous_version.deployed_at = datetime.utcnow()
                
                await self.mlops_manager._save_registry()
                
                return {
                    "status": "success",
                    "rolled_back_from": model_id,
                    "rolled_back_to": previous_version.model_id,
                    "message": "Model rollback completed"
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Model rollback failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/models/compare")
        async def compare_models(model_a_id: str, model_b_id: str):
            """Compare two models side by side."""
            try:
                model_a = self.mlops_manager._find_model_version(model_a_id)
                model_b = self.mlops_manager._find_model_version(model_b_id)
                
                if not model_a:
                    raise HTTPException(status_code=404, detail=f"Model A not found: {model_a_id}")
                if not model_b:
                    raise HTTPException(status_code=404, detail=f"Model B not found: {model_b_id}")
                
                # Get performance histories
                history_a = await self.mlops_manager.get_model_performance_history(model_a_id, 30)
                history_b = await self.mlops_manager.get_model_performance_history(model_b_id, 30)
                
                comparison = {
                    "model_a": {
                        **model_a.to_dict(),
                        "performance_history": history_a
                    },
                    "model_b": {
                        **model_b.to_dict(),
                        "performance_history": history_b
                    },
                    "metrics_comparison": {
                        "mse_difference": model_b.metrics.mse - model_a.metrics.mse,
                        "accuracy_difference": model_b.metrics.accuracy - model_a.metrics.accuracy,
                        "r2_difference": model_b.metrics.r2_score - model_a.metrics.r2_score,
                        "better_model": model_a_id if model_a.metrics.mse < model_b.metrics.mse else model_b_id
                    }
                }
                
                return comparison
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Model comparison failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _retrain_model_background(self, model_id: str, training_data: pd.DataFrame, strategy: str):
        """Background task for model retraining."""
        try:
            logger.info(f"Starting background retraining for model {model_id}")
            
            new_model_version = await self.mlops_manager.retrain_model(
                model_id=model_id,
                new_data=training_data,
                retrain_strategy=strategy
            )
            
            logger.info(f"Model retraining completed: {new_model_version.model_id}")
            
        except Exception as e:
            logger.error(f"Background retraining failed for {model_id}: {e}")
    
    async def _continuous_monitoring_task(self, model_id: str, interval_hours: int):
        """Continuous monitoring task."""
        try:
            while True:
                logger.info(f"Running continuous monitoring for {model_id}")
                
                # Get recent data (placeholder - would fetch real data)
                recent_data = pd.DataFrame({
                    'feature1': [1, 2, 3, 4, 5],
                    'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
                })
                
                # Monitor drift
                try:
                    drift_result = await self.mlops_manager.monitor_model_drift(
                        model_id=model_id,
                        recent_data=recent_data
                    )
                    
                    # Check if action is needed
                    if drift_result.recommendation in ["RETRAIN_IMMEDIATELY", "RETRAIN_SOON"]:
                        logger.warning(f"Drift detected for {model_id}: {drift_result.recommendation}")
                        
                        # Could trigger automatic retraining here
                        if drift_result.recommendation == "RETRAIN_IMMEDIATELY":
                            logger.info(f"Triggering automatic retraining for {model_id}")
                            # await self._trigger_automatic_retraining(model_id)
                    
                except Exception as e:
                    logger.error(f"Monitoring failed for {model_id}: {e}")
                
                # Wait for next monitoring cycle
                await asyncio.sleep(interval_hours * 3600)
                
        except asyncio.CancelledError:
            logger.info(f"Continuous monitoring stopped for {model_id}")
        except Exception as e:
            logger.error(f"Continuous monitoring error for {model_id}: {e}")
    
    async def _get_recent_activities(self) -> List[Dict[str, Any]]:
        """Get recent MLOps activities."""
        # Placeholder implementation
        return [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "activity": "model_deployed",
                "model_id": "lstm_BTC_20241205_120000",
                "details": "Blue-green deployment completed successfully"
            },
            {
                "timestamp": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                "activity": "drift_detected",
                "model_id": "transformer_ETH_20241204_100000",
                "details": "Performance degradation detected, retraining recommended"
            }
        ]
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get aggregated performance metrics."""
        # Placeholder implementation
        return {
            "average_mse": 0.15,
            "average_accuracy": 0.78,
            "deployment_success_rate": 0.95,
            "average_deployment_time": 120  # seconds
        }
    
    async def _get_drift_alerts(self) -> List[Dict[str, Any]]:
        """Get active drift alerts."""
        # Placeholder implementation
        return [
            {
                "model_id": "transformer_ETH_20241204_100000",
                "drift_score": 0.12,
                "threshold": 0.1,
                "recommendation": "RETRAIN_SOON",
                "timestamp": datetime.utcnow().isoformat()
            }
        ]


# Create service instance
def create_mlops_service(config) -> MLOpsService:
    """Create MLOps service instance."""
    return MLOpsService(config)