"""
Correlation and Concentration Monitoring Engine for Advanced Trading Platform.
Implements real-time correlation matrix calculation, portfolio heat maps, and concentration risk alerts.
"""
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import math
import warnings
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
# Plotly imports (optional for visualization)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
warnings.filterwarnings('ignore')

import structlog
from pydantic import BaseModel, Field

# Configure logging
logger = structlog.get_logger("correlation-monitor")

class CorrelationMethod(str, Enum):
    """Correlation calculation methods."""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    ROLLING = "rolling"
    EXPONENTIAL = "exponential"
    DCC_GARCH = "dcc_garch"  # Dynamic Conditional Correlation

class ConcentrationMetric(str, Enum):
    """Concentration risk metrics."""
    HERFINDAHL = "herfindahl"
    ENTROPY = "entropy"
    MAX_WEIGHT = "max_weight"
    TOP_N_CONCENTRATION = "top_n_concentration"
    EFFECTIVE_NUMBER = "effective_number"

class AlertSeverity(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class CorrelationAlert:
    """Correlation-based alert."""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    asset_pair: Tuple[str, str]
    correlation_value: float
    threshold: float
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConcentrationAlert:
    """Concentration risk alert."""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    asset: Optional[str]
    concentration_value: float
    threshold: float
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CorrelationMatrix:
    """Correlation matrix with metadata."""
    matrix: np.ndarray
    symbols: List[str]
    method: CorrelationMethod
    calculation_timestamp: datetime
    lookback_days: int
    confidence_intervals: Optional[np.ndarray] = None
    p_values: Optional[np.ndarray] = None
    eigenvalues: Optional[np.ndarray] = None
    condition_number: Optional[float] = None

@dataclass
class ConcentrationMetrics:
    """Portfolio concentration metrics."""
    herfindahl_index: float
    entropy_measure: float
    max_weight: float
    max_weight_asset: str
    top_3_concentration: float
    top_5_concentration: float
    effective_number_assets: float
    diversification_ratio: float
    concentration_alerts: List[ConcentrationAlert]
    calculation_timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class PortfolioHeatMap:
    """Portfolio heat map data."""
    correlation_heatmap: Dict[str, Any]
    concentration_heatmap: Dict[str, Any]
    risk_contribution_heatmap: Dict[str, Any]
    sector_heatmap: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

class CorrelationMonitorConfig(BaseModel):
    """Configuration for correlation and concentration monitoring."""
    correlation_threshold_high: float = Field(default=0.7, description="High correlation threshold")
    correlation_threshold_critical: float = Field(default=0.9, description="Critical correlation threshold")
    concentration_threshold_medium: float = Field(default=0.25, description="Medium concentration threshold")
    concentration_threshold_high: float = Field(default=0.4, description="High concentration threshold")
    lookback_days: int = Field(default=252, description="Lookback period for correlation calculation")
    rolling_window: int = Field(default=30, description="Rolling window for dynamic correlations")
    ewma_lambda: float = Field(default=0.94, description="EWMA decay factor")
    update_frequency_minutes: int = Field(default=15, description="Update frequency in minutes")
    min_observations: int = Field(default=30, description="Minimum observations for correlation")
    confidence_level: float = Field(default=0.95, description="Confidence level for correlation tests")

class CorrelationConcentrationMonitor:
    """Advanced correlation and concentration monitoring system."""
    
    def __init__(self, config: CorrelationMonitorConfig = None):
        self.config = config or CorrelationMonitorConfig()
        self.correlation_cache = {}
        self.concentration_cache = {}
        self.alert_history = []
        self.active_alerts = {}
        
        # Monitoring state
        self.last_update = None
        self.correlation_matrices = {}
        self.concentration_metrics = {}
        
        logger.info("Correlation and Concentration Monitor initialized", config=self.config.dict())
    
    async def calculate_correlation_matrix(self,
                                         returns_data: Dict[str, List[float]],
                                         method: CorrelationMethod = CorrelationMethod.PEARSON,
                                         lookback_days: Optional[int] = None) -> CorrelationMatrix:
        """Calculate correlation matrix using specified method."""
        try:
            lookback_days = lookback_days or self.config.lookback_days
            symbols = list(returns_data.keys())
            
            # Prepare returns matrix
            returns_matrix = self._prepare_returns_matrix(returns_data, lookback_days)
            
            if returns_matrix.shape[0] < self.config.min_observations:
                raise ValueError(f"Insufficient data: {returns_matrix.shape[0]} < {self.config.min_observations}")
            
            # Calculate correlation matrix based on method
            if method == CorrelationMethod.PEARSON:
                corr_matrix, p_values = self._calculate_pearson_correlation(returns_matrix)
            elif method == CorrelationMethod.SPEARMAN:
                corr_matrix, p_values = self._calculate_spearman_correlation(returns_matrix)
            elif method == CorrelationMethod.ROLLING:
                corr_matrix = self._calculate_rolling_correlation(returns_matrix)
                p_values = None
            elif method == CorrelationMethod.EXPONENTIAL:
                corr_matrix = self._calculate_exponential_correlation(returns_matrix)
                p_values = None
            else:
                # Default to Pearson
                corr_matrix, p_values = self._calculate_pearson_correlation(returns_matrix)
            
            # Calculate additional statistics
            eigenvalues = np.linalg.eigvals(corr_matrix)
            condition_number = np.max(eigenvalues) / np.min(eigenvalues) if np.min(eigenvalues) > 1e-10 else np.inf
            
            # Calculate confidence intervals if p-values available
            confidence_intervals = None
            if p_values is not None:
                confidence_intervals = self._calculate_correlation_confidence_intervals(
                    corr_matrix, returns_matrix.shape[0]
                )
            
            correlation_matrix = CorrelationMatrix(
                matrix=corr_matrix,
                symbols=symbols,
                method=method,
                calculation_timestamp=datetime.utcnow(),
                lookback_days=lookback_days,
                confidence_intervals=confidence_intervals,
                p_values=p_values,
                eigenvalues=eigenvalues,
                condition_number=condition_number
            )
            
            # Cache the result
            cache_key = f"{'-'.join(symbols)}_{method.value}_{lookback_days}"
            self.correlation_cache[cache_key] = correlation_matrix
            
            logger.info("Correlation matrix calculated",
                       method=method.value,
                       n_assets=len(symbols),
                       avg_correlation=np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]),
                       condition_number=condition_number)
            
            return correlation_matrix
            
        except Exception as e:
            logger.error("Correlation matrix calculation failed", error=str(e), method=method.value)
            raise
    
    async def calculate_concentration_metrics(self,
                                            portfolio_weights: Dict[str, float],
                                            sector_mapping: Optional[Dict[str, str]] = None) -> ConcentrationMetrics:
        """Calculate comprehensive concentration risk metrics."""
        try:
            if not portfolio_weights:
                raise ValueError("Portfolio weights cannot be empty")
            
            # Normalize weights to ensure they sum to 1
            total_weight = sum(abs(w) for w in portfolio_weights.values())
            if total_weight == 0:
                raise ValueError("Total portfolio weight cannot be zero")
            
            normalized_weights = {asset: abs(weight) / total_weight 
                                for asset, weight in portfolio_weights.items()}
            
            weights_array = np.array(list(normalized_weights.values()))
            assets = list(normalized_weights.keys())
            
            # Calculate concentration metrics
            herfindahl_index = np.sum(weights_array ** 2)
            entropy_measure = -np.sum(weights_array * np.log(weights_array + 1e-10))
            max_weight = np.max(weights_array)
            max_weight_asset = assets[np.argmax(weights_array)]
            
            # Top N concentrations
            sorted_weights = np.sort(weights_array)[::-1]  # Descending order
            top_3_concentration = np.sum(sorted_weights[:min(3, len(sorted_weights))])
            top_5_concentration = np.sum(sorted_weights[:min(5, len(sorted_weights))])
            
            # Effective number of assets
            effective_number_assets = 1.0 / herfindahl_index if herfindahl_index > 0 else len(assets)
            
            # Diversification ratio (simplified)
            diversification_ratio = len(assets) / effective_number_assets
            
            # Generate concentration alerts
            concentration_alerts = await self._generate_concentration_alerts(
                normalized_weights, herfindahl_index, max_weight, max_weight_asset
            )
            
            # Sector concentration if mapping provided
            if sector_mapping:
                sector_concentration = self._calculate_sector_concentration(
                    normalized_weights, sector_mapping
                )
                # Add sector alerts if needed
                sector_alerts = await self._generate_sector_concentration_alerts(sector_concentration)
                concentration_alerts.extend(sector_alerts)
            
            concentration_metrics = ConcentrationMetrics(
                herfindahl_index=herfindahl_index,
                entropy_measure=entropy_measure,
                max_weight=max_weight,
                max_weight_asset=max_weight_asset,
                top_3_concentration=top_3_concentration,
                top_5_concentration=top_5_concentration,
                effective_number_assets=effective_number_assets,
                diversification_ratio=diversification_ratio,
                concentration_alerts=concentration_alerts
            )
            
            # Cache the result
            cache_key = f"concentration_{hash(str(sorted(portfolio_weights.items())))}"
            self.concentration_cache[cache_key] = concentration_metrics
            
            logger.info("Concentration metrics calculated",
                       herfindahl_index=herfindahl_index,
                       max_weight=max_weight,
                       max_weight_asset=max_weight_asset,
                       effective_assets=effective_number_assets,
                       alerts_count=len(concentration_alerts))
            
            return concentration_metrics
            
        except Exception as e:
            logger.error("Concentration metrics calculation failed", error=str(e))
            raise
    
    async def generate_portfolio_heatmap(self,
                                       correlation_matrix: CorrelationMatrix,
                                       portfolio_weights: Dict[str, float],
                                       risk_contributions: Optional[Dict[str, float]] = None) -> PortfolioHeatMap:
        """Generate comprehensive portfolio heat map visualizations."""
        try:
            # Correlation heatmap
            correlation_heatmap = self._create_correlation_heatmap(correlation_matrix)
            
            # Concentration heatmap
            concentration_heatmap = self._create_concentration_heatmap(portfolio_weights)
            
            # Risk contribution heatmap
            risk_contribution_heatmap = None
            if risk_contributions:
                risk_contribution_heatmap = self._create_risk_contribution_heatmap(
                    risk_contributions, portfolio_weights
                )
            
            heatmap = PortfolioHeatMap(
                correlation_heatmap=correlation_heatmap,
                concentration_heatmap=concentration_heatmap,
                risk_contribution_heatmap=risk_contribution_heatmap
            )
            
            logger.info("Portfolio heatmap generated",
                       n_assets=len(portfolio_weights),
                       has_risk_contributions=risk_contributions is not None)
            
            return heatmap
            
        except Exception as e:
            logger.error("Portfolio heatmap generation failed", error=str(e))
            raise
    
    async def monitor_correlations(self,
                                 correlation_matrix: CorrelationMatrix) -> List[CorrelationAlert]:
        """Monitor correlations and generate alerts for threshold breaches."""
        try:
            alerts = []
            symbols = correlation_matrix.symbols
            corr_matrix = correlation_matrix.matrix
            
            # Check pairwise correlations
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    correlation = corr_matrix[i, j]
                    asset_pair = (symbols[i], symbols[j])
                    
                    # Generate alerts based on thresholds
                    if abs(correlation) >= self.config.correlation_threshold_critical:
                        alert = CorrelationAlert(
                            alert_id=f"corr_critical_{symbols[i]}_{symbols[j]}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                            alert_type="high_correlation",
                            severity=AlertSeverity.CRITICAL,
                            asset_pair=asset_pair,
                            correlation_value=correlation,
                            threshold=self.config.correlation_threshold_critical,
                            message=f"Critical correlation detected: {symbols[i]} vs {symbols[j]} = {correlation:.3f}",
                            metadata={
                                'method': correlation_matrix.method.value,
                                'lookback_days': correlation_matrix.lookback_days,
                                'p_value': correlation_matrix.p_values[i, j] if correlation_matrix.p_values is not None else None
                            }
                        )
                        alerts.append(alert)
                        
                    elif abs(correlation) >= self.config.correlation_threshold_high:
                        alert = CorrelationAlert(
                            alert_id=f"corr_high_{symbols[i]}_{symbols[j]}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                            alert_type="high_correlation",
                            severity=AlertSeverity.HIGH,
                            asset_pair=asset_pair,
                            correlation_value=correlation,
                            threshold=self.config.correlation_threshold_high,
                            message=f"High correlation detected: {symbols[i]} vs {symbols[j]} = {correlation:.3f}",
                            metadata={
                                'method': correlation_matrix.method.value,
                                'lookback_days': correlation_matrix.lookback_days
                            }
                        )
                        alerts.append(alert)
            
            # Check for correlation clustering
            clustering_alerts = await self._detect_correlation_clusters(correlation_matrix)
            alerts.extend(clustering_alerts)
            
            # Update active alerts
            for alert in alerts:
                self.active_alerts[alert.alert_id] = alert
                self.alert_history.append(alert)
            
            logger.info("Correlation monitoring completed",
                       total_alerts=len(alerts),
                       critical_alerts=len([a for a in alerts if a.severity == AlertSeverity.CRITICAL]),
                       high_alerts=len([a for a in alerts if a.severity == AlertSeverity.HIGH]))
            
            return alerts
            
        except Exception as e:
            logger.error("Correlation monitoring failed", error=str(e))
            raise
    
    async def detect_regime_changes(self,
                                  returns_data: Dict[str, List[float]],
                                  window_size: int = 60) -> Dict[str, Any]:
        """Detect correlation regime changes using rolling correlations."""
        try:
            symbols = list(returns_data.keys())
            returns_matrix = self._prepare_returns_matrix(returns_data)
            
            if len(symbols) < 2:
                return {'regime_changes': [], 'current_regime': 'stable'}
            
            # Calculate rolling correlations
            rolling_correlations = []
            dates = []
            
            for i in range(window_size, returns_matrix.shape[0]):
                window_data = returns_matrix[i-window_size:i]
                corr_matrix = np.corrcoef(window_data.T)
                
                # Average correlation (excluding diagonal)
                mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                avg_correlation = np.mean(np.abs(corr_matrix[mask]))
                
                rolling_correlations.append(avg_correlation)
                dates.append(i)
            
            # Detect regime changes using change point detection
            regime_changes = self._detect_change_points(rolling_correlations)
            
            # Classify current regime
            recent_correlation = rolling_correlations[-10:] if len(rolling_correlations) >= 10 else rolling_correlations
            avg_recent_correlation = np.mean(recent_correlation)
            
            if avg_recent_correlation > 0.8:
                current_regime = 'high_correlation'
            elif avg_recent_correlation > 0.5:
                current_regime = 'medium_correlation'
            else:
                current_regime = 'low_correlation'
            
            return {
                'regime_changes': regime_changes,
                'current_regime': current_regime,
                'rolling_correlations': rolling_correlations,
                'average_correlation': avg_recent_correlation,
                'correlation_volatility': np.std(rolling_correlations) if len(rolling_correlations) > 1 else 0
            }
            
        except Exception as e:
            logger.error("Regime change detection failed", error=str(e))
            raise
    
    # Helper methods
    
    def _prepare_returns_matrix(self, returns_data: Dict[str, List[float]], 
                               lookback_days: Optional[int] = None) -> np.ndarray:
        """Prepare returns matrix from dictionary data."""
        symbols = list(returns_data.keys())
        
        # Find minimum length
        min_length = min(len(returns) for returns in returns_data.values())
        
        if lookback_days:
            min_length = min(min_length, lookback_days)
        
        # Create matrix
        returns_matrix = np.array([
            returns_data[symbol][-min_length:] for symbol in symbols
        ]).T
        
        # Remove NaN and infinite values
        returns_matrix = returns_matrix[~np.any(np.isnan(returns_matrix), axis=1)]
        returns_matrix = returns_matrix[~np.any(np.isinf(returns_matrix), axis=1)]
        
        return returns_matrix
    
    def _calculate_pearson_correlation(self, returns_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Pearson correlation matrix with p-values."""
        n_assets = returns_matrix.shape[1]
        corr_matrix = np.zeros((n_assets, n_assets))
        p_values = np.zeros((n_assets, n_assets))
        
        for i in range(n_assets):
            for j in range(n_assets):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    p_values[i, j] = 0.0
                else:
                    corr, p_val = pearsonr(returns_matrix[:, i], returns_matrix[:, j])
                    corr_matrix[i, j] = corr
                    p_values[i, j] = p_val
        
        return corr_matrix, p_values
    
    def _calculate_spearman_correlation(self, returns_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Spearman rank correlation matrix."""
        n_assets = returns_matrix.shape[1]
        corr_matrix = np.zeros((n_assets, n_assets))
        p_values = np.zeros((n_assets, n_assets))
        
        for i in range(n_assets):
            for j in range(n_assets):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    p_values[i, j] = 0.0
                else:
                    corr, p_val = spearmanr(returns_matrix[:, i], returns_matrix[:, j])
                    corr_matrix[i, j] = corr
                    p_values[i, j] = p_val
        
        return corr_matrix, p_values
    
    def _calculate_rolling_correlation(self, returns_matrix: np.ndarray) -> np.ndarray:
        """Calculate rolling correlation matrix."""
        window = self.config.rolling_window
        n_obs, n_assets = returns_matrix.shape
        
        if n_obs < window:
            # Fallback to full sample correlation
            return np.corrcoef(returns_matrix.T)
        
        # Use the last window for correlation calculation
        recent_data = returns_matrix[-window:]
        return np.corrcoef(recent_data.T)
    
    def _calculate_exponential_correlation(self, returns_matrix: np.ndarray) -> np.ndarray:
        """Calculate exponentially weighted correlation matrix."""
        lambda_param = self.config.ewma_lambda
        n_obs, n_assets = returns_matrix.shape
        
        # Initialize with equal weights
        weights = np.array([(1 - lambda_param) * (lambda_param ** i) for i in range(n_obs)])
        weights = weights[::-1]  # Reverse to give more weight to recent observations
        weights = weights / np.sum(weights)  # Normalize
        
        # Calculate weighted correlation
        weighted_returns = returns_matrix * weights.reshape(-1, 1)
        return np.corrcoef(weighted_returns.T)
    
    def _calculate_correlation_confidence_intervals(self, corr_matrix: np.ndarray, 
                                                  n_obs: int) -> np.ndarray:
        """Calculate confidence intervals for correlation coefficients."""
        confidence_level = self.config.confidence_level
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        
        # Fisher transformation
        z_corr = 0.5 * np.log((1 + corr_matrix) / (1 - corr_matrix + 1e-10))
        se_z = 1.0 / np.sqrt(n_obs - 3)
        
        # Confidence intervals in z-space
        z_lower = z_corr - z_score * se_z
        z_upper = z_corr + z_score * se_z
        
        # Transform back to correlation space
        ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        return np.stack([ci_lower, ci_upper], axis=2)
    
    async def _generate_concentration_alerts(self, weights: Dict[str, float],
                                           herfindahl_index: float,
                                           max_weight: float,
                                           max_weight_asset: str) -> List[ConcentrationAlert]:
        """Generate concentration risk alerts."""
        alerts = []
        
        # Max weight alert
        if max_weight >= self.config.concentration_threshold_high:
            alert = ConcentrationAlert(
                alert_id=f"conc_high_{max_weight_asset}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                alert_type="high_concentration",
                severity=AlertSeverity.HIGH,
                asset=max_weight_asset,
                concentration_value=max_weight,
                threshold=self.config.concentration_threshold_high,
                message=f"High concentration in {max_weight_asset}: {max_weight:.1%}"
            )
            alerts.append(alert)
        elif max_weight >= self.config.concentration_threshold_medium:
            alert = ConcentrationAlert(
                alert_id=f"conc_medium_{max_weight_asset}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                alert_type="medium_concentration",
                severity=AlertSeverity.MEDIUM,
                asset=max_weight_asset,
                concentration_value=max_weight,
                threshold=self.config.concentration_threshold_medium,
                message=f"Medium concentration in {max_weight_asset}: {max_weight:.1%}"
            )
            alerts.append(alert)
        
        # Herfindahl index alert
        if herfindahl_index > 0.5:  # Very concentrated
            alert = ConcentrationAlert(
                alert_id=f"hhi_high_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                alert_type="high_herfindahl",
                severity=AlertSeverity.HIGH,
                asset=None,
                concentration_value=herfindahl_index,
                threshold=0.5,
                message=f"High portfolio concentration (HHI: {herfindahl_index:.3f})"
            )
            alerts.append(alert)
        
        return alerts
    
    async def _generate_sector_concentration_alerts(self, 
                                                  sector_concentration: Dict[str, float]) -> List[ConcentrationAlert]:
        """Generate sector concentration alerts."""
        alerts = []
        
        for sector, concentration in sector_concentration.items():
            if concentration >= 0.6:  # 60% in one sector
                alert = ConcentrationAlert(
                    alert_id=f"sector_conc_{sector}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    alert_type="sector_concentration",
                    severity=AlertSeverity.HIGH,
                    asset=sector,
                    concentration_value=concentration,
                    threshold=0.6,
                    message=f"High sector concentration in {sector}: {concentration:.1%}"
                )
                alerts.append(alert)
        
        return alerts
    
    def _calculate_sector_concentration(self, weights: Dict[str, float],
                                      sector_mapping: Dict[str, str]) -> Dict[str, float]:
        """Calculate concentration by sector."""
        sector_weights = {}
        
        for asset, weight in weights.items():
            sector = sector_mapping.get(asset, 'Unknown')
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        return sector_weights
    
    async def _detect_correlation_clusters(self, 
                                         correlation_matrix: CorrelationMatrix) -> List[CorrelationAlert]:
        """Detect correlation clusters using hierarchical clustering."""
        try:
            alerts = []
            corr_matrix = correlation_matrix.matrix
            symbols = correlation_matrix.symbols
            
            # Convert correlation to distance
            distance_matrix = 1 - np.abs(corr_matrix)
            
            # Perform hierarchical clustering
            condensed_distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Form clusters
            clusters = fcluster(linkage_matrix, t=0.3, criterion='distance')  # 70% correlation threshold
            
            # Identify clusters with more than 2 assets
            unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
            large_clusters = unique_clusters[cluster_counts > 2]
            
            for cluster_id in large_clusters:
                cluster_assets = [symbols[i] for i in range(len(symbols)) if clusters[i] == cluster_id]
                
                # Calculate average intra-cluster correlation
                cluster_indices = [i for i in range(len(symbols)) if clusters[i] == cluster_id]
                cluster_corrs = []
                for i in range(len(cluster_indices)):
                    for j in range(i + 1, len(cluster_indices)):
                        cluster_corrs.append(abs(corr_matrix[cluster_indices[i], cluster_indices[j]]))
                
                avg_cluster_corr = np.mean(cluster_corrs) if cluster_corrs else 0
                
                if avg_cluster_corr > 0.7:
                    alert = CorrelationAlert(
                        alert_id=f"cluster_{cluster_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                        alert_type="correlation_cluster",
                        severity=AlertSeverity.MEDIUM,
                        asset_pair=tuple(cluster_assets[:2]),  # Representative pair
                        correlation_value=avg_cluster_corr,
                        threshold=0.7,
                        message=f"High correlation cluster detected: {', '.join(cluster_assets)} (avg corr: {avg_cluster_corr:.3f})",
                        metadata={'cluster_assets': cluster_assets, 'cluster_size': len(cluster_assets)}
                    )
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error("Correlation cluster detection failed", error=str(e))
            return []
    
    def _detect_change_points(self, time_series: List[float]) -> List[Dict[str, Any]]:
        """Detect change points in correlation time series."""
        try:
            if len(time_series) < 20:
                return []
            
            # Simple change point detection using moving averages
            window = 10
            change_points = []
            
            for i in range(window, len(time_series) - window):
                before_mean = np.mean(time_series[i-window:i])
                after_mean = np.mean(time_series[i:i+window])
                
                # Detect significant change
                if abs(after_mean - before_mean) > 0.2:  # 20% change threshold
                    change_points.append({
                        'index': i,
                        'before_mean': before_mean,
                        'after_mean': after_mean,
                        'change_magnitude': after_mean - before_mean
                    })
            
            return change_points
            
        except Exception as e:
            logger.error("Change point detection failed", error=str(e))
            return []
    
    def _create_correlation_heatmap(self, correlation_matrix: CorrelationMatrix) -> Dict[str, Any]:
        """Create correlation heatmap visualization data."""
        try:
            corr_matrix = correlation_matrix.matrix
            symbols = correlation_matrix.symbols
            
            if not PLOTLY_AVAILABLE:
                # Return basic data structure without plotly
                return {
                    'matrix': corr_matrix.tolist(),
                    'symbols': symbols,
                    'title': f'Correlation Matrix ({correlation_matrix.method.value.title()})',
                    'type': 'correlation_heatmap',
                    'plotly_available': False
                }
            
            # Create heatmap data
            heatmap_data = {
                'z': corr_matrix.tolist(),
                'x': symbols,
                'y': symbols,
                'type': 'heatmap',
                'colorscale': 'RdBu',
                'zmid': 0,
                'zmin': -1,
                'zmax': 1,
                'text': [[f'{corr_matrix[i,j]:.3f}' for j in range(len(symbols))] for i in range(len(symbols))],
                'texttemplate': '%{text}',
                'textfont': {'size': 10},
                'hoverongaps': False
            }
            
            layout = {
                'title': f'Correlation Matrix ({correlation_matrix.method.value.title()})',
                'xaxis': {'title': 'Assets', 'side': 'bottom'},
                'yaxis': {'title': 'Assets'},
                'width': 600,
                'height': 600
            }
            
            return {
                'data': [heatmap_data],
                'layout': layout,
                'config': {'displayModeBar': True},
                'plotly_available': True
            }
            
        except Exception as e:
            logger.error("Correlation heatmap creation failed", error=str(e))
            return {}
    
    def _create_concentration_heatmap(self, portfolio_weights: Dict[str, float]) -> Dict[str, Any]:
        """Create concentration heatmap visualization data."""
        try:
            # Normalize weights
            total_weight = sum(abs(w) for w in portfolio_weights.values())
            normalized_weights = {asset: abs(weight) / total_weight 
                                for asset, weight in portfolio_weights.items()}
            
            # Sort by weight
            sorted_weights = sorted(normalized_weights.items(), key=lambda x: x[1], reverse=True)
            
            assets = [item[0] for item in sorted_weights]
            weights = [item[1] for item in sorted_weights]
            
            if not PLOTLY_AVAILABLE:
                # Return basic data structure without plotly
                return {
                    'assets': assets,
                    'weights': weights,
                    'title': 'Portfolio Concentration',
                    'type': 'concentration_bar',
                    'plotly_available': False
                }
            
            # Create color scale based on concentration
            colors = []
            for weight in weights:
                if weight > 0.4:
                    colors.append('red')
                elif weight > 0.25:
                    colors.append('orange')
                elif weight > 0.1:
                    colors.append('yellow')
                else:
                    colors.append('green')
            
            bar_data = {
                'x': assets,
                'y': weights,
                'type': 'bar',
                'marker': {'color': colors},
                'text': [f'{w:.1%}' for w in weights],
                'textposition': 'auto'
            }
            
            layout = {
                'title': 'Portfolio Concentration',
                'xaxis': {'title': 'Assets'},
                'yaxis': {'title': 'Weight', 'tickformat': '.1%'},
                'showlegend': False
            }
            
            return {
                'data': [bar_data],
                'layout': layout,
                'config': {'displayModeBar': True},
                'plotly_available': True
            }
            
        except Exception as e:
            logger.error("Concentration heatmap creation failed", error=str(e))
            return {}
    
    def _create_risk_contribution_heatmap(self, risk_contributions: Dict[str, float],
                                        portfolio_weights: Dict[str, float]) -> Dict[str, Any]:
        """Create risk contribution heatmap."""
        try:
            # Create scatter plot of weight vs risk contribution
            assets = list(risk_contributions.keys())
            weights = [portfolio_weights.get(asset, 0) for asset in assets]
            risk_contribs = [risk_contributions[asset] for asset in assets]
            
            if not PLOTLY_AVAILABLE:
                # Return basic data structure without plotly
                return {
                    'assets': assets,
                    'weights': weights,
                    'risk_contributions': risk_contribs,
                    'title': 'Weight vs Risk Contribution',
                    'type': 'risk_contribution_scatter',
                    'plotly_available': False
                }
            
            scatter_data = {
                'x': weights,
                'y': risk_contribs,
                'mode': 'markers+text',
                'type': 'scatter',
                'text': assets,
                'textposition': 'top center',
                'marker': {
                    'size': [abs(w) * 1000 for w in weights],  # Size proportional to weight
                    'color': risk_contribs,
                    'colorscale': 'Viridis',
                    'showscale': True,
                    'colorbar': {'title': 'Risk Contribution'}
                }
            }
            
            layout = {
                'title': 'Weight vs Risk Contribution',
                'xaxis': {'title': 'Portfolio Weight', 'tickformat': '.1%'},
                'yaxis': {'title': 'Risk Contribution', 'tickformat': '.1%'},
                'hovermode': 'closest'
            }
            
            return {
                'data': [scatter_data],
                'layout': layout,
                'config': {'displayModeBar': True},
                'plotly_available': True
            }
            
        except Exception as e:
            logger.error("Risk contribution heatmap creation failed", error=str(e))
            return {}


# Factory function
def create_correlation_concentration_monitor(config: CorrelationMonitorConfig = None) -> CorrelationConcentrationMonitor:
    """Create a correlation and concentration monitor instance."""
    return CorrelationConcentrationMonitor(config)


# Example usage and testing
if __name__ == "__main__":
    async def test_correlation_concentration_monitor():
        """Test the correlation and concentration monitor."""
        print("Testing Correlation and Concentration Monitor...")
        
        # Create monitor
        config = CorrelationMonitorConfig(
            correlation_threshold_high=0.7,
            concentration_threshold_high=0.4
        )
        monitor = create_correlation_concentration_monitor(config)
        
        # Generate sample data
        np.random.seed(42)
        
        # Create correlated returns
        base_returns = np.random.normal(0.001, 0.02, 252)
        returns_data = {
            'BTC': base_returns.tolist(),
            'ETH': (base_returns * 0.8 + np.random.normal(0, 0.01, 252)).tolist(),  # Correlated with BTC
            'ADA': np.random.normal(0.0005, 0.025, 252).tolist(),  # Less correlated
            'DOT': (base_returns * 0.6 + np.random.normal(0, 0.015, 252)).tolist()  # Moderately correlated
        }
        
        # Test correlation matrix calculation
        print("1. Correlation Matrix Calculation:")
        corr_matrix = await monitor.calculate_correlation_matrix(
            returns_data=returns_data,
            method=CorrelationMethod.PEARSON
        )
        
        print(f"Average correlation: {np.mean(corr_matrix.matrix[np.triu_indices_from(corr_matrix.matrix, k=1)]):.3f}")
        print(f"Condition number: {corr_matrix.condition_number:.2f}")
        print(f"Matrix shape: {corr_matrix.matrix.shape}")
        
        # Test concentration metrics
        print("\n2. Concentration Metrics:")
        portfolio_weights = {'BTC': 0.5, 'ETH': 0.3, 'ADA': 0.15, 'DOT': 0.05}
        
        concentration_metrics = await monitor.calculate_concentration_metrics(
            portfolio_weights=portfolio_weights
        )
        
        print(f"Herfindahl Index: {concentration_metrics.herfindahl_index:.3f}")
        print(f"Max Weight: {concentration_metrics.max_weight:.1%} ({concentration_metrics.max_weight_asset})")
        print(f"Effective Assets: {concentration_metrics.effective_number_assets:.1f}")
        print(f"Concentration Alerts: {len(concentration_metrics.concentration_alerts)}")
        
        # Test correlation monitoring
        print("\n3. Correlation Monitoring:")
        correlation_alerts = await monitor.monitor_correlations(corr_matrix)
        
        print(f"Total Alerts: {len(correlation_alerts)}")
        for alert in correlation_alerts:
            print(f"  - {alert.severity.value.upper()}: {alert.message}")
        
        # Test regime change detection
        print("\n4. Regime Change Detection:")
        regime_analysis = await monitor.detect_regime_changes(returns_data)
        
        print(f"Current Regime: {regime_analysis['current_regime']}")
        print(f"Average Correlation: {regime_analysis['average_correlation']:.3f}")
        print(f"Regime Changes: {len(regime_analysis['regime_changes'])}")
        
        # Test heatmap generation
        print("\n5. Heatmap Generation:")
        heatmap = await monitor.generate_portfolio_heatmap(
            correlation_matrix=corr_matrix,
            portfolio_weights=portfolio_weights
        )
        
        print(f"Correlation heatmap created: {len(heatmap.correlation_heatmap) > 0}")
        print(f"Concentration heatmap created: {len(heatmap.concentration_heatmap) > 0}")
        
        print("\nCorrelation and Concentration Monitor test completed successfully!")
    
    # Run test
    asyncio.run(test_correlation_concentration_monitor())