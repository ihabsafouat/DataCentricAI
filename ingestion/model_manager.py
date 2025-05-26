import os
import logging
import mlflow
import mlflow.keras
from pathlib import Path
from datetime import datetime, timedelta
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class RollbackCondition:
    def __init__(
        self,
        metric_name: str,
        threshold: float,
        comparison: str = "less_than",
        window_size: int = 3,
        consecutive_failures: int = 2
    ):
        self.metric_name = metric_name
        self.threshold = threshold
        self.comparison = comparison
        self.window_size = window_size
        self.consecutive_failures = consecutive_failures

    def check_condition(self, current_value: float) -> bool:
        if self.comparison == "less_than":
            return current_value < self.threshold
        elif self.comparison == "greater_than":
            return current_value > self.threshold
        elif self.comparison == "equals":
            return current_value == self.threshold
        return False

class ModelManager:
    def __init__(
        self,
        experiment_name: str,
        model_dir: str = "models",
        registry_dir: str = "model_registry",
        tracking_uri: str = "http://mlflow:5000",
        performance_threshold: float = 0.85,
        rollback_conditions: Optional[List[RollbackCondition]] = None
    ):
        self.experiment_name = experiment_name
        self.model_dir = Path(model_dir)
        self.registry_dir = Path(registry_dir)
        self.tracking_uri = tracking_uri
        self.performance_threshold = performance_threshold
        self.rollback_conditions = rollback_conditions or [
            RollbackCondition("accuracy", 0.85, "less_than"),
            RollbackCondition("f1", 0.80, "less_than")
        ]
        
        # Initialize MLflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        (self.registry_dir / "performance_logs").mkdir(exist_ok=True)
        (self.registry_dir / "rollback_logs").mkdir(exist_ok=True)
        (self.registry_dir / "alert_logs").mkdir(exist_ok=True)

    def add_rollback_condition(
        self,
        metric_name: str,
        threshold: float,
        comparison: str = "less_than",
        window_size: int = 3,
        consecutive_failures: int = 2
    ):
        """Add a new rollback condition."""
        condition = RollbackCondition(
            metric_name=metric_name,
            threshold=threshold,
            comparison=comparison,
            window_size=window_size,
            consecutive_failures=consecutive_failures
        )
        self.rollback_conditions.append(condition)
        logger.info(f"Added new rollback condition for {metric_name}")

    def check_rollback_conditions(
        self,
        model_name: str,
        current_metrics: Dict[str, float],
        window_size: int = 3
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Check if any rollback conditions are met."""
        try:
            # Get recent performance history
            performance_history = self.get_performance_history(model_name, days=window_size)
            if not performance_history:
                return False, None, None

            # Check each condition
            for condition in self.rollback_conditions:
                metric_value = current_metrics.get(condition.metric_name)
                if metric_value is None:
                    continue

                # Check current value
                if condition.check_condition(metric_value):
                    # Check historical values for consecutive failures
                    consecutive_failures = 0
                    for log in performance_history:
                        historical_value = log["metrics"].get(condition.metric_name)
                        if historical_value is not None and condition.check_condition(historical_value):
                            consecutive_failures += 1
                        else:
                            break

                    if consecutive_failures >= condition.consecutive_failures:
                        # Find the last good version
                        last_good_version = self._find_last_good_version(
                            model_name,
                            condition.metric_name,
                            condition.threshold,
                            condition.comparison
                        )
                        
                        if last_good_version:
                            return True, last_good_version["version"], {
                                "condition": condition.metric_name,
                                "current_value": metric_value,
                                "threshold": condition.threshold,
                                "consecutive_failures": consecutive_failures
                            }

            return False, None, None

        except Exception as e:
            logger.error(f"Error checking rollback conditions: {str(e)}")
            raise

    def _find_last_good_version(
        self,
        model_name: str,
        metric_name: str,
        threshold: float,
        comparison: str
    ) -> Optional[Dict]:
        """Find the last version that met the performance criteria."""
        try:
            versions = self.list_model_versions(model_name)
            condition = RollbackCondition(metric_name, threshold, comparison)
            
            for version in versions:
                metrics = version["metrics"]
                if metric_name in metrics and not condition.check_condition(metrics[metric_name]):
                    return version
            return None

        except Exception as e:
            logger.error(f"Error finding last good version: {str(e)}")
            raise

    def register_model(
        self,
        model: keras.Model,
        metrics: Dict[str, float],
        model_name: str,
        version: str = None,
        tags: Dict[str, str] = None
    ) -> str:
        """Register a model in MLflow and save its metadata."""
        try:
            with mlflow.start_run():
                # Log model
                mlflow.keras.log_model(model, "model")
                
                # Log metrics
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, value)
                
                # Log tags
                if tags:
                    for tag_name, value in tags.items():
                        mlflow.set_tag(tag_name, value)
                
                # Save model locally
                model_path = self.model_dir / f"{model_name}_v{version or 'latest'}.h5"
                model.save(model_path)
                
                # Save metadata
                metadata = {
                    "model_name": model_name,
                    "version": version or "latest",
                    "timestamp": datetime.now().isoformat(),
                    "metrics": metrics,
                    "tags": tags or {},
                    "model_path": str(model_path),
                    "mlflow_run_id": mlflow.active_run().info.run_id,
                    "status": "active"  # Add status field
                }
                
                metadata_path = self.registry_dir / f"{model_name}_v{version or 'latest'}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Model {model_name} version {version} registered successfully")
                return mlflow.active_run().info.run_id
                
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            raise

    def get_model_version(self, model_name: str, version: str = "latest") -> Tuple[keras.Model, Dict]:
        """Retrieve a specific version of a model and its metadata."""
        try:
            # Load metadata
            metadata_path = self.registry_dir / f"{model_name}_v{version}_metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load model
            model_path = metadata["model_path"]
            model = keras.models.load_model(model_path)
            
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error retrieving model version: {str(e)}")
            raise

    def list_model_versions(self, model_name: str) -> List[Dict]:
        """List all versions of a model."""
        try:
            versions = []
            for metadata_file in self.registry_dir.glob(f"{model_name}_v*_metadata.json"):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    versions.append(metadata)
            return sorted(versions, key=lambda x: x["timestamp"], reverse=True)
        except Exception as e:
            logger.error(f"Error listing model versions: {str(e)}")
            raise

    def compare_models(
        self,
        model_a: keras.Model,
        model_b: keras.Model,
        test_data: Tuple[np.ndarray, np.ndarray],
        metrics: List[str] = ["accuracy", "precision", "recall", "f1"]
    ) -> Dict[str, Dict[str, float]]:
        """Compare two models using specified metrics."""
        try:
            x_test, y_test = test_data
            results = {}
            
            for model, name in [(model_a, "model_a"), (model_b, "model_b")]:
                predictions = model.predict(x_test)
                y_pred = np.argmax(predictions, axis=1)
                
                model_metrics = {}
                if "accuracy" in metrics:
                    model_metrics["accuracy"] = np.mean(y_pred == y_test)
                if "precision" in metrics:
                    model_metrics["precision"] = tf.keras.metrics.Precision()(y_test, y_pred).numpy()
                if "recall" in metrics:
                    model_metrics["recall"] = tf.keras.metrics.Recall()(y_test, y_pred).numpy()
                if "f1" in metrics:
                    model_metrics["f1"] = tf.keras.metrics.F1Score()(y_test, y_pred).numpy()
                
                results[name] = model_metrics
            
            return results
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            raise

    def setup_ab_test(
        self,
        model_a: keras.Model,
        model_b: keras.Model,
        model_a_name: str,
        model_b_name: str,
        traffic_split: float = 0.5
    ) -> Dict:
        """Set up A/B testing configuration for two models."""
        try:
            ab_config = {
                "model_a": {
                    "name": model_a_name,
                    "traffic_percentage": traffic_split
                },
                "model_b": {
                    "name": model_b_name,
                    "traffic_percentage": 1 - traffic_split
                },
                "start_time": datetime.now().isoformat(),
                "status": "active"
            }
            
            config_path = self.registry_dir / "ab_test_config.json"
            with open(config_path, 'w') as f:
                json.dump(ab_config, f, indent=2)
            
            logger.info(f"A/B test configured with traffic split: {traffic_split}")
            return ab_config
            
        except Exception as e:
            logger.error(f"Error setting up A/B test: {str(e)}")
            raise

    def get_ab_test_config(self) -> Optional[Dict]:
        """Get current A/B testing configuration."""
        try:
            config_path = self.registry_dir / "ab_test_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Error getting A/B test config: {str(e)}")
            raise

    def update_ab_test_traffic(self, model_a_traffic: float):
        """Update traffic split for A/B testing."""
        try:
            config = self.get_ab_test_config()
            if config:
                config["model_a"]["traffic_percentage"] = model_a_traffic
                config["model_b"]["traffic_percentage"] = 1 - model_a_traffic
                
                config_path = self.registry_dir / "ab_test_config.json"
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                logger.info(f"A/B test traffic split updated to: {model_a_traffic}")
                return config
            return None
        except Exception as e:
            logger.error(f"Error updating A/B test traffic: {str(e)}")
            raise

    def end_ab_test(self, winning_model: str):
        """End A/B testing and declare a winner."""
        try:
            config = self.get_ab_test_config()
            if config:
                config["status"] = "completed"
                config["end_time"] = datetime.now().isoformat()
                config["winning_model"] = winning_model
                
                config_path = self.registry_dir / "ab_test_config.json"
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                logger.info(f"A/B test ended. Winner: {winning_model}")
                return config
            return None
        except Exception as e:
            logger.error(f"Error ending A/B test: {str(e)}")
            raise

    def rollback_model(
        self,
        model_name: str,
        target_version: str,
        reason: str = "Manual rollback"
    ) -> bool:
        """Rollback to a specific model version."""
        try:
            # Get current active version
            current_versions = self.list_model_versions(model_name)
            if not current_versions:
                raise ValueError(f"No versions found for model {model_name}")
            
            current_version = current_versions[0]
            
            # Get target version
            target_model, target_metadata = self.get_model_version(model_name, target_version)
            
            # Create rollback log
            rollback_log = {
                "timestamp": datetime.now().isoformat(),
                "model_name": model_name,
                "from_version": current_version["version"],
                "to_version": target_version,
                "reason": reason,
                "metrics_before": current_version["metrics"],
                "metrics_after": target_metadata["metrics"]
            }
            
            # Save rollback log
            log_path = self.registry_dir / "rollback_logs" / f"rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_path, 'w') as f:
                json.dump(rollback_log, f, indent=2)
            
            # Update current version status
            current_metadata_path = self.registry_dir / f"{model_name}_v{current_version['version']}_metadata.json"
            with open(current_metadata_path, 'r') as f:
                current_metadata = json.load(f)
            current_metadata["status"] = "rolled_back"
            with open(current_metadata_path, 'w') as f:
                json.dump(current_metadata, f, indent=2)
            
            # Update target version status
            target_metadata["status"] = "active"
            target_metadata_path = self.registry_dir / f"{model_name}_v{target_version}_metadata.json"
            with open(target_metadata_path, 'w') as f:
                json.dump(target_metadata, f, indent=2)
            
            logger.info(f"Successfully rolled back model {model_name} to version {target_version}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back model: {str(e)}")
            raise

    def monitor_model_performance(
        self,
        model: keras.Model,
        test_data: Tuple[np.ndarray, np.ndarray],
        model_name: str,
        version: str,
        auto_rollback: bool = True
    ) -> Dict:
        """Monitor model performance and generate reports."""
        try:
            x_test, y_test = test_data
            predictions = model.predict(x_test)
            y_pred = np.argmax(predictions, axis=1)
            
            # Calculate metrics
            metrics = {
                "accuracy": np.mean(y_pred == y_test),
                "precision": tf.keras.metrics.Precision()(y_test, y_pred).numpy(),
                "recall": tf.keras.metrics.Recall()(y_test, y_pred).numpy(),
                "f1": tf.keras.metrics.F1Score()(y_test, y_pred).numpy()
            }
            
            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name} v{version}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save confusion matrix
            cm_path = self.registry_dir / "performance_logs" / f"confusion_matrix_{model_name}_v{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(cm_path)
            plt.close()
            
            # Generate classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Save performance log
            performance_log = {
                "timestamp": datetime.now().isoformat(),
                "model_name": model_name,
                "version": version,
                "metrics": metrics,
                "classification_report": report,
                "confusion_matrix_path": str(cm_path)
            }
            
            log_path = self.registry_dir / "performance_logs" / f"performance_{model_name}_v{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_path, 'w') as f:
                json.dump(performance_log, f, indent=2)
            
            # Check rollback conditions if auto_rollback is enabled
            if auto_rollback:
                should_rollback, target_version, condition_info = self.check_rollback_conditions(
                    model_name,
                    metrics
                )
                
                if should_rollback and target_version:
                    logger.warning(f"Auto-rollback triggered: {condition_info}")
                    self.rollback_model(
                        model_name=model_name,
                        target_version=target_version,
                        reason=f"Auto-rollback: {condition_info['condition']} below threshold for {condition_info['consecutive_failures']} consecutive checks"
                    )
            
            return performance_log
            
        except Exception as e:
            logger.error(f"Error monitoring model performance: {str(e)}")
            raise

    def get_performance_history(self, model_name: str, days: int = 30) -> List[Dict]:
        """Get performance history for a model."""
        try:
            performance_logs = []
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for log_file in (self.registry_dir / "performance_logs").glob(f"performance_{model_name}_*.json"):
                with open(log_file, 'r') as f:
                    log = json.load(f)
                    log_date = datetime.fromisoformat(log["timestamp"])
                    if log_date >= cutoff_date:
                        performance_logs.append(log)
            
            return sorted(performance_logs, key=lambda x: x["timestamp"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting performance history: {str(e)}")
            raise

    def get_rollback_history(self, model_name: str) -> List[Dict]:
        """Get rollback history for a model."""
        try:
            rollback_logs = []
            
            for log_file in (self.registry_dir / "rollback_logs").glob("rollback_*.json"):
                with open(log_file, 'r') as f:
                    log = json.load(f)
                    if log["model_name"] == model_name:
                        rollback_logs.append(log)
            
            return sorted(rollback_logs, key=lambda x: x["timestamp"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting rollback history: {str(e)}")
            raise 