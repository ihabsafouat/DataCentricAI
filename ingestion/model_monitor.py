import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import shap
import lime
import lime.lime_tabular
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(
        self,
        model: keras.Model,
        feature_names: List[str],
        class_names: List[str],
        output_dir: str = "monitoring",
        drift_threshold: float = 0.05,
        performance_window: int = 30
    ):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.drift_threshold = drift_threshold
        self.performance_window = performance_window
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "drift").mkdir(exist_ok=True)
        (self.output_dir / "explainability").mkdir(exist_ok=True)
        (self.output_dir / "performance").mkdir(exist_ok=True)
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None

    def detect_data_drift(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """Detect data drift between reference and current data."""
        try:
            drift_results = {}
            
            # Statistical tests for each feature
            for i, feature in enumerate(feature_names or self.feature_names):
                # Kolmogorov-Smirnov test
                ks_stat, p_value = stats.ks_2samp(
                    reference_data[:, i],
                    current_data[:, i]
                )
                
                # Calculate distribution statistics
                ref_mean = np.mean(reference_data[:, i])
                ref_std = np.std(reference_data[:, i])
                curr_mean = np.mean(current_data[:, i])
                curr_std = np.std(current_data[:, i])
                
                # Calculate drift score
                drift_score = abs(curr_mean - ref_mean) / ref_std
                
                drift_results[feature] = {
                    "ks_statistic": ks_stat,
                    "p_value": p_value,
                    "drift_detected": p_value < self.drift_threshold,
                    "drift_score": drift_score,
                    "reference_mean": ref_mean,
                    "reference_std": ref_std,
                    "current_mean": curr_mean,
                    "current_std": curr_std
                }
            
            # PCA-based drift detection
            pca = PCA(n_components=2)
            ref_pca = pca.fit_transform(reference_data)
            curr_pca = pca.transform(current_data)
            
            # Calculate distribution distance in PCA space
            pca_drift = np.mean(np.abs(curr_pca - ref_pca))
            
            drift_results["pca_drift"] = {
                "drift_score": pca_drift,
                "drift_detected": pca_drift > self.drift_threshold
            }
            
            # Save drift results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            drift_path = self.output_dir / "drift" / f"drift_results_{timestamp}.json"
            with open(drift_path, 'w') as f:
                json.dump(drift_results, f, indent=2)
            
            # Generate drift visualization
            self._plot_drift_analysis(reference_data, current_data, drift_results, timestamp)
            
            return drift_results
            
        except Exception as e:
            logger.error(f"Error detecting data drift: {str(e)}")
            raise

    def _plot_drift_analysis(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        drift_results: Dict,
        timestamp: str
    ):
        """Generate visualizations for drift analysis."""
        try:
            # Feature distribution plots
            for feature, results in drift_results.items():
                if feature == "pca_drift":
                    continue
                    
                plt.figure(figsize=(10, 6))
                sns.kdeplot(reference_data[:, self.feature_names.index(feature)], label='Reference')
                sns.kdeplot(current_data[:, self.feature_names.index(feature)], label='Current')
                plt.title(f'Distribution Drift - {feature}')
                plt.xlabel('Value')
                plt.ylabel('Density')
                plt.legend()
                
                plot_path = self.output_dir / "drift" / f"drift_plot_{feature}_{timestamp}.png"
                plt.savefig(plot_path)
                plt.close()
            
            # PCA visualization
            pca = PCA(n_components=2)
            ref_pca = pca.fit_transform(reference_data)
            curr_pca = pca.transform(current_data)
            
            plt.figure(figsize=(10, 6))
            plt.scatter(ref_pca[:, 0], ref_pca[:, 1], label='Reference', alpha=0.5)
            plt.scatter(curr_pca[:, 0], curr_pca[:, 1], label='Current', alpha=0.5)
            plt.title('PCA Drift Visualization')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend()
            
            plot_path = self.output_dir / "drift" / f"pca_drift_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting drift analysis: {str(e)}")
            raise

    def explain_prediction(
        self,
        input_data: np.ndarray,
        sample_index: int = 0,
        num_features: int = 10
    ) -> Dict:
        """Generate model explanations using SHAP and LIME."""
        try:
            # Initialize explainers if not already done
            if self.shap_explainer is None:
                self.shap_explainer = shap.DeepExplainer(self.model, input_data[:100])
            
            if self.lime_explainer is None:
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    input_data,
                    feature_names=self.feature_names,
                    class_names=self.class_names,
                    mode='classification'
                )
            
            # SHAP explanation
            shap_values = self.shap_explainer.shap_values(input_data[sample_index:sample_index+1])
            
            # LIME explanation
            lime_exp = self.lime_explainer.explain_instance(
                input_data[sample_index],
                self.model.predict,
                num_features=num_features
            )
            
            # Generate visualizations
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # SHAP plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values[0],
                input_data[sample_index:sample_index+1],
                feature_names=self.feature_names,
                show=False
            )
            plot_path = self.output_dir / "explainability" / f"shap_explanation_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close()
            
            # LIME plot
            lime_exp.as_pyplot_figure()
            plot_path = self.output_dir / "explainability" / f"lime_explanation_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close()
            
            # Save explanation results
            explanation_results = {
                "timestamp": timestamp,
                "sample_index": sample_index,
                "shap_values": {
                    feature: float(value)
                    for feature, value in zip(self.feature_names, shap_values[0][0])
                },
                "lime_explanation": {
                    str(feature): float(weight)
                    for feature, weight in lime_exp.as_list()
                }
            }
            
            results_path = self.output_dir / "explainability" / f"explanation_{timestamp}.json"
            with open(results_path, 'w') as f:
                json.dump(explanation_results, f, indent=2)
            
            return explanation_results
            
        except Exception as e:
            logger.error(f"Error generating model explanation: {str(e)}")
            raise

    def monitor_performance(
        self,
        test_data: Tuple[np.ndarray, np.ndarray],
        predictions: np.ndarray,
        metadata: Dict
    ) -> Dict:
        """Monitor model performance with detailed metrics and visualizations."""
        try:
            x_test, y_test = test_data
            y_pred = np.argmax(predictions, axis=1)
            
            # Calculate detailed metrics
            performance_metrics = {
                "accuracy": np.mean(y_pred == y_test),
                "precision": tf.keras.metrics.Precision()(y_test, y_pred).numpy(),
                "recall": tf.keras.metrics.Recall()(y_test, y_pred).numpy(),
                "f1": tf.keras.metrics.F1Score()(y_test, y_pred).numpy(),
                "confusion_matrix": tf.math.confusion_matrix(y_test, y_pred).numpy().tolist()
            }
            
            # Calculate per-class metrics
            for i, class_name in enumerate(self.class_names):
                class_metrics = {
                    "precision": tf.keras.metrics.Precision()(
                        y_test == i,
                        y_pred == i
                    ).numpy(),
                    "recall": tf.keras.metrics.Recall()(
                        y_test == i,
                        y_pred == i
                    ).numpy(),
                    "f1": tf.keras.metrics.F1Score()(
                        y_test == i,
                        y_pred == i
                    ).numpy()
                }
                performance_metrics[f"class_{class_name}"] = class_metrics
            
            # Generate visualizations
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                performance_metrics["confusion_matrix"],
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names
            )
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plot_path = self.output_dir / "performance" / f"confusion_matrix_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close()
            
            # ROC curves for each class
            plt.figure(figsize=(10, 8))
            for i, class_name in enumerate(self.class_names):
                fpr, tpr, _ = tf.keras.metrics.roc_curve(
                    y_test == i,
                    predictions[:, i]
                )
                auc = tf.keras.metrics.auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
            plt.legend()
            plot_path = self.output_dir / "performance" / f"roc_curves_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close()
            
            # Save performance results
            performance_results = {
                "timestamp": timestamp,
                "metrics": performance_metrics,
                "metadata": metadata,
                "visualization_paths": {
                    "confusion_matrix": str(plot_path),
                    "roc_curves": str(plot_path)
                }
            }
            
            results_path = self.output_dir / "performance" / f"performance_{timestamp}.json"
            with open(results_path, 'w') as f:
                json.dump(performance_results, f, indent=2)
            
            return performance_results
            
        except Exception as e:
            logger.error(f"Error monitoring performance: {str(e)}")
            raise

    def get_performance_trends(self, days: int = 30) -> Dict:
        """Analyze performance trends over time."""
        try:
            performance_files = sorted(
                (self.output_dir / "performance").glob("performance_*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )[:days]
            
            trends = {
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": [],
                "timestamps": []
            }
            
            for file in performance_files:
                with open(file, 'r') as f:
                    data = json.load(f)
                    metrics = data["metrics"]
                    trends["accuracy"].append(metrics["accuracy"])
                    trends["precision"].append(metrics["precision"])
                    trends["recall"].append(metrics["recall"])
                    trends["f1"].append(metrics["f1"])
                    trends["timestamps"].append(data["timestamp"])
            
            # Generate trend visualization
            plt.figure(figsize=(12, 6))
            for metric in ["accuracy", "precision", "recall", "f1"]:
                plt.plot(trends["timestamps"], trends[metric], label=metric)
            
            plt.title('Performance Trends')
            plt.xlabel('Time')
            plt.ylabel('Score')
            plt.legend()
            plt.xticks(rotation=45)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.output_dir / "performance" / f"trends_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close()
            
            return {
                "trends": trends,
                "visualization_path": str(plot_path)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance trends: {str(e)}")
            raise 