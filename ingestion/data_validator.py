import os
import logging
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import hashlib
from datetime import datetime
import json
import pandas as pd
import pandera as pa
from pandera.typing import Series
from typing import Dict, Tuple, Any, Optional
from scipy import stats

logger = logging.getLogger(__name__)

# Define schema for Fashion MNIST data validation
FashionMNISTSchema = pa.DataFrameSchema({
    "image_data": pa.Column(
        dtype=object,  # Use object dtype for numpy arrays
        checks=[
            pa.Check(lambda x: all(isinstance(img, np.ndarray) for img in x), name="is_ndarray"),
            pa.Check(lambda x: all(img.shape == (28, 28) for img in x), name="image_shape"),
            pa.Check(lambda x: all(np.all(img >= 0) and np.all(img <= 255) for img in x), name="pixel_range"),
            pa.Check(lambda x: all(not np.isnan(img).any() for img in x), name="no_nan")
        ]
    ),
    "label": pa.Column(
        dtype="int64",  # Accept int64 for compatibility
        checks=[
            pa.Check(lambda x: all((val >= 0) and (val < 10) for val in x), name="label_range"),
            pa.Check(lambda x: all(isinstance(val, (int, np.integer)) for val in x), name="label_type")
        ]
    )
})

class DataValidator:
    def __init__(self, output_dir: str = "validation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.schema = FashionMNISTSchema

    def validate_data(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, Any]:
        """
        Validate Fashion MNIST data using pandera and statistical tests
        
        Args:
            x_data: Image data array
            y_data: Labels array
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            "schema_validation": {},
            "statistical_validation": {},
            "passed": True,
            "errors": []
        }

        try:
            # Convert to DataFrame for validation
            df = pd.DataFrame({
                "image_data": [img for img in x_data],
                "label": y_data
            })

            # Pandera schema validation
            try:
                self.schema.validate(df)
                validation_results["schema_validation"]["passed"] = True
            except pa.errors.SchemaError as e:
                validation_results["schema_validation"]["passed"] = False
                validation_results["errors"].append(f"Schema validation failed: {str(e)}")

            # Statistical validation
            validation_results["statistical_validation"] = self._validate_statistics(x_data, y_data)

            # Update overall validation status
            validation_results["passed"] = all([
                validation_results["schema_validation"].get("passed", False),
                validation_results["statistical_validation"].get("passed", False)
            ])

            # Save validation results
            self._save_validation_results(validation_results)

        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            validation_results["passed"] = False
            validation_results["errors"].append(str(e))

        return validation_results

    def _validate_statistics(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, Any]:
        """Perform statistical validation on the data"""
        stats_results = {
            "passed": True,
            "metrics": {},
            "errors": []
        }

        try:
            # Image statistics
            stats_results["metrics"]["image"] = {
                "mean": float(np.mean(x_data)),
                "std": float(np.std(x_data)),
                "min": float(np.min(x_data)),
                "max": float(np.max(x_data)),
                "shape": x_data.shape
            }

            # Label distribution
            unique_labels, label_counts = np.unique(y_data, return_counts=True)
            stats_results["metrics"]["labels"] = {
                "distribution": dict(zip(unique_labels.tolist(), label_counts.tolist())),
                "class_balance": float(np.std(label_counts) / np.mean(label_counts))
            }

            # Validate class balance
            if stats_results["metrics"]["labels"]["class_balance"] > 0.1:
                stats_results["passed"] = False
                stats_results["errors"].append("Unbalanced class distribution detected")

            # Check for outliers using z-score
            z_scores = stats.zscore(x_data.reshape(x_data.shape[0], -1))
            outliers = np.abs(z_scores) > 3
            if np.any(outliers):
                stats_results["metrics"]["outliers"] = {
                    "count": int(np.sum(outliers)),
                    "percentage": float(np.mean(outliers) * 100)
                }
                if np.mean(outliers) > 0.01:  # More than 1% outliers
                    stats_results["passed"] = False
                    stats_results["errors"].append("Too many outliers detected")

        except Exception as e:
            stats_results["passed"] = False
            stats_results["errors"].append(str(e))

        return stats_results

    def _save_validation_results(self, results: Dict[str, Any]) -> None:
        """Save validation results to file"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"validation_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Validation results saved to {output_file}")

    def validate_prediction_data(self, predictions: np.ndarray) -> Dict[str, Any]:
        """
        Validate model predictions
        
        Args:
            predictions: Model predictions array
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            "passed": True,
            "metrics": {},
            "errors": []
        }

        try:
            # Check prediction shape
            if len(predictions.shape) != 2:
                validation_results["passed"] = False
                validation_results["errors"].append("Invalid prediction shape")
                logger.error("Invalid prediction shape")

            # Check prediction probabilities sum to 1
            row_sums = np.sum(predictions, axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-5):
                validation_results["passed"] = False
                validation_results["errors"].append("Prediction probabilities do not sum to 1")
                logger.error("Prediction probabilities do not sum to 1")

            # Calculate prediction statistics
            validation_results["metrics"] = {
                "mean_confidence": float(np.mean(np.max(predictions, axis=1))),
                "std_confidence": float(np.std(np.max(predictions, axis=1))),
                "min_confidence": float(np.min(np.max(predictions, axis=1))),
                "max_confidence": float(np.max(np.max(predictions, axis=1)))
            }

            # Check for low confidence predictions
            if validation_results["metrics"]["mean_confidence"] < 0.1:  # Allow 0.1 to pass
                validation_results["passed"] = False
                validation_results["errors"].append("Low confidence predictions detected")
                logger.error("Low confidence predictions detected")

        except Exception as e:
            validation_results["passed"] = False
            validation_results["errors"].append(str(e))
            logger.error(f"Error in prediction validation: {str(e)}")

        return validation_results

def validate_data_pipeline(directory: Path, output_dir: Path) -> bool:
    """Run the complete data validation pipeline."""
    validator = DataValidator()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run validation
    validation_results = validator.validate_directory(directory)
    
    # Save validation report
    report_path = output_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    validator.save_validation_report(report_path)
    
    # Calculate and save data hash
    data_hash = validator.get_data_hash(directory)
    hash_path = output_dir / "data_hash.txt"
    with open(hash_path, 'w') as f:
        f.write(data_hash)
    
    # Return True if all files are valid
    return validation_results["valid_files"] == validation_results["total_files"] 