import pytest
import numpy as np
from keras.datasets import fashion_mnist
from ingestion.data_validator import DataValidator, FashionMNISTSchema
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import pandera as pa

@pytest.fixture
def sample_data():
    # Create sample Fashion MNIST data
    x_data = np.random.randint(0, 256, size=(100, 28, 28), dtype=np.uint8)
    y_data = np.random.randint(0, 10, size=(100,), dtype=np.int64)  # Cast to int64
    return x_data, y_data

@pytest.fixture
def validator():
    return DataValidator(output_dir="test_validation_results")

def test_schema_validation(validator, sample_data):
    x_data, y_data = sample_data
    results = validator.validate_data(x_data, y_data)
    
    assert isinstance(results, dict)
    assert "schema_validation" in results
    assert "statistical_validation" in results
    assert "passed" in results
    assert "errors" in results

def test_statistical_validation(validator, sample_data):
    x_data, y_data = sample_data
    results = validator.validate_data(x_data, y_data)
    
    assert "statistical_validation" in results
    stats_results = results["statistical_validation"]
    
    assert "metrics" in stats_results
    assert "image" in stats_results["metrics"]
    assert "labels" in stats_results["metrics"]
    
    # Check image statistics
    image_stats = stats_results["metrics"]["image"]
    assert "mean" in image_stats
    assert "std" in image_stats
    assert "min" in image_stats
    assert "max" in image_stats
    
    # Check label statistics
    label_stats = stats_results["metrics"]["labels"]
    assert "distribution" in label_stats
    assert "class_balance" in label_stats

def test_outlier_detection(validator):
    # Create data with outliers
    x_data = np.random.randint(0, 256, size=(100, 28, 28), dtype=np.uint8)
    # Add some outliers
    x_data[0:5] = 0  # Add some zero images
    x_data[5:10] = 255  # Add some white images
    y_data = np.random.randint(0, 10, size=(100,), dtype=np.int32)
    
    results = validator.validate_data(x_data, y_data)
    stats_results = results["statistical_validation"]
    
    assert "metrics" in stats_results
    if "outliers" in stats_results["metrics"]:
        assert "count" in stats_results["metrics"]["outliers"]
        assert "percentage" in stats_results["metrics"]["outliers"]

def test_prediction_validation(validator):
    # Create sample predictions
    predictions = np.random.rand(100, 10)
    predictions = predictions / predictions.sum(axis=1, keepdims=True)  # Normalize to probabilities
    
    results = validator.validate_prediction_data(predictions)
    
    assert isinstance(results, dict)
    assert "passed" in results
    assert "metrics" in results
    assert "errors" in results
    
    # Check prediction metrics
    metrics = results["metrics"]
    assert "mean_confidence" in metrics
    assert "std_confidence" in metrics
    assert "min_confidence" in metrics
    assert "max_confidence" in metrics

def test_invalid_predictions(validator):
    # Test with invalid predictions (probabilities don't sum to 1)
    predictions = np.random.rand(100, 10)
    
    results = validator.validate_prediction_data(predictions)
    
    assert not results["passed"]
    assert len(results["errors"]) > 0
    assert any("do not sum to 1" in error for error in results["errors"])

def test_schema_validation():
    # Test the FashionMNISTSchema
    schema = FashionMNISTSchema
    
    # Test valid data
    valid_data = pd.DataFrame({
        "image_data": [np.random.randint(0, 256, size=(28, 28), dtype=np.uint8) for _ in range(10)],
        "label": np.random.randint(0, 10, size=(10,), dtype=np.int64)  # Cast to int64
    })
    
    # This should not raise an exception
    schema.validate(valid_data)
    
    # Test invalid data
    invalid_data = pd.DataFrame({
        "image_data": [np.random.randint(0, 256, size=(28, 28), dtype=np.uint8) for _ in range(10)],
        "label": np.random.randint(10, 20, size=(10,), dtype=np.int64)  # Invalid labels
    })
    
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(invalid_data)

def test_prediction_validation_with_known_good_predictions(validator):
    # Create known good predictions with boosted confidence
    predictions = np.full((100, 10), 0.1 / 9)  # Distribute 0.1 across 9 classes
    predictions[:, 0] = 0.9  # Set the first class to 0.9
    results = validator.validate_prediction_data(predictions)
    print(f"Actual mean confidence: {results['metrics']['mean_confidence']}")
    assert results["passed"]
    assert "mean_confidence" in results["metrics"]
    assert np.isclose(results["metrics"]["mean_confidence"], 0.9, atol=1e-10)

class TestDataValidation:
    @classmethod
    def setup_class(cls):
        # Load Fashion MNIST data
        (cls.x_train, cls.y_train), (cls.x_test, cls.y_test) = fashion_mnist.load_data()
        cls.y_train = cls.y_train.astype(np.int64)
        cls.y_test = cls.y_test.astype(np.int64)
        # Create temporary directory for validation results
        cls.test_dir = tempfile.mkdtemp()
        # Initialize validator
        cls.validator = DataValidator(output_dir=cls.test_dir)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.test_dir)

    def test_schema_validation(self):
        train_results = self.validator.validate_data(self.x_train, self.y_train)
        assert train_results["schema_validation"]["passed"]
        test_results = self.validator.validate_data(self.x_test, self.y_test)
        assert test_results["schema_validation"]["passed"]

    def test_statistical_validation(self):
        train_results = self.validator.validate_data(self.x_train, self.y_train)
        stats = train_results["statistical_validation"]
        assert "image" in stats["metrics"]
        assert "mean" in stats["metrics"]["image"]
        assert "std" in stats["metrics"]["image"]
        assert "labels" in stats["metrics"]
        assert "distribution" in stats["metrics"]["labels"]
        assert "class_balance" in stats["metrics"]["labels"]
        assert stats["metrics"]["labels"]["class_balance"] < 0.1

    def test_prediction_validation(self):
        predictions = np.random.random((100, 10))
        predictions = predictions / predictions.sum(axis=1, keepdims=True)
        results = self.validator.validate_prediction_data(predictions)
        assert results["passed"]
        assert "mean_confidence" in results["metrics"]
        assert "std_confidence" in results["metrics"]
        invalid_predictions = np.random.random((100, 10))
        results = self.validator.validate_prediction_data(invalid_predictions)
        assert not results["passed"]
        assert any("Prediction probabilities do not sum to 1" in err for err in results["errors"])

    def test_validation_output(self):
        results = self.validator.validate_data(self.x_train, self.y_train)
        output_files = list(Path(self.test_dir).glob("validation_results_*.json"))
        assert len(output_files) > 0
        assert "schema_validation" in results
        assert "statistical_validation" in results
        assert "passed" in results
        assert "errors" in results

    def test_error_handling(self):
        invalid_x = np.random.random((100, 20, 20))
        invalid_y = np.random.randint(0, 20, size=(100,)).astype(np.int64)
        results = self.validator.validate_data(invalid_x, invalid_y)
        assert not results["passed"]
        assert len(results["errors"]) > 0

if __name__ == '__main__':
    pytest.main() 