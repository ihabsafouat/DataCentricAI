import unittest
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tempfile
import shutil
from pathlib import Path
from ingestion.model_monitor import ModelMonitor

class TestModelMonitor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a simple CNN model for testing
        cls.model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation='softmax')
        ])
        cls.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Create test data
        cls.feature_names = [f"pixel_{i}" for i in range(784)]
        cls.class_names = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4',
                          'class_5', 'class_6', 'class_7', 'class_8', 'class_9']
        
        # Create temporary directory for test outputs
        cls.test_dir = tempfile.mkdtemp()
        
        # Initialize ModelMonitor
        cls.monitor = ModelMonitor(
            model=cls.model,
            feature_names=cls.feature_names,
            class_names=cls.class_names,
            output_dir=cls.test_dir
        )
        
        # Generate test data
        cls.x_test = np.random.random((100, 28, 28, 1))
        cls.y_test = np.random.randint(0, 10, size=(100,))
        cls.predictions = cls.model.predict(cls.x_test)

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary directory
        shutil.rmtree(cls.test_dir)

    def test_detect_data_drift(self):
        # Create reference and current data
        reference_data = np.random.random((100, 784))
        current_data = np.random.random((100, 784))
        
        # Test drift detection
        drift_results = self.monitor.detect_data_drift(reference_data, current_data)
        
        # Verify results structure
        self.assertIsInstance(drift_results, dict)
        self.assertTrue(all(key in drift_results for key in self.feature_names))
        self.assertTrue('pca_drift' in drift_results)
        
        # Verify drift metrics
        for feature in self.feature_names:
            self.assertIn('drift_detected', drift_results[feature])
            self.assertIn('drift_score', drift_results[feature])
            self.assertIn('ks_statistic', drift_results[feature])
            self.assertIn('p_value', drift_results[feature])

    def test_explain_prediction(self):
        # Test explanation generation
        explanation = self.monitor.explain_prediction(
            input_data=self.x_test,
            sample_index=0,
            num_features=10
        )
        
        # Verify explanation structure
        self.assertIsInstance(explanation, dict)
        self.assertIn('timestamp', explanation)
        self.assertIn('sample_index', explanation)
        self.assertIn('shap_values', explanation)
        self.assertIn('lime_explanation', explanation)
        
        # Verify explanation files were created
        explainability_dir = Path(self.test_dir) / "explainability"
        self.assertTrue(any(explainability_dir.glob("shap_explanation_*.png")))
        self.assertTrue(any(explainability_dir.glob("lime_explanation_*.png")))
        self.assertTrue(any(explainability_dir.glob("explanation_*.json")))

    def test_monitor_performance(self):
        # Test performance monitoring
        metadata = {"version": "1.0", "timestamp": "2024-01-01"}
        performance_results = self.monitor.monitor_performance(
            test_data=(self.x_test, self.y_test),
            predictions=self.predictions,
            metadata=metadata
        )
        
        # Verify results structure
        self.assertIsInstance(performance_results, dict)
        self.assertIn('timestamp', performance_results)
        self.assertIn('metrics', performance_results)
        self.assertIn('metadata', performance_results)
        self.assertIn('visualization_paths', performance_results)
        
        # Verify metrics
        metrics = performance_results['metrics']
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('confusion_matrix', metrics)
        
        # Verify visualization files were created
        performance_dir = Path(self.test_dir) / "performance"
        self.assertTrue(any(performance_dir.glob("confusion_matrix_*.png")))
        self.assertTrue(any(performance_dir.glob("roc_curves_*.png")))
        self.assertTrue(any(performance_dir.glob("performance_*.json")))

    def test_get_performance_trends(self):
        # Test performance trend analysis
        trends = self.monitor.get_performance_trends(days=7)
        
        # Verify results structure
        self.assertIsInstance(trends, dict)
        self.assertIn('trends', trends)
        self.assertIn('visualization_path', trends)
        
        # Verify trend data
        trend_data = trends['trends']
        self.assertIn('accuracy', trend_data)
        self.assertIn('precision', trend_data)
        self.assertIn('recall', trend_data)
        self.assertIn('f1', trend_data)
        self.assertIn('timestamps', trend_data)
        
        # Verify visualization file was created
        performance_dir = Path(self.test_dir) / "performance"
        self.assertTrue(any(performance_dir.glob("trends_*.png")))

if __name__ == '__main__':
    unittest.main() 