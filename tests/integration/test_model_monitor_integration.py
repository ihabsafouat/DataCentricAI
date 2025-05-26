import unittest
import numpy as np
from tensorflow import keras
import tempfile
import shutil
from ingestion.model_monitor import ModelMonitor

class TestModelMonitorIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a simple CNN model for integration testing
        cls.model = keras.Sequential([
            keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation='softmax')
        ])
        cls.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        cls.feature_names = [f"pixel_{i}" for i in range(784)]
        cls.class_names = [f'class_{i}' for i in range(10)]
        cls.test_dir = tempfile.mkdtemp()
        cls.monitor = ModelMonitor(
            model=cls.model,
            feature_names=cls.feature_names,
            class_names=cls.class_names,
            output_dir=cls.test_dir
        )
        cls.x_test = np.random.random((50, 28, 28, 1))
        cls.y_test = np.random.randint(0, 10, size=(50,))
        cls.predictions = cls.model.predict(cls.x_test)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def test_end_to_end_monitoring(self):
        # Performance monitoring
        perf = self.monitor.monitor_performance(
            test_data=(self.x_test, self.y_test),
            predictions=self.predictions,
            metadata={"integration": True}
        )
        self.assertIn('metrics', perf)
        # Drift detection
        drift = self.monitor.detect_data_drift(
            reference_data=self.predictions,
            current_data=self.predictions + np.random.normal(0, 0.01, self.predictions.shape)
        )
        self.assertIn('pca_drift', drift)
        # Explainability
        explanation = self.monitor.explain_prediction(
            input_data=self.x_test,
            sample_index=0
        )
        self.assertIn('shap_values', explanation)
        self.assertIn('lime_explanation', explanation)

if __name__ == '__main__':
    unittest.main()
