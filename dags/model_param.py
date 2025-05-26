from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import mlflow
import os
import mlflow.keras
from tensorflow import keras
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
import numpy as np
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from ingestion.model_manager import ModelManager, RollbackCondition
from ingestion.model_monitor import ModelMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "CNN Fashion MNIST"
MODEL_PATH = "models/fashion_model.h5"
THRESHOLD_ACCURACY = 0.85
THRESHOLD_F1 = 0.80

# Fashion MNIST class names
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Define rollback conditions
rollback_conditions = [
    RollbackCondition(
        metric_name="accuracy",
        threshold=THRESHOLD_ACCURACY,
        comparison="less_than",
        window_size=3,
        consecutive_failures=2
    ),
    RollbackCondition(
        metric_name="f1_score",
        threshold=THRESHOLD_F1,
        comparison="less_than",
        window_size=3,
        consecutive_failures=2
    ),
    RollbackCondition(
        metric_name="precision",
        threshold=0.80,
        comparison="less_than",
        window_size=3,
        consecutive_failures=2
    )
]

# Initialize ModelManager
model_manager = ModelManager(
    experiment_name=EXPERIMENT_NAME,
    model_dir="models",
    registry_dir="model_registry",
    tracking_uri=MLFLOW_TRACKING_URI,
    performance_threshold=THRESHOLD_ACCURACY,
    rollback_conditions=rollback_conditions
)

default_args = {
    "owner": "airflow",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email_on_retry": True,
}

def check_metrics(**context):
    try:
        # Get latest model version
        latest_versions = model_manager.list_model_versions("fashion_mnist")
        if not latest_versions:
            raise ValueError("No model versions found")

        latest_model, metadata = model_manager.get_model_version("fashion_mnist", latest_versions[0]["version"])
        latest_accuracy = metadata["metrics"]["accuracy"]
        latest_f1 = metadata["metrics"].get("f1_score", 0.0)
        
        context['ti'].xcom_push(key='latest_accuracy', value=latest_accuracy)
        context['ti'].xcom_push(key='latest_f1', value=latest_f1)

        logger.info(f"Latest model metrics - Accuracy: {latest_accuracy}, F1: {latest_f1}")

        # Check rollback conditions
        should_rollback, target_version, condition_info = model_manager.check_rollback_conditions(
            "fashion_mnist",
            metadata["metrics"]
        )

        if should_rollback and target_version:
            logger.warning(f"Rollback condition met: {condition_info}")
            context['ti'].xcom_push(key='should_rollback', value=True)
            context['ti'].xcom_push(key='target_version', value=target_version)
            context['ti'].xcom_push(key='rollback_reason', value=str(condition_info))
            return "rollback_model"
        elif latest_accuracy < THRESHOLD_ACCURACY or latest_f1 < THRESHOLD_F1:
            logger.warning(f"Model performance below thresholds. Triggering retraining.")
            return "retrain_model"
        else:
            logger.info("Model performance is good. No retraining needed.")
            return "no_op"
    except Exception as e:
        logger.error(f"Error in check_metrics: {str(e)}")
        raise

def monitor_model(**context):
    try:
        # Get latest model version
        latest_versions = model_manager.list_model_versions("fashion_mnist")
        if not latest_versions:
            raise ValueError("No model versions found")

        latest_model, metadata = model_manager.get_model_version("fashion_mnist", latest_versions[0]["version"])
        
        # Load data for monitoring
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_test = x_test / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # Initialize model monitor
        monitor = ModelMonitor(
            model=latest_model,
            feature_names=[f"pixel_{i}" for i in range(784)],
            class_names=CLASS_NAMES,
            output_dir="monitoring"
        )
        
        # Get predictions
        predictions = latest_model.predict(x_test)
        
        # Monitor performance
        performance_results = monitor.monitor_performance(
            test_data=(x_test, y_test),
            predictions=predictions,
            metadata=metadata
        )
        
        # Check for data drift
        if len(latest_versions) > 1:
            previous_model, _ = model_manager.get_model_version("fashion_mnist", latest_versions[1]["version"])
            previous_predictions = previous_model.predict(x_test)
            
            drift_results = monitor.detect_data_drift(
                reference_data=previous_predictions,
                current_data=predictions
            )
            
            if any(result["drift_detected"] for result in drift_results.values()):
                logger.warning("Data drift detected!")
                context['ti'].xcom_push(key='drift_detected', value=True)
                context['ti'].xcom_push(key='drift_results', value=str(drift_results))
        
        # Generate explanations for some samples
        for i in range(3):  # Explain 3 random samples
            sample_idx = np.random.randint(0, len(x_test))
            explanation = monitor.explain_prediction(
                input_data=x_test,
                sample_index=sample_idx
            )
            logger.info(f"Generated explanation for sample {sample_idx}")
        
        # Get performance trends
        trends = monitor.get_performance_trends(days=30)
        logger.info("Generated performance trends")
        
        return "no_op"
        
    except Exception as e:
        logger.error(f"Error in monitor_model: {str(e)}")
        raise

def rollback_model(**context):
    try:
        should_rollback = context['ti'].xcom_pull(key='should_rollback')
        target_version = context['ti'].xcom_pull(key='target_version')
        rollback_reason = context['ti'].xcom_pull(key='rollback_reason')

        if should_rollback and target_version:
            success = model_manager.rollback_model(
                model_name="fashion_mnist",
                target_version=target_version,
                reason=rollback_reason
            )
            if success:
                logger.info(f"Successfully rolled back to version {target_version}")
                return "no_op"
            else:
                logger.warning("Rollback failed, triggering retraining")
                return "retrain_model"
        else:
            logger.warning("No rollback needed, triggering retraining")
            return "retrain_model"
    except Exception as e:
        logger.error(f"Error in rollback_model: {str(e)}")
        raise

def retrain_model():
    try:
        # Load and preprocess data
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        # Create and compile model
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train model
        history = model.fit(
            x_train, y_train,
            epochs=10,
            validation_split=0.2,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )

        # Evaluate model
        test_loss, test_accuracy = model.evaluate(x_test, y_test)
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred_classes, average='weighted')
        recall = recall_score(y_test, y_pred_classes, average='weighted')
        f1 = f1_score(y_test, y_pred_classes, average='weighted')

        # Register new model version
        metrics = {
            "accuracy": test_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        # Get current version number
        versions = model_manager.list_model_versions("fashion_mnist")
        new_version = str(len(versions) + 1) if versions else "1"

        # Register model
        model_manager.register_model(
            model=model,
            metrics=metrics,
            model_name="fashion_mnist",
            version=new_version,
            tags={
                "framework": "tensorflow",
                "dataset": "fashion_mnist",
                "architecture": "cnn"
            }
        )

        # Monitor model performance with auto-rollback
        performance_log = model_manager.monitor_model_performance(
            model=model,
            test_data=(x_test, y_test),
            model_name="fashion_mnist",
            version=new_version,
            auto_rollback=True
        )

        # If this is not the first version, set up A/B testing
        if len(versions) > 0:
            previous_model, _ = model_manager.get_model_version("fashion_mnist", versions[0]["version"])
            
            # Compare models
            comparison = model_manager.compare_models(
                model_a=model,
                model_b=previous_model,
                test_data=(x_test, y_test)
            )
            
            # If new model is better, set up A/B testing
            if comparison["model_a"]["accuracy"] > comparison["model_b"]["accuracy"]:
                model_manager.setup_ab_test(
                    model_a=model,
                    model_b=previous_model,
                    model_a_name=f"fashion_mnist_v{new_version}",
                    model_b_name=f"fashion_mnist_v{versions[0]['version']}",
                    traffic_split=0.5
                )

        logger.info(f"Model training completed. Metrics - Accuracy: {test_accuracy}, F1: {f1}")

    except Exception as e:
        logger.error(f"Error in retrain_model: {str(e)}")
        raise

def no_op():
    logger.info("No action needed. Model performance is sufficient.")

with DAG(
    "check_mlflow_model_accuracy",
    default_args=default_args,
    description="Check model metrics and trigger retraining if needed",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlflow", "monitoring"],
) as dag:

    check_metrics_task = PythonOperator(
        task_id="check_metrics",
        python_callable=check_metrics,
        provide_context=True,
    )

    monitor_task = PythonOperator(
        task_id="monitor_model",
        python_callable=monitor_model,
        provide_context=True,
    )

    rollback_task = PythonOperator(
        task_id="rollback_model",
        python_callable=rollback_model,
        provide_context=True,
    )

    retrain_task = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model,
    )

    noop_task = PythonOperator(
        task_id="no_op",
        python_callable=no_op,
    )

    check_metrics_task >> [rollback_task, retrain_task, noop_task]
    rollback_task >> [retrain_task, noop_task]
    [retrain_task, noop_task] >> monitor_task
