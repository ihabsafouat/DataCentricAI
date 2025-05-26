from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.models import Variable
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.append(os.path.abspath("ingestion"))

from ingestion.ing import (
    download_and_extract_data,
    validate_and_process_data,
    upload_to_minio_and_save_metadata
)

def get_minio_config():
    """Get MinIO configuration from environment variables or Airflow Variables."""
    return {
        "endpoint_url": os.getenv("MINIO_ENDPOINT") or Variable.get("MINIO_ENDPOINT", "http://minio:9000"),
        "access_key": os.getenv("MINIO_ACCESS_KEY") or Variable.get("MINIO_ACCESS_KEY", "minioadmin"),
        "secret_key": os.getenv("MINIO_SECRET_KEY") or Variable.get("MINIO_SECRET_KEY", "minioadmin")
    }

def get_postgres_config():
    """Get PostgreSQL configuration from environment variables or Airflow Variables."""
    return {
        "host": os.getenv("POSTGRES_HOST") or Variable.get("POSTGRES_HOST", "postgres"),
        "dbname": os.getenv("POSTGRES_DB") or Variable.get("POSTGRES_DB", "datacentric"),
        "user": os.getenv("POSTGRES_USER") or Variable.get("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD") or Variable.get("POSTGRES_PASSWORD", "postgres")
    }

# Default arguments
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email": [os.getenv("AIRFLOW_EMAIL") or Variable.get("AIRFLOW_EMAIL", "your-email@example.com")],
    "email_on_failure": True,
    "email_on_retry": True,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

# DAG definition
with DAG(
    "data_ingestion_pipeline",
    default_args=default_args,
    description="Data ingestion pipeline with validation and versioning",
    schedule_interval="@daily",
    catchup=False,
    tags=["data", "ingestion"],
) as dag:

    # Define paths
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    validation_dir = Path("data/validation")

    # Start task
    start = DummyOperator(task_id="start")

    # Download and extract data
    download_task = PythonOperator(
        task_id="download_and_extract",
        python_callable=download_and_extract_data,
        op_kwargs={
            "target_dir": raw_dir
        }
    )

    # Validate and process data
    validate_task = PythonOperator(
        task_id="validate_and_process",
        python_callable=validate_and_process_data,
        op_kwargs={
            "raw_dir": raw_dir,
            "processed_dir": processed_dir,
            "validation_dir": validation_dir
        }
    )

    # Upload to MinIO and save metadata
    upload_task = PythonOperator(
        task_id="upload_and_save_metadata",
        python_callable=upload_to_minio_and_save_metadata,
        op_kwargs={
            "processed_dir": processed_dir,
            "minio_config": get_minio_config(),
            "postgres_config": get_postgres_config()
        }
    )

    # End task
    end = DummyOperator(task_id="end")

    # Define task dependencies
    start >> download_task >> validate_task >> upload_task >> end
