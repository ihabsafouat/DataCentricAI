from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

sys.path.append(os.path.abspath("src"))

from ingestion.ing import download_and_extract_data
from ingestion.upload_to_minio import upload_directory_to_minio
from ingestion.save_metadata import save_metadata_to_postgres

default_args = {
    "start_date": datetime(2024, 1, 1),
}

with DAG("data_ingestion_pipeline", schedule_interval=None, default_args=default_args, catchup=False) as dag:

    t1 = PythonOperator(
        task_id="download_and_extract",
        python_callable=download_and_extract_data
    )

    t2 = PythonOperator(
        task_id="upload_to_minio",
        python_callable=upload_directory_to_minio,
        op_kwargs={
            "directory": "data/raw/images",
            "bucket_name": "pet-images",
            "minio_config": {
                "endpoint_url": "http://minio:9000",
                "access_key": "minioadmin",
                "secret_key": "minioadmin"
            }
        }
    )

    t3 = PythonOperator(
        task_id="save_metadata_to_postgres",
        python_callable=save_metadata_to_postgres,
        op_kwargs={
            "directory": "data/raw/images",
            "db_config": {
                "host": "postgres",
                "dbname": "datacentric",
                "user": "postgres",
                "password": "postgres"
            }
        }
    )

    t1 >> t2 >> t3
