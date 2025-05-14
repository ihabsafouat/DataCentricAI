from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

def train_model():
    os.system("python src/training/train.py")

with DAG(
    dag_id="FMNIST_training_dag",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["training"],
) as dag:

    run_training = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    run_training
