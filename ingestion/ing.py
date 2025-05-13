# src/data_ingestion.py

import os
import tarfile
import requests
from pathlib import Path
import logging

import boto3
import psycopg2
from psycopg2.extras import execute_values

DATA_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
TARGET_DIR = Path("data/raw")
S3_BUCKET = "pet-images"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def download_and_extract_data(url=DATA_URL, target_dir=TARGET_DIR):
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / "images.tar.gz"

    if not archive_path.exists():
        logger.info(f"Downloading dataset from {url}...")
        response = requests.get(url, stream=True)
        with open(archive_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Download complete.")

    logger.info("Extracting dataset...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=target_dir)
    logger.info("Extraction complete.")


def upload_to_minio_and_save_metadata():
    # MinIO client
    s3 = boto3.client(
        's3',
        endpoint_url=os.environ['MINIO_ENDPOINT'],
        aws_access_key_id=os.environ['MINIO_ACCESS_KEY'],
        aws_secret_access_key=os.environ['MINIO_SECRET_KEY'],
    )

    try:
        s3.create_bucket(Bucket=S3_BUCKET)
        logger.info(f"Created bucket {S3_BUCKET}")
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
            logger.info("Bucket already exists")
        else:
            raise

    images_dir = TARGET_DIR / "images"
    metadata = []

    for img_path in images_dir.glob("*.jpg"):
        s3.upload_file(str(img_path), S3_BUCKET, img_path.name)
        file_size = img_path.stat().st_size
        metadata.append((img_path.name, file_size))

    logger.info(f"Uploaded {len(metadata)} files to MinIO")

    # Save metadata to PostgreSQL
    conn = psycopg2.connect(
        host=os.environ['POSTGRES_HOST'],
        dbname=os.environ['POSTGRES_DB'],
        user=os.environ['POSTGRES_USER'],
        password=os.environ['POSTGRES_PASSWORD']
    )

    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pet_images (
                id SERIAL PRIMARY KEY,
                filename TEXT,
                filesize INT
            )
        """)
        execute_values(cur,
            "INSERT INTO pet_images (filename, filesize) VALUES %s",
            metadata
        )
        conn.commit()
        logger.info("Metadata saved to PostgreSQL")
    conn.close()

def full_ingestion_pipeline():
    download_and_extract_data()
    upload_to_minio_and_save_metadata()

if __name__ == "__main__":
    full_ingestion_pipeline()
