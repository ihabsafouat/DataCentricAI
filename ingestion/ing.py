# src/data_ingestion.py

import os
import tarfile
import requests
from pathlib import Path
import logging
import boto3
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
from .data_validator import validate_data_pipeline, DataValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
TARGET_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
VALIDATION_DIR = Path("data/validation")
S3_BUCKET = "pet-images"

def download_and_extract_data(url=DATA_URL, target_dir=TARGET_DIR):
    """Download and extract the dataset."""
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        archive_path = target_dir / "images.tar.gz"

        if not archive_path.exists():
            logger.info(f"Downloading dataset from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise exception for bad status codes
            
            with open(archive_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Download complete.")

        logger.info("Extracting dataset...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=target_dir)
        logger.info("Extraction complete.")
        
        return True
    except Exception as e:
        logger.error(f"Error in download_and_extract_data: {str(e)}")
        raise

def validate_and_process_data(raw_dir=TARGET_DIR, processed_dir=PROCESSED_DIR, validation_dir=VALIDATION_DIR):
    """Validate and process the downloaded data."""
    try:
        # Run data validation
        logger.info("Starting data validation...")
        is_valid = validate_data_pipeline(raw_dir, validation_dir)
        
        if not is_valid:
            logger.error("Data validation failed. Check validation report for details.")
            return False
            
        # Process valid images
        processed_dir.mkdir(parents=True, exist_ok=True)
        validator = DataValidator()
        
        for img_path in raw_dir.glob("*.jpg"):
            result = validator.validate_image(img_path)
            if result["valid"]:
                # Copy valid images to processed directory
                processed_path = processed_dir / img_path.name
                img_path.rename(processed_path)
                
        logger.info("Data validation and processing complete.")
        return True
    except Exception as e:
        logger.error(f"Error in validate_and_process_data: {str(e)}")
        raise

def upload_to_minio_and_save_metadata(processed_dir=PROCESSED_DIR, minio_config=None, postgres_config=None):
    """Upload processed data to MinIO and save metadata to PostgreSQL."""
    try:
        # Use provided configs or environment variables
        minio_config = minio_config or {
            "endpoint_url": os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
            "access_key": os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            "secret_key": os.getenv("MINIO_SECRET_KEY", "minioadmin")
        }

        postgres_config = postgres_config or {
            "host": os.getenv("POSTGRES_HOST", "postgres"),
            "dbname": os.getenv("POSTGRES_DB", "datacentric"),
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", "postgres")
        }

        # MinIO client
        s3 = boto3.client(
            's3',
            endpoint_url=minio_config["endpoint_url"],
            aws_access_key_id=minio_config["access_key"],
            aws_secret_access_key=minio_config["secret_key"]
        )

        # Create bucket if it doesn't exist
        try:
            s3.create_bucket(Bucket=S3_BUCKET)
            logger.info(f"Created bucket {S3_BUCKET}")
        except s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
                logger.info("Bucket already exists")
            else:
                raise

        # Upload files and collect metadata
        metadata = []
        for img_path in processed_dir.glob("*.jpg"):
            try:
                # Upload to MinIO
                s3.upload_file(str(img_path), S3_BUCKET, img_path.name)
                
                # Collect metadata
                file_size = img_path.stat().st_size
                upload_time = datetime.now().isoformat()
                metadata.append((
                    img_path.name,
                    file_size,
                    upload_time
                ))
                logger.info(f"Uploaded {img_path.name} to MinIO")
            except Exception as e:
                logger.error(f"Error uploading {img_path.name}: {str(e)}")
                continue

        # Save metadata to PostgreSQL
        conn = psycopg2.connect(**postgres_config)

        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS image_metadata (
                    id SERIAL PRIMARY KEY,
                    filename TEXT UNIQUE,
                    filesize INT,
                    upload_time TIMESTAMP,
                    version_hash TEXT
                )
            """)
            
            # Get data version hash
            validator = DataValidator()
            version_hash = validator.get_data_hash(processed_dir)
            
            # Insert metadata
            execute_values(cur,
                """
                INSERT INTO image_metadata (filename, filesize, upload_time, version_hash)
                VALUES %s
                ON CONFLICT (filename) DO UPDATE
                SET filesize = EXCLUDED.filesize,
                    upload_time = EXCLUDED.upload_time,
                    version_hash = EXCLUDED.version_hash
                """,
                [(m[0], m[1], m[2], version_hash) for m in metadata]
            )
            conn.commit()
            logger.info("Metadata saved to PostgreSQL")
        conn.close()
        
        return True
    except Exception as e:
        logger.error(f"Error in upload_to_minio_and_save_metadata: {str(e)}")
        raise

def full_ingestion_pipeline():
    """Run the complete data ingestion pipeline."""
    try:
        # Step 1: Download and extract data
        if not download_and_extract_data():
            return False
            
        # Step 2: Validate and process data
        if not validate_and_process_data():
            return False
            
        # Step 3: Upload to MinIO and save metadata
        if not upload_to_minio_and_save_metadata():
            return False
            
        logger.info("Data ingestion pipeline completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in full_ingestion_pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    full_ingestion_pipeline()
