import os
import boto3
from pathlib import Path

def upload_directory_to_minio(directory, bucket_name, minio_config):
    s3 = boto3.client(
        's3',
        endpoint_url=minio_config["endpoint_url"],
        aws_access_key_id=minio_config["access_key"],
        aws_secret_access_key=minio_config["secret_key"]
    )

    # Ensure bucket exists
    try:
        s3.head_bucket(Bucket=bucket_name)
    except:
        s3.create_bucket(Bucket=bucket_name)

    for file_path in Path(directory).glob("*.jpg"):
        key = file_path.name
        s3.upload_file(str(file_path), bucket_name, key)
        print(f"Uploaded {key} to {bucket_name}")

if __name__ == "__main__":
    upload_directory_to_minio(
        "data/raw/images",
        "pet-images",
        {
            "endpoint_url": "http://localhost:9000",
            "access_key": "minioadmin",
            "secret_key": "minioadmin"
        }
    )
