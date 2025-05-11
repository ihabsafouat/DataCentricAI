# src/data_ingestion.py

import os
import zipfile
import requests
from pathlib import Path

DATA_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
TARGET_DIR = Path("data/raw")

def download_and_extract_data(url=DATA_URL, target_dir=TARGET_DIR):
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / "images.tar.gz"

    if not archive_path.exists():
        print(f"Downloading dataset from {url}...")
        response = requests.get(url, stream=True)
        with open(archive_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

    # Extract the tar.gz
    print("Extracting dataset...")
    import tarfile
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=target_dir)
    print("Extraction complete.")

if __name__ == "__main__":
    download_and_extract_data()