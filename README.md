# 🐶 Pet Mood Detector - Data-Centric AI Project

This project implements a **Pet Mood Detection System** inspired by
the paper _"Data-Centric AI: A Systematic Treatment"_ by Andrew Ng
et al. It leverages MLOps best practices to iteratively improve data
quality and model performance.

---

## 📌 Project Summary

- **Goal**:
- Predict a pet's mood (e.g., `happy`, `sad`,`neutral`) from an image.
- **Dataset**: Oxford-IIIT Pet Dataset (`images.tar.gz`)
-**Approach**:
  - Download & validate data
  - Upload to MinIO (S3)
  - Store metadata in PostgreSQL
  - Train & evaluate a mood classification model

---

## 🧱 Project Structure
```
mlops-data-centric-ai/
│
|      ├── dags/ # Airflow DAGs for data ingestion
|      ├── data/
|      ├── ingestion/
|      ├── models/
|      ├── src/
|      |     ├── data_ingestion.py # Download & extract data
│      |     ├──label_breeds.py # Use HF model to predict breed
│      |     ├──upload_to_minio.py # Upload data to MinIO
│      |     ├── save_metadata.py #Save metadata to PostgreSQL
│      └── mood_labelling.py # Manual
|                 /heuristic mood labels 
|      ├── Dockerfile
|      ├── docker-compose.yml
|      ├──requirements.txt
└── README.md

```

---

## ⚙️ Setup & Run

### 1. Clone the Repo

```bash git clone
https://github.com/your-username/mlops-data-centric-ai.git cd
mlops-data-centric-ai 2. Start the MLOps Stack bash Copy Edit
docker-compose up --build Access:

Airflow UI → http://localhost:8080

MinIO Console → http://localhost:9001

```

# 3. Run Airflow DAG for Ingestion Enable and trigger the
#### data_ingestion_dag in the Airflow UI to:

#### Download & extract data

#### Upload to MinIO

#### Save metadata in PostgreSQL

# 🔍 Labeling & Mood Detection Breed Labeling Run:

``` 
bash 
 
Copy Edit python src/label_breeds.py 

Mood Labeling Create a CSV: mood.csv with columns: image, breed, mood

Labels can be created manually or with weak heuristics

```

# 📊 Model Training (Coming Soon) A separate DAG or script will:

#### Load labeled images

#### Train a mood classification model (e.g. ViT or CNN)

#### Evaluate accuracy, precision, recall

#### Store results in PostgreSQL

## 📌 Paper Reference This project is inspired by:

### Data-Centric AI: A Systematic Treatment Andrew Ng, et al. arXiv:2205.01580

# 🛠️ Tech Stack 🐍 Python 3.10

#### ☁️ MinIO (S3-compatible)

#### 🐘 PostgreSQL

#### 📅 Apache Airflow 2.8

#### 🤗 Hugging Face Transformers

#### 🐳 Docker Compose

#### ✅ TODO Data ingestion DAG

#### Upload to MinIO

#### Metadata to PostgreSQL

#### Breed labeling with HF model

#### Mood label propagation

#### Model training + evaluation

#### Model registry + inference pipeline

# 👥 Contributing Feel free to fork and submit PRs! Contributions to mood
labeling, evaluation pipelines, or metrics visualization are welcome.

# 📄 License MIT License -- feel free to use and adapt...


