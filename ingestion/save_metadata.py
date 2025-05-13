import psycopg2
from pathlib import Path

def save_metadata_to_postgres(directory, db_config):
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS image_metadata (
            filename TEXT PRIMARY KEY,
            size INTEGER
        );
    """)

    for file_path in Path(directory).glob("*.jpg"):
        cursor.execute(
            "INSERT INTO image_metadata (filename, size) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (file_path.name, file_path.stat().st_size)
        )

    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    save_metadata_to_postgres("data/raw/images", {
        "host": "localhost",
        "dbname": "datacentric",
        "user": "postgres",
        "password": "postgres"
    })
