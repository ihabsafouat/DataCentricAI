CREATE DATABASE pet_mood;

\c pet_mood

CREATE TABLE image_metadata (
    id SERIAL PRIMARY KEY,
    image_name TEXT NOT NULL,
    image_url TEXT NOT NULL,
    label TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
