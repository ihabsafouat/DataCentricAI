import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import os
import boto3







# Load environment vars (Docker Compose passes them)
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MODEL_PATH = "models/fashion_model.h5"
BUCKET_NAME = "models"








def build_model():
# Build the CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])

# Compile the model
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model


def upload_to_minio(file_path):
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )
    # Ensure bucket exists
    buckets = s3.list_buckets()
    if BUCKET_NAME not in [b["Name"] for b in buckets.get("Buckets", [])]:
        s3.create_bucket(Bucket=BUCKET_NAME)
    s3.upload_file(file_path, BUCKET_NAME, os.path.basename(file_path))
    print(f"✅ Model uploaded to MinIO bucket '{BUCKET_NAME}'")


def main():
    # Load and preprocess data
    (train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    model = build_model()
    # Train the model
    epochs = int(os.getenv("EPOCHS", 5))
    model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))


    loss, acc = model.evaluate(test_images, test_labels)
    print(f"✅ Test accuracy: {acc:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)


    # Upload to MinIO
    upload_to_minio(MODEL_PATH)


if __name__ == "__main__":
    main()
    print("✅ Model saved to models/fashion_mnist_cnn.h5")
