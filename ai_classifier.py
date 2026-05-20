from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory


def create_classifier(input_shape=(224, 224, 3), num_classes: int = 7) -> tf.keras.Model:
    """Create a simple transfer learning classifier model."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_training_dataset(dataset_folder: Path, image_size=(224, 224), batch_size=8):
    """Load labeled images from folder subdirectories."""
    dataset_folder = Path(dataset_folder)
    return image_dataset_from_directory(
        str(dataset_folder),
        image_size=image_size,
        batch_size=batch_size,
    )


def train_model(
    dataset_folder: Path,
    model_path: Path = Path("models/checkpoints/radar_classifier.h5"),
    epochs: int = 5,
    batch_size: int = 8,
) -> tf.keras.Model:
    """Train a radar image classifier using data in labeled folders."""
    train_ds = load_training_dataset(dataset_folder, batch_size=batch_size)
    model = create_classifier(num_classes=len(train_ds.class_names))
    model.fit(train_ds, epochs=epochs)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    return model


def preprocess_image(image_path: Path, image_size=(224, 224)) -> np.ndarray:
    """Load and preprocess a single image for prediction."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size)
    array = np.array(image) / 255.0
    return np.expand_dims(array, axis=0)


def predict_image(model: tf.keras.Model, image_path: Path, class_names: List[str]) -> dict:
    """Predict labels and confidence values for a radar image."""
    tensor = preprocess_image(image_path)
    predictions = model.predict(tensor)
    score = predictions[0]
    results = {
        class_names[i]: float(score[i]) for i in range(len(class_names))
    }
    best_index = int(np.argmax(score))
    results["best_label"] = class_names[best_index]
    results["confidence"] = float(score[best_index])
    return results


def classify_image_file(
    image_path: Path,
    class_names: List[str],
    model_path: Path = Path("models/checkpoints/radar_classifier.h5"),
) -> dict:
    """Classify a radar image using the saved model."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Radar image not found: {image_path}")

    if Path(model_path).exists():
        model = tf.keras.models.load_model(model_path)
    else:
        model = create_classifier(num_classes=len(class_names))

    return predict_image(model, image_path, class_names)


def classify_uploaded_image(uploaded_file, class_names: List[str], model_path: Path = Path("models/checkpoints/radar_classifier.h5")) -> dict:
    """Classify a file uploaded through Streamlit."""
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    array = np.array(image) / 255.0
    tensor = np.expand_dims(array, axis=0)

    if Path(model_path).exists():
        model = tf.keras.models.load_model(model_path)
    else:
        model = create_classifier(num_classes=len(class_names))

    predictions = model.predict(tensor)
    score = predictions[0]
    best_index = int(np.argmax(score))
    results = {class_names[i]: float(score[i]) for i in range(len(class_names))}
    results["best_label"] = class_names[best_index]
    results["confidence"] = float(score[best_index])
    return results
